import ccxt
import pandas as pd
import numpy as np
import sqlite3
import os
import time
import requests
import json
import logging
import threading
from datetime import datetime, timezone
from dotenv import load_dotenv
import pytz
import pandas_ta as ta
from scipy.stats import linregress

# Suppress pandas_ta warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas_ta")

# Configuración e Inicialización
load_dotenv()
DB_NAME = "trading_real.db"

# Lista de monedas establecidas
ESTABLISHED_COINS = ['BTC', 'ETH', 'ADA']
SELECTED_CRYPTOS = [f"{coin}/USDT" for coin in ESTABLISHED_COINS]

def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            timestamp TEXT NOT NULL,
            trade_id TEXT NOT NULL,
            rsi REAL,
            adx REAL,
            atr REAL,
            relative_volume REAL,
            divergence TEXT,
            bb_position TEXT,
            confidence REAL,
            has_macd_crossover INTEGER,
            candles_since_crossover INTEGER,
            volume_trend TEXT,
            price_trend TEXT,
            short_volume_trend TEXT,
            support_level REAL,
            spread REAL,
            imbalance REAL,
            depth REAL,
            status TEXT DEFAULT 'open'
        )
    ''')
    conn.commit()
    conn.close()

initialize_db()

exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY_REAL"),
    'secret': os.getenv("BINANCE_SECRET_KEY_REAL"),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logger = logging.getLogger("inversion_binance")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(os.path.expanduser("~/hobbies/trading.log"))
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)
logger.info("Prueba de escritura en trading.log al iniciar")

# Constantes
MAX_OPEN_TRADES = 10
MIN_NOTIONAL = 10
RSI_THRESHOLD = 70
ADX_THRESHOLD = 25
VOLUME_GROWTH_THRESHOLD = 0.5

# Cache de decisiones
decision_cache = {}
CACHE_EXPIRATION = 300

def get_market_sentiment():
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        response.raise_for_status()
        data = response.json()
        if data['data']:
            score = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            return score, classification
        return 0, "Neutral"
    except Exception as e:
        logging.error(f"Error fetching market sentiment: {e}")
        return 0, "Neutral"

def detect_support_level(data, price_series, window=15, max_threshold_multiplier=2.5):
    if len(price_series) < window:
        logging.warning(f"Series too short for {price_series.name}: {len(price_series)} < {window}")
        return None

    recent_prices = price_series[-window:]
    local_mins = [
        recent_prices.iloc[i] for i in range(1, len(recent_prices) - 1)
        if recent_prices.iloc[i] < recent_prices.iloc[i - 1] and recent_prices.iloc[i] < recent_prices.iloc[i + 1]
    ]

    if not local_mins:
        logging.warning(f"No local minima found in the last {window} candles for {price_series.name}")
        return None

    support_level = min(local_mins)
    current_price = price_series.iloc[-1]

    atr_value = None
    for tf in ['1h', '4h', '1d']:
        if tf in data and not data[tf].empty and len(data[tf]) >= 14:
            df = data[tf].sort_index().drop_duplicates(keep='first').dropna(subset=['high', 'low', 'close'])
            atr_series = ta.atr(df['high'], df['low'], df['close'], length=14).ffill().bfill()
            if not atr_series.isna().all():
                atr_value = atr_series.iloc[-1]
                break

    if atr_value is None:
        atr_value = price_series.pct_change().std() * current_price if len(price_series) > 10 else current_price * 0.02
        logging.warning(f"No ATR calculado para {price_series.name}, usando volatilidad estimada: {atr_value}")

    threshold = 1 + (atr_value * max_threshold_multiplier / current_price) if current_price > 0 else 1.02
    threshold = min(threshold, 1.03)

    logging.debug(f"Soporte detectado: precio={current_price}, soporte={support_level}, umbral={threshold:.3f}")
    return support_level if current_price <= support_level * threshold else None

def detect_bullish_candlestick(data, timeframe='1h'):
    if timeframe not in data or data[timeframe].empty or len(data[timeframe]) < 2:
        return False

    df = data[timeframe].iloc[-2:]
    prev_candle = df.iloc[0]
    curr_candle = df.iloc[1]

    body_size = abs(curr_candle['close'] - curr_candle['open'])
    lower_wick = curr_candle['open'] - curr_candle['low'] if curr_candle['close'] > curr_candle['open'] else curr_candle['close'] - curr_candle['low']
    hammer = (body_size < lower_wick * 2 and curr_candle['close'] > curr_candle['open'])

    bullish_engulfing = (prev_candle['close'] < prev_candle['open'] and 
                         curr_candle['close'] > curr_candle['open'] and 
                         curr_candle['open'] <= prev_candle['close'] and 
                         curr_candle['close'] >= prev_candle['open'])

    return hammer or bullish_engulfing

def calculate_volume_behavior(data, timeframe='1h', window=5):
    if timeframe not in data or data[timeframe].empty or len(data[timeframe]) < window:
        return "insufficient_data"

    volume_series = data[timeframe]['volume'].iloc[-window:]
    last_volume = volume_series.iloc[-1]
    avg_volume = volume_series.iloc[:-1].mean()

    if avg_volume == 0:
        return "insufficient_data"
    
    if last_volume > avg_volume * 1.2:
        return "increasing"
    elif last_volume < avg_volume * 0.8:
        return "decreasing"
    else:
        return "stable"

def calculate_short_volume_trend(data, window=3):
    if '15m' not in data or data['15m'].empty or len(data['15m']) < window:
        logging.warning("Datos de 15m no disponibles o insuficientes para calcular short_volume_trend")
        return "insufficient_data"

    volume_series = data['15m']['volume']
    
    if len(volume_series) < window:
        logging.warning(f"Series de volumen 15m demasiado corta: {len(volume_series)} < {window}")
        return "insufficient_data"
    
    last_volume = volume_series.iloc[-1]
    avg_volume = volume_series[-window:-1].mean()
    
    if avg_volume == 0:
        logging.warning("Promedio de volumen es 0, no se puede calcular short_volume_trend")
        return "insufficient_data"
    
    if last_volume > avg_volume * 1.10:
        return "increasing"
    elif last_volume < avg_volume * 0.90:
        return "decreasing"
    else:
        return "stable"

def fetch_order_book_data(symbol, limit=100):
    try:
        order_book = exchange.fetch_order_book(symbol, limit=limit)
        bids = order_book['bids']
        asks = order_book['asks']
        spread = asks[0][0] - bids[0][0] if bids and asks else None
        bid_volume = sum([volume for _, volume in bids])
        ask_volume = sum([volume for _, volume in asks])
        imbalance = bid_volume / ask_volume if ask_volume > 0 else None
        current_price = fetch_price(symbol) or 1
        depth = (bid_volume + ask_volume) * current_price  # Convert to USDT
        logging.info(f"Order book data for {symbol}: spread={spread}, imbalance={imbalance}, depth={depth}")
        print(f"Order book for {symbol}: Spread={spread}, Imbalance={imbalance}, Depth={depth}")
        return {
            'spread': spread,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            'depth': depth
        }
    except Exception as e:
        logging.error(f"Error al obtener order book para {symbol}: {e}")
        print(f"Error fetching order book for {symbol}: {e}")
        return None

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram no configurado.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=3)
        logging.info(f"Telegram message sent: {message[:50]}...")
    except Exception as e:
        logging.error(f"Error al enviar a Telegram: {e}")

def get_colombia_timestamp():
    colombia_tz = pytz.timezone("America/Bogota")
    return datetime.now(colombia_tz).strftime("%Y-%m-%d %H:%M:%S")

def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        logging.info(f"Price for {symbol}: {price}")
        print(f"Current price for {symbol}: {price}")
        return price
    except Exception as e:
        logging.error(f"Error al obtener precio de {symbol}: {e}")
        print(f"Error fetching price for {symbol}: {e}")
        return None

def fetch_volume(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        volume = ticker['quoteVolume']
        logging.info(f"Volume for {symbol}: {volume}")
        print(f"24h volume for {symbol}: {volume}")
        return volume
    except Exception as e:
        logging.error(f"Error al obtener volumen de {symbol}: {e}")
        print(f"Error fetching volume for {symbol}: {e}")
        return None

def fetch_and_prepare_data(symbol, atr_length=7, rsi_length=14, bb_length=20, roc_length=7, limit=100):
    timeframes = ['15m', '1h', '4h', '1d']
    data = {}
    logging.debug(f"Inicio de fetch_and_prepare_data para {symbol}")
    print(f"Fetching and preparing data for {symbol}")

    for tf in timeframes:
        try:
            logging.debug(f"Iniciando fetch_ohlcv para {symbol} en {tf} con limit={limit}")
            print(f"Fetching OHLCV for {symbol} on {tf} timeframe")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if not ohlcv or len(ohlcv) == 0:
                logging.warning(f"Datos vacíos para {symbol} en {tf}")
                print(f"No data for {symbol} on {tf}")
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            logging.debug(f"DataFrame para {symbol} en {tf} creado con {len(df)} velas.")
            print(f"Data frame created for {symbol} on {tf} with {len(df)} candles")

            if len(df) < max(atr_length, rsi_length, bb_length, roc_length, 15):
                logging.warning(f"Datos insuficientes (<{max(atr_length, rsi_length, bb_length, roc_length, 15)} velas) para {symbol} en {tf}")
                print(f"Insufficient data for {symbol} on {tf}")
                continue

            expected_freq = pd.Timedelta('15m') if tf == '15m' else pd.Timedelta('1h') if tf == '1h' else pd.Timedelta('4h') if tf == '4h' else pd.Timedelta('1d')
            expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq)
            if len(expected_index) > len(df.index):
                gap_ratio = (len(expected_index) - len(df.index)) / len(expected_index)
                if gap_ratio > 0.1:
                    logging.error(f"Demasiados gaps en {symbol} en {tf} (ratio: {gap_ratio:.2f}), omitiendo timeframe")
                    print(f"Too many gaps in data for {symbol} on {tf}, skipping")
                    continue
                df = df.reindex(expected_index, method='ffill').dropna(how='all')

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    df[col] = df[col].ffill().bfill()

            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep='first')]
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()

            if len(df) >= atr_length:
                df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length).ffill().bfill()
            else:
                df['ATR'] = np.nan

            if len(df) >= rsi_length:
                df['RSI'] = ta.rsi(df['close'], length=rsi_length).ffill().bfill()
            else:
                df['RSI'] = np.nan

            if len(df) >= bb_length:
                bb = ta.bbands(df['close'], length=bb_length, std=2)
                df['BB_upper'] = bb.get(f'BBU_{bb_length}_2.0', pd.Series(np.nan, index=df.index))
                df['BB_middle'] = bb.get(f'BBM_{bb_length}_2.0', pd.Series(np.nan, index=df.index))
                df['BB_lower'] = bb.get(f'BBL_{bb_length}_2.0', pd.Series(np.nan, index=df.index))
            else:
                df['BB_upper'] = df['BB_middle'] = df['BB_lower'] = np.nan

            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is not None and not macd.empty:
                df['MACD'] = macd.get('MACD_12_26_9', pd.Series(np.nan, index=df.index))
                df['MACD_signal'] = macd.get('MACDs_12_26_9', pd.Series(np.nan, index=df.index))
            else:
                df['MACD'] = df['MACD_signal'] = np.nan

            if len(df) >= roc_length:
                df['ROC'] = ta.roc(df['close'], length=5 if tf == '4h' else roc_length).ffill().bfill()
            else:
                df['ROC'] = np.nan

            if len(df) >= 7:
                df['SMA_7'] = ta.sma(df['close'], length=7)
                df['tendencia_alcista'] = df['close'] > df['SMA_7']
            else:
                df['SMA_7'] = np.nan
                df['tendencia_alcista'] = False

            if len(df) >= 10:
                last_10_volume = df['volume'].iloc[-10:]
                slope, _, _, _, _ = linregress(range(10), last_10_volume)
                df['volume_slope'] = slope
                df['volumen_creciente'] = slope > 0.01
            else:
                df['volume_slope'] = np.nan
                df['volumen_creciente'] = False

            if tf == '15m':
                if len(df) >= 20:
                    df['EMA_7'] = ta.ema(df['close'], length=7)
                    df['EMA_20'] = ta.ema(df['close'], length=20)
                    df['ema_crossover'] = (df['EMA_7'] > df['EMA_20']) & (df['EMA_7'].shift(1) <= df['EMA_20'].shift(1))
                else:
                    df['EMA_7'] = df['EMA_20'] = np.nan
                    df['ema_crossover'] = False

                if len(df) >= 14:
                    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
                    df['stoch_k'] = stoch['STOCHk_14_3_3']
                    df['stoch_d'] = stoch['STOCHd_14_3_3']
                    df['stoch_crossover'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & (df['stoch_k'] > 20)
                else:
                    df['stoch_k'] = df['stoch_d'] = np.nan
                    df['stoch_crossover'] = False

                if len(df) >= 14:
                    df['ATR_trend'] = df['ATR'].pct_change().rolling(3).mean()
                    df['atr_increasing'] = df['ATR_trend'] > 0.05
                else:
                    df['ATR_trend'] = np.nan
                    df['atr_increasing'] = False

                if len(df) >= 2:
                    df['OBV'] = ta.obv(df['close'], df['volume'])
                    df['obv_slope'] = df['OBV'].diff().rolling(3).mean()
                    df['obv_increasing'] = df['obv_slope'] > 0
                else:
                    df['OBV'] = df['obv_slope'] = np.nan
                    df['obv_increasing'] = False

            data[tf] = df
            logging.info(f"Indicators for {symbol} on {tf}: RSI={df['RSI'].iloc[-1] if 'RSI' in df else 'N/A'}, ATR={df['ATR'].iloc[-1] if 'ATR' in df else 'N/A'}")
            print(f"Indicators for {symbol} on {tf}: RSI={df['RSI'].iloc[-1] if 'RSI' in df else 'N/A'}, ATR={df['ATR'].iloc[-1] if 'ATR' in df else 'N/A'}")

        except Exception as e:
            logging.error(f"Error procesando {symbol} en {tf}: {e}")
            print(f"Error processing {symbol} on {tf}: {e}")
            continue

    if not data:
        logging.error(f"No se obtuvieron datos válidos para {symbol} en ningún timeframe")
        print(f"No valid data for {symbol} in any timeframe")
        return None, None

    has_valid_indicators = any(
        not df[['RSI', 'ATR', 'MACD']].isna().all().all()
        for df in data.values()
    )
    if not has_valid_indicators:
        logging.error(f"No hay indicadores válidos para {symbol}")
        print(f"No valid indicators for {symbol}")
        return None, None

    for tf in ['15m', '1h', '4h', '1d']:
        if tf in data and not data[tf].empty:
            price_series = data[tf]['close']
            logging.debug(f"Seleccionada serie de precios para soporte: {tf} con {len(price_series)} velas")
            print(f"Price series selected for support on {tf} with {len(price_series)} candles")
            break
    else:
        logging.error(f"No se pudieron obtener series de precios para {symbol}")
        print(f"No price series for {symbol}")
        return None, None

    return data, price_series

def calculate_adx(df):
    try:
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        adx_value = adx['ADX_14'].iloc[-1] if not pd.isna(adx['ADX_14'].iloc[-1]) else None
        logging.info(f"ADX calculated: {adx_value}")
        print(f"ADX: {adx_value}")
        return adx_value
    except Exception as e:
        logging.error(f"Error al calcular ADX: {e}")
        print(f"Error calculating ADX: {e}")
        return None

def detect_momentum_divergences(price_series, rsi_series):
    try:
        price = np.array(price_series)
        rsi = np.array(rsi_series)
        divergences = []
        window = 5
        for i in range(window, len(price) - window):
            if (price[i] > max(price[i-window:i]) and price[i] > max(price[i+1:i+window+1]) and
                rsi[i] < rsi[i-window]):
                divergences.append(("bearish", i))
            elif (price[i] < min(price[i-window:i]) and price[i] < min(price[i+1:i+window+1]) and
                  rsi[i] > rsi[i-window]):
                divergences.append(("bullish", i))
        result = "bullish" if any(d[0] == "bullish" for d in divergences) else "bearish" if any(d[0] == "bearish" for d in divergences) else "none"
        logging.info(f"Momentum divergences: {result}")
        print(f"Momentum divergences: {result}")
        return result
    except Exception as e:
        logging.error(f"Error en divergencias: {e}")
        print(f"Error in divergences: {e}")
        return "none"

def get_bb_position(price, bb_upper, bb_middle, bb_lower):
    if price is None or bb_upper is None or bb_lower is None:
        return "unknown"
    if price > bb_upper:
        return "above_upper"
    elif price < bb_lower:
        return "below_lower"
    elif price > bb_middle:
        return "above_middle"
    else:
        return "below_middle"

def has_recent_macd_crossover(macd_series, signal_series, lookback=5):
    if len(macd_series) < 2 or len(signal_series) < 2:
        return False, None
    for i in range(-1, -lookback-1, -1):
        if i < -len(macd_series) or i-1 < -len(macd_series):
            break
        if macd_series.iloc[i-1] <= signal_series.iloc[i-1] and macd_series.iloc[i] > signal_series.iloc[i]:
            logging.info(f"Recent MACD crossover found, candles since: {abs(i)}")
            print(f"Recent MACD crossover, candles since: {abs(i)}")
            return True, abs(i)
    return False, None

def get_open_trades():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM transactions_new WHERE action='buy' AND status='open'")
    count = cursor.fetchone()[0]
    cursor.execute("SELECT symbol, timestamp, trade_id FROM transactions_new WHERE action='buy' AND status='open'")
    trades = cursor.fetchall()
    logging.info(f"Operaciones abiertas: {count}. Detalles: {trades}")
    print(f"Open trades: {count}. Details: {trades}")
    conn.close()
    return count

def execute_order_buy(symbol, amount, indicators, confidence):
    try:
        print(f"Executing buy order for {symbol}, amount={amount}, confidence={confidence}")
        order = exchange.create_market_buy_order(symbol, amount)
        price = order.get("price", fetch_price(symbol))
        executed_amount = order.get("filled", amount)
        if price is None:
            logging.error(f"No se pudo obtener precio para {symbol} después de la orden")
            print(f"Error: No price after order for {symbol}")
            send_telegram_message(f"❌ *Error en Compra* `{symbol}`\nNo se obtuvo precio tras la orden.")
            return None
        timestamp = datetime.now(timezone.utc).isoformat()
        trade_id = f"{symbol}_{timestamp.replace(':', '-')}"
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions_new (
                symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr, 
                relative_volume, divergence, bb_position, confidence, has_macd_crossover, 
                candles_since_crossover, volume_trend, price_trend, short_volume_trend, 
                support_level, spread, imbalance, depth, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, "buy", price, executed_amount, timestamp, trade_id,
            indicators.get('rsi'), indicators.get('adx'), indicators.get('atr'),
            indicators.get('relative_volume'), indicators.get('divergence'), indicators.get('bb_position'), 
            confidence, 1 if indicators.get('has_macd_crossover') else 0, 
            indicators.get('candles_since_crossover'),
            indicators.get('volume_trend'), indicators.get('price_trend'), indicators.get('short_volume_trend'),
            indicators.get('support_level'), indicators.get('spread'), indicators.get('imbalance'), 
            indicators.get('depth'), 'open'
        ))
        conn.commit()
        conn.close()
        
        logging.info(f"Compra ejecutada: {symbol} a {price} por {executed_amount} (ID: {trade_id})")
        print(f"Buy executed: {symbol} at {price} for {executed_amount} (ID: {trade_id})")
        send_telegram_message(f"✅ *Compra* `{symbol}`\nPrecio: `{price}`\nCantidad: `{executed_amount}`\nConfianza: `{confidence}%`")
        return {"price": price, "filled": executed_amount, "trade_id": trade_id, "indicators": indicators}
    except Exception as e:
        logging.error(f"Error al ejecutar orden de compra para {symbol}: {e}")
        print(f"Error executing buy for {symbol}: {e}")
        send_telegram_message(f"❌ *Fallo en Compra* `{symbol}`\nError: `{str(e)}`\nCantidad intentada: `{amount}`")
        return None

def sell_symbol(symbol, amount, trade_id):
    try:
        print(f"Executing sell order for {symbol}, amount={amount}, trade_id={trade_id}")
        base_asset = symbol.split('/')[0]
        balance_info = exchange.fetch_balance()
        available = balance_info['free'].get(base_asset, 0)
        if available < amount:
            logging.warning(f"Balance insuficiente para {symbol}: se intenta vender {amount} pero disponible es {available}. Ajustando cantidad.")
            print(f"Insufficient balance for {symbol}: adjusting amount to {available}")
            amount = available
            try:
                amount = float(exchange.amount_to_precision(symbol, amount))
            except Exception as ex:
                logging.warning(f"No se pudo redondear la cantidad para {symbol}: {ex}")
                print(f"Error precision amount for {symbol}: {ex}")

        data, price_series = fetch_and_prepare_data(symbol)
        if data is None:
            price = fetch_price(symbol)
            timestamp = datetime.now(timezone.utc).isoformat()
            order = exchange.create_market_sell_order(symbol, amount)
            sell_price = order.get("price", price)
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transactions_new (symbol, action, price, amount, timestamp, trade_id, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, "sell", sell_price, amount, timestamp, trade_id, 'closed'))
            cursor.execute("UPDATE transactions_new SET status='closed' WHERE trade_id=? AND action='buy'", (trade_id,))
            conn.commit()
            conn.close()
            return
        
        price = fetch_price(symbol)
        timestamp = datetime.now(timezone.utc).isoformat()
        rsi = data['1h']['RSI'].iloc[-1] if not pd.isna(data['1h']['RSI'].iloc[-1]) else None
        adx = calculate_adx(data['1h'])
        atr = data['1h']['ATR'].iloc[-1] if not pd.isna(data['1h']['ATR'].iloc[-1]) else None

        order = exchange.create_market_sell_order(symbol, amount)
        sell_price = order.get("price", price)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions_new (symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, "sell", sell_price, amount, timestamp, trade_id, rsi, adx, atr, 'closed'))
        cursor.execute("UPDATE transactions_new SET status='closed' WHERE trade_id=? AND action='buy'", (trade_id,))
        conn.commit()
        conn.close()
        
        logging.info(f"Venta ejecutada: {symbol} a {sell_price} (ID: {trade_id})")
        print(f"Sell executed: {symbol} at {sell_price} (ID: {trade_id})")
        send_telegram_message(f"✅ *Venta Ejecutada* para `{symbol}`\nPrecio: `{sell_price}`\nCantidad: `{amount}`")
        analyze_trade_outcome(trade_id)
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")
        print(f"Error selling {symbol}: {e}")

def dynamic_trailing_stop(symbol, amount, purchase_price, trade_id, indicators):
    def trailing_logic():
        try:
            highest_price = purchase_price
            take_profit_price = purchase_price * 1.05
            stop_loss_price = purchase_price * 0.98
            data, price_series = fetch_and_prepare_data(symbol)
            if data is None or price_series is None:
                logging.error(f"No data para trailing stop de {symbol}, forzando venta inmediata")
                print(f"No data for trailing stop {symbol}, forcing sell")
                sell_symbol(symbol, amount, trade_id)
                return

            atr = data['1h']['ATR'].iloc[-1] if '1h' in data and 'ATR' in data['1h'] and not pd.isna(data['1h']['ATR'].iloc[-1]) else purchase_price * 0.02
            volatility = atr / purchase_price * 100 if purchase_price > 0 else 3.0
            support_level = indicators.get('support_level', None)

            while True:
                current_price = fetch_price(symbol)
                if current_price is None:
                    logging.warning(f"No se pudo obtener precio para {symbol}, reintentando en 15s")
                    print(f"No price for {symbol}, retrying in 15s")
                    time.sleep(15)
                    continue

                if current_price <= stop_loss_price:
                    sell_symbol(symbol, amount, trade_id)
                    logging.info(f"Stop-loss alcanzado para {symbol} a {current_price}")
                    print(f"Stop-loss hit for {symbol} at {current_price}")
                    break

                if current_price >= take_profit_price:
                    sell_symbol(symbol, amount, trade_id)
                    logging.info(f"Take-profit alcanzado para {symbol} a {current_price}")
                    print(f"Take-profit hit for {symbol} at {current_price}")
                    break

                support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else float('inf')
                if support_distance < 0.05:
                    trailing_percent = max(3.0, min(6.0, volatility * 1.0))
                else:
                    trailing_percent = max(4.0, min(8.0, volatility * 1.5))

                if current_price > highest_price:
                    highest_price = current_price
                trailing_stop_price = highest_price * (1 - trailing_percent / 100)

                logging.info(f"Trailing stop {symbol}: precio actual={current_price}, máximo={highest_price}, stop={trailing_stop_price}, trailing_percent={trailing_percent:.2f}%")
                print(f"Trailing stop for {symbol}: Current={current_price}, High={highest_price}, Stop={trailing_stop_price}, Percent={trailing_percent:.2f}%")

                if current_price <= trailing_stop_price:
                    sell_symbol(symbol, amount, trade_id)
                    break
                time.sleep(15)  # Shortened for faster checks

        except Exception as e:
            logging.error(f"Error en trailing stop para {symbol}: {e}")
            print(f"Error in trailing stop for {symbol}: {e}")
            sell_symbol(symbol, amount, trade_id)

    threading.Thread(target=trailing_logic, daemon=True).start()

def calculate_established_strategy(indicators, data=None):
    rsi = indicators.get('rsi', None)
    relative_volume = indicators.get('relative_volume', None)
    price_trend = indicators.get('price_trend', 'insufficient_data')
    short_volume_trend = indicators.get('short_volume_trend', 'insufficient_data')
    current_price = indicators.get('current_price', 0)
    support_level = indicators.get('support_level', None)
    adx = indicators.get('adx', None)
    symbol = indicators.get('symbol', 'desconocido')

    logging.info(f"Calculating strategy for {symbol}: RSI={rsi}, Relative Volume={relative_volume}, ADX={adx}")
    print(f"Strategy calc for {symbol}: RSI={rsi}, Rel Vol={relative_volume}, ADX={adx}")

    # Adjusted thresholds for established coins
    MIN_ADX = 20
    MIN_RELATIVE_VOLUME = 0.5  # Lowered to allow more trades
    MAX_SUPPORT_DISTANCE = 0.03
    OVERSOLD_THRESHOLD = 0.95
    VOLUME_SPIKE_FACTOR = 1.5

    support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else 0.5

    # Initial filters
    if adx is None or adx < MIN_ADX:
        return "mantener", 50, f"Tendencia débil (ADX: {adx if adx else 'None'}) para {symbol}"
    if relative_volume is None or relative_volume < MIN_RELATIVE_VOLUME:
        return "mantener", 50, f"Volumen relativo bajo ({relative_volume}) para {symbol}"
    if short_volume_trend != "increasing":
        return "mantener", 50, f"Volumen no favorable para {symbol}"
    if support_distance > MAX_SUPPORT_DISTANCE:
        return "mantener", 50, f"Lejos del soporte ({support_distance:.2%}) para {symbol}"
    if rsi is None or rsi < 30:
        return "mantener", 50, f"RSI bajo ({rsi}) para {symbol}"

    # Volume spike filter
    volume_spike = False
    if data and '1h' in data:
        df = data['1h']
        if len(df) >= 10:
            avg_volume_10 = df['volume'].rolling(window=10).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_spike = current_volume > avg_volume_10 * VOLUME_SPIKE_FACTOR
        if not volume_spike:
            return "mantener", 50, f"Sin pico de volumen para {symbol}"

    # Oversold condition
    oversold = False
    if data and '1h' in data:
        df = data['1h']
        if len(df) >= 7:
            ma7 = df['close'].rolling(window=7).mean().iloc[-1]
            oversold = current_price < ma7 * OVERSOLD_THRESHOLD

    # Weighted scoring (simplified)
    weighted_signals = [
        3 * (relative_volume > 0.5 if relative_volume else False),
        2 * (short_volume_trend == "increasing"),
        1 * (price_trend == "increasing"),
        3 * (support_distance <= 0.03),
        1 * (adx > 30 if adx else False),
        -1 * (rsi > 70 if rsi else False),  # Penalty for overbought
        2 * oversold,
        1 * volume_spike
    ]
    signals_score = sum(weighted_signals)
    logging.info(f"Strategy score for {symbol}: {signals_score}, Oversold={oversold}, Volume Spike={volume_spike}")
    print(f"Strategy score for {symbol}: {signals_score}, Oversold={oversold}, Volume Spike={volume_spike}")

    # Decision with adjusted threshold
    if signals_score >= 5:  # Lowered to allow more trades
        action = "comprar"
        confidence = 80 if signals_score < 7 else 90
        explanation = f"Compra fuerte (establecido): Volumen={relative_volume}, ADX={adx}, soporte_dist={support_distance:.2%}, RSI={rsi}, Sobrevendido={oversold}, Pico de volumen={volume_spike} para {symbol}"
    else:
        action = "mantener"
        confidence = 60
        explanation = f"Insuficiente (establecido): Volumen={relative_volume}, ADX={adx}, soporte_dist={support_distance:.2%}, RSI={rsi}, Sobrevendido={oversold}, Pico de volumen={volume_spike}, puntaje={signals_score} para {symbol}"

    return action, confidence, explanation

def evaluate_missed_opportunity(symbol, initial_price, confidence, explanation, indicators):
    time.sleep(1800)  # Shortened to 30 min for quicker feedback
    final_price = fetch_price(symbol)
    if final_price and initial_price:
        price_change = ((final_price - initial_price) / initial_price) * 100
        if price_change > 6.0:
            missed_opportunity = {
                "symbol": symbol,
                "initial_timestamp": get_colombia_timestamp(),
                "initial_price": initial_price,
                "final_price": final_price,
                "price_change": price_change,
                "confidence": confidence,
                "explanation": explanation,
                "rsi": indicators.get('rsi', None),
                "relative_volume": indicators.get('relative_volume', None),
                "short_volume_trend": indicators.get('short_volume_trend', 'insufficient_data'),
                "price_trend": indicators.get('price_trend', 'insufficient_data'),
                "support_distance": None if indicators.get('support_level', None) is None or initial_price <= 0 else (initial_price - indicators['support_level']) / indicators['support_level'],
                "depth": indicators.get('depth', 0),
                "spread": indicators.get('spread', float('inf'))
            }
            logging.info(f"Missed opportunity for {symbol}: Change={price_change:.2f}%")
            print(f"Missed opportunity for {symbol}: Change={price_change:.2f}%")
            with open("missed_opportunities.csv", "a", newline='') as f:
                f.write(f"{missed_opportunity['initial_timestamp']},{missed_opportunity['symbol']},{missed_opportunity['initial_price']},{missed_opportunity['final_price']},{missed_opportunity['price_change']:.2f},{missed_opportunity['confidence']},{missed_opportunity['explanation']},{json.dumps(missed_opportunity)}\n")
            print(f"\n=== Oportunidad Perdida Confirmada ===\n"
                  f"Símbolo: {symbol}\n"
                  f"Precio Inicial: {initial_price:.4f} USDT\n"
                  f"Precio Final: {final_price:.4f} USDT\n"
                  f"Cambio: {price_change:.2f}%\n"
                  f"Confianza: {confidence}%\n"
                  f"Explicación: {explanation}\n")
            send_telegram_message(f"⚠️ *Oportunidad Perdida Confirmada* `{symbol}`\nPrecio Inicial: `{initial_price:.4f}`\nPrecio Final: `{final_price:.4f}`\nCambio: `{price_change:.2f}%`\nConfianza: `{confidence}%`\nExplicación: `{explanation}`")

            with open("trade_stats.csv", "a", newline='') as f:
                f.write(f"{symbol}_{missed_opportunity['initial_timestamp'].replace(':', '-')},"
                        f"{symbol},"
                        f"{missed_opportunity['initial_price']},"
                        f"{missed_opportunity['final_price']},"
                        f"0.0,"
                        f"{(missed_opportunity['final_price'] - missed_opportunity['initial_price']) * 0.0},"
                        f"{missed_opportunity['rsi'] or 'N/A'},"
                        f"{missed_opportunity['initial_timestamp']},"
                        f"'Missed Opportunity',"
                        f"{missed_opportunity['confidence']},"
                        f"{missed_opportunity['explanation']},"
                        f"{missed_opportunity['relative_volume'] or 'N/A'},"
                        f"{missed_opportunity['short_volume_trend']},"
                        f"{missed_opportunity['price_trend']},"
                        f"{missed_opportunity['support_distance'] or 'N/A'},"
                        f"{missed_opportunity['depth']},"
                        f"{missed_opportunity['spread']}\n")

def fetch_ohlcv_with_retry(symbol, timeframe, limit=100, max_retries=3):
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 0:
                logging.info(f"OHLCV fetched for {symbol} on {timeframe}, length={len(ohlcv)}")
                print(f"OHLCV fetched for {symbol} on {timeframe}, length={len(ohlcv)}")
                return ohlcv
            logging.warning(f"Datos vacíos o None para {symbol} en {timeframe}, intento {attempt + 1}/{max_retries}")
            print(f"No data for {symbol} on {timeframe}, attempt {attempt + 1}")
        except Exception as e:
            logging.error(f"Error al obtener OHLCV para {symbol} en {timeframe}, intento {attempt + 1}/{max_retries}: {e}")
            print(f"Error fetching OHLCV for {symbol} on {timeframe}: {e}")
        time.sleep(2 ** attempt)  # Exponential backoff, short
    return None

buy_lock = threading.Lock()

def demo_trading():
    logging.info("Iniciando trading en segundo plano para los activos establecidos...")
    print("Starting trading loop...")
    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
    logging.info(f"Saldo USDT disponible: {usdt_balance}")
    print(f"USDT balance: {usdt_balance}")
    
    reserve = 150
    available_for_trading = max(usdt_balance - reserve, 0)
    if available_for_trading < MIN_NOTIONAL:
        logging.warning(f"Disponible para trading insuficiente ({available_for_trading}) después de reserva.")
        print(f"Insufficient trading balance: {available_for_trading}")
        return False

    open_trades = get_open_trades()
    logging.info(f"Operaciones abiertas actualmente: {open_trades}")
    print(f"Current open trades: {open_trades}")
    if open_trades >= MAX_OPEN_TRADES:
        logging.info("Límite de operaciones abiertas alcanzado.")
        print("Max open trades reached.")
        return False

    budget_per_trade = available_for_trading / (MAX_OPEN_TRADES - open_trades) if (MAX_OPEN_TRADES - open_trades) > 0 else available_for_trading
    logging.info(f"Presupuesto por operación: {budget_per_trade}")
    print(f"Budget per trade: {budget_per_trade}")
    balance = exchange.fetch_balance()['free']
    logging.info(f"Balance actual: {balance}")
    print(f"Current balance: {balance}")

    failed_conditions_count = {}
    symbols_processed = 0

    for symbol in SELECTED_CRYPTOS:
        try:
            print(f"\n--- Processing symbol: {symbol} ---")
            open_trades = get_open_trades()
            logging.info(f"Operaciones abiertas antes de procesar {symbol}: {open_trades}")
            print(f"Open trades before processing {symbol}: {open_trades}")
            if open_trades >= MAX_OPEN_TRADES:
                logging.info("Límite de operaciones abiertas alcanzado.")
                print("Max open trades reached, skipping.")
                return False

            base_asset = symbol.split('/')[0]
            if base_asset in balance and balance[base_asset] > 0.001:  # Ignore dust
                logging.info(f"Se omite {symbol} porque ya tienes una posición abierta.")
                print(f"Skipping {symbol}: Position already open.")
                continue

            conditions = {}
            daily_volume = fetch_volume(symbol)
            conditions['daily_volume >= 250000'] = daily_volume is not None and daily_volume >= 250000
            if not conditions['daily_volume >= 250000']:
                logging.info(f"Se omite {symbol} por volumen insuficiente: {daily_volume}")
                print(f"Skipping {symbol}: Insufficient volume {daily_volume}")
                failed_conditions_count['daily_volume >= 250000'] = failed_conditions_count.get('daily_volume >= 250000', 0) + 1
                continue

            order_book_data = fetch_order_book_data(symbol)
            conditions['order_book_available'] = order_book_data is not None
            if not conditions['order_book_available']:
                logging.warning(f"Se omite {symbol} por fallo en datos del libro de órdenes")
                print(f"Skipping {symbol}: Order book fetch failed")
                failed_conditions_count['order_book_available'] = failed_conditions_count.get('order_book_available', 0) + 1
                continue

            conditions['depth >= 100000'] = order_book_data['depth'] >= 100000
            if not conditions['depth >= 100000']:
                logging.warning(f"Profundidad baja para {symbol}: {order_book_data['depth']}, pero se evalúa de todos modos")
            else:
                logging.info(f"Profundidad aceptable para {symbol}: {order_book_data['depth']}")

            current_price = fetch_price(symbol)
            conditions['price_available'] = current_price is not None
            if not conditions['price_available']:
                logging.warning(f"Se omite {symbol} por no obtener precio")
                print(f"Skipping {symbol}: No price available")
                failed_conditions_count['price_available'] = failed_conditions_count.get('price_available', 0) + 1
                continue

            try:
                current_price = float(current_price)
                spread = float(order_book_data['spread']) if order_book_data['spread'] is not None else float('inf')
                imbalance = float(order_book_data['imbalance']) if order_book_data['imbalance'] is not None else 0
                depth = float(order_book_data['depth'])
            except (ValueError, TypeError) as e:
                logging.error(f"Error al convertir datos numéricos para {symbol}: {e}")
                print(f"Error converting numeric data for {symbol}: {e}")
                continue

            conditions['spread <= 0.005 * price'] = spread <= 0.005 * current_price
            if not conditions['spread <= 0.005 * price']:
                logging.info(f"Se omite {symbol} por spread alto: {spread}")
                print(f"Skipping {symbol}: High spread {spread}")
                failed_conditions_count['spread <= 0.005 * price'] = failed_conditions_count.get('spread <= 0.005 * price', 0) + 1
                continue

            conditions['imbalance >= 1.0'] = imbalance >= 1.0 if imbalance is not None else False
            if not conditions['imbalance >= 1.0']:
                logging.warning(f"Imbalance bajo para {symbol}: {imbalance}, pero se evalúa de todos modos")
            else:
                logging.info(f"Imbalance aceptable para {symbol}: {imbalance}")

            data, price_series = fetch_and_prepare_data(symbol)
            if data is None or price_series is None:
                logging.warning(f"Se omite {symbol} por datos insuficientes")
                print(f"Skipping {symbol}: Insufficient data")
                failed_conditions_count['data_available'] = failed_conditions_count.get('data_available', 0) + 1
                continue

            df_slice = data.get('1h', data.get('4h', data.get('1d')))
            if df_slice.empty:
                logging.warning(f"No hay datos válidos en ningún timeframe para {symbol}")
                print(f"No valid data in any timeframe for {symbol}")
                failed_conditions_count['timeframe_available'] = failed_conditions_count.get('timeframe_available', 0) + 1
                continue
            df_slice = df_slice.iloc[-100:]

            adx = calculate_adx(df_slice) if not df_slice.empty else None
            atr = df_slice['ATR'].iloc[-1] if 'ATR' in df_slice and not pd.isna(df_slice['ATR'].iloc[-1]) else None
            rsi = df_slice['RSI'].iloc[-1] if 'RSI' in df_slice and not pd.isna(df_slice['RSI'].iloc[-1]) else None
            volume_series = df_slice['volume']
            relative_volume = volume_series.iloc[-1] / volume_series[-10:].mean() if len(volume_series) >= 10 and volume_series[-10:].mean() != 0 else None
            macd = df_slice['MACD'].iloc[-1] if 'MACD' in df_slice and not pd.isna(df_slice['MACD'].iloc[-1]) else None
            macd_signal = df_slice['MACD_signal'].iloc[-1] if 'MACD_signal' in df_slice and not pd.isna(df_slice['MACD_signal'].iloc[-1]) else None
            roc = df_slice['ROC'].iloc[-1] if 'ROC' in df_slice and not pd.isna(df_slice['ROC'].iloc[-1]) else None
            has_crossover, candles_since = has_recent_macd_crossover(
                df_slice['MACD'] if 'MACD' in df_slice else pd.Series(),
                df_slice['MACD_signal'] if 'MACD_signal' in df_slice else pd.Series(),
                lookback=5
            )
            short_volume_trend = calculate_short_volume_trend(data) if data else "insufficient_data"
            volume_trend = "insufficient_data"
            price_trend = "insufficient_data"

            if len(volume_series) >= 10:
                last_10_volume = volume_series[-10:]
                try:
                    slope_volume, _, _, _, _ = linregress(range(10), last_10_volume)
                    volume_trend = "increasing" if slope_volume > 0.01 else "decreasing" if slope_volume < -0.01 else "stable"
                    logging.debug(f"Volume trend calculado para {symbol}: slope={slope_volume}, trend={volume_trend}")
                    print(f"Volume trend for {symbol}: {volume_trend}")
                except Exception as e:
                    logging.error(f"Error al calcular volume_trend para {symbol}: {e}", exc_info=True)
                    print(f"Error calculating volume trend for {symbol}: {e}")
                    volume_trend = "insufficient_data"
            else:
                logging.debug(f"No hay suficientes datos para calcular volume_trend para {symbol}: {len(volume_series)} velas")
                print(f"Insufficient data for volume trend {symbol}")

            if len(price_series) >= 10:
                last_10_price = price_series[-10:]
                try:
                    slope_price, _, _, _, _ = linregress(range(10), last_10_price)
                    price_trend = "increasing" if slope_price > 0.01 else "decreasing" if slope_price < -0.01 else "stable"
                    logging.debug(f"Price trend calculado para {symbol}: slope={slope_price}, trend={price_trend}")
                    print(f"Price trend for {symbol}: {price_trend}")
                except Exception as e:
                    logging.error(f"Error al calcular price_trend para {symbol}: {e}", exc_info=True)
                    print(f"Error calculating price trend for {symbol}: {e}")
                    price_trend = "insufficient_data"
            else:
                logging.debug(f"No hay suficientes datos para calcular price_trend para {symbol}: {len(price_series)} velas")
                print(f"Insufficient data for price trend {symbol}")

            support_level = detect_support_level(data, price_series, window=15, max_threshold_multiplier=1.0)
            support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else None
            logging.info(f"Support level for {symbol}: {support_level}, distance={support_distance}")
            print(f"Support level for {symbol}: {support_level}, distance={support_distance}")

            indicators = {
                "adx": adx,
                "atr": atr,
                "rsi": rsi,
                "relative_volume": relative_volume,
                "macd": macd,
                "macd_signal": macd_signal,
                "has_macd_crossover": has_crossover,
                "candles_since_crossover": candles_since,
                "spread": spread,
                "imbalance": imbalance,
                "depth": depth,
                "volume_trend": volume_trend,
                "price_trend": price_trend,
                "short_volume_trend": short_volume_trend,
                "support_level": support_level,
                "roc": roc,
                "current_price": current_price,
                "symbol": symbol
            }

            # Print full indicators for screen debug
            print(f"Indicators for {symbol}: {indicators}")

            conditions_str = "\n".join([f"{key}: {'Sí' if value is True else 'No' if value is False else 'Desconocido'}" for key, value in sorted(conditions.items())])
            logging.info(f"Condiciones evaluadas para {symbol}:\n{conditions_str}")
            print(f"Conditions for {symbol}:\n{conditions_str}")

            for key, value in conditions.items():
                logging.debug(f"Condición {key} para {symbol}: valor={value}, tipo={type(value)}")

            action, confidence, explanation = calculate_established_strategy(indicators, data)
            logging.info(f"Decisión para {symbol}: {action} (Confianza: {confidence}%) - {explanation}")
            print(f"Decision for {symbol}: {action} (Confidence: {confidence}%) - {explanation}")

            sentiment_score, classification = get_market_sentiment()
            logging.info(f"Market sentiment: Score={sentiment_score}, Classification={classification} for {symbol}")
            print(f"Market sentiment for {symbol}: Score={sentiment_score}, Class={classification}")
            if sentiment_score > 50:  # Greed/positive
                confidence = min(confidence + 10, 100)
                explanation += f" (Market sentiment positive: {classification}, Score={sentiment_score})"
            elif sentiment_score < 50:  # Fear/negative
                action = "mantener"
                confidence = 50
                explanation += f" (Market sentiment negative: {classification}, Score={sentiment_score}, overriding to hold)"

            logging.info(f"Final decision for {symbol}: {action} (Confidence: {confidence}%) - {explanation}")
            print(f"Final decision for {symbol}: {action} (Confidence: {confidence}%) - {explanation}")
            logging.debug(f"Verificación post-decisión para {symbol}: action={action}, confidence={confidence}, explanation={explanation}, conditions={conditions}")

            if action == "comprar" and confidence >= 75:
                with buy_lock:
                    open_trades = get_open_trades()
                    if open_trades >= MAX_OPEN_TRADES:
                        logging.info(f"Límite de operaciones abiertas alcanzado, no se ejecuta compra para {symbol}.")
                        print(f"Max open trades, no buy for {symbol}")
                        return False

                    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
                    available_for_trading = max(usdt_balance - reserve, 0)
                    if available_for_trading < MIN_NOTIONAL:
                        logging.warning(f"Disponible para trading insuficiente ({available_for_trading}) para {symbol}.")
                        print(f"Insufficient trading balance for {symbol}")
                        continue

                    confidence_factor = confidence / 100
                    if current_price is None or current_price <= 0:
                        logging.error(f"Precio actual inválido para {symbol} ({current_price}), omitiendo operación")
                        print(f"Invalid price for {symbol}, skipping")
                        continue

                    atr_value = atr if atr is not None else 0.02 * current_price
                    if atr_value <= 0 or current_price <= 0:
                        volatility_factor = 1.0
                        logging.warning(f"ATR o precio inválido para {symbol}, usando volatility_factor por defecto: {volatility_factor}")
                        print(f"Invalid ATR/price for {symbol}, default volatility")
                    else:
                        volatility_factor = min(2.0, (atr_value / current_price * 100))
                    size_multiplier = confidence_factor * volatility_factor
                    
                    adjusted_budget = budget_per_trade * size_multiplier
                    min_amount_for_notional = MIN_NOTIONAL / current_price
                    target_amount = max(adjusted_budget / current_price, min_amount_for_notional)
                    amount = min(target_amount, 0.10 * usdt_balance / current_price)
                    trade_value = amount * current_price

                    logging.info(f"Intentando compra para {symbol}: amount={amount}, trade_value={trade_value}, confidence={confidence}%, volatility_factor={volatility_factor:.2f}x")
                    print(f"Attempting buy for {symbol}: Amount={amount}, Value={trade_value}, Conf={confidence}%, Vol Factor={volatility_factor:.2f}")
                    if trade_value >= MIN_NOTIONAL or (trade_value < MIN_NOTIONAL and trade_value >= usdt_balance):
                        order = execute_order_buy(symbol, amount, indicators, confidence)
                        if order:
                            logging.info(f"Compra ejecutada para {symbol}: {explanation}")
                            print(f"Buy executed for {symbol}: {explanation}")
                            send_telegram_message(f"✅ *Compra* `{symbol}`\nConfianza: `{confidence}%`\nCantidad: `{amount}`\nExplicación: `{explanation}`")
                            dynamic_trailing_stop(symbol, order['filled'], order['price'], order['trade_id'], indicators)
                            # Actualizar presupuesto después de compra
                            usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
                            available_for_trading = max(usdt_balance - reserve, 0)
                            remaining_slots = MAX_OPEN_TRADES - get_open_trades()
                            budget_per_trade = available_for_trading / remaining_slots if remaining_slots > 0 else available_for_trading
                            logging.info(f"Presupuesto actualizado después de compra: {budget_per_trade}")
                            print(f"Budget updated after buy: {budget_per_trade}")
                        else:
                            logging.error(f"Error al ejecutar compra para {symbol}: orden no completada")
                            print(f"Error executing buy for {symbol}")
                    else:
                        logging.info(f"Compra no ejecutada para {symbol}: valor de la operación ({trade_value}) < MIN_NOTIONAL ({MIN_NOTIONAL}) y saldo insuficiente")
                        print(f"No buy for {symbol}: Value < min notional")

            failed_conditions = [key for key, value in conditions.items() if not value]
            for condition in failed_conditions:
                failed_conditions_count[condition] = failed_conditions_count.get(condition, 0) + 1
            symbols_processed += 1

        except Exception as e:
            logging.error(f"Error en demo_trading para {symbol}: {e}", exc_info=True)
            print(f"Error in demo_trading for {symbol}: {e}")
            continue

    if symbols_processed > 0 and failed_conditions_count:
        most_common_condition = max(failed_conditions_count, key=failed_conditions_count.get)
        most_common_count = failed_conditions_count[most_common_condition]
        summary_message = (f"Resumen final: Después de procesar {symbols_processed} símbolos, "
                           f"la condición más común que impidió operaciones fue '{most_common_condition}' "
                           f"con {most_common_count} ocurrencias ({(most_common_count / symbols_processed) * 100:.1f}%).")
        logging.info(summary_message)
        print(summary_message)

        with open("trade_blockers_summary.txt", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {summary_message}\n")
            f.write(f"Detalles de condiciones fallidas: {dict(failed_conditions_count)}\n\n")
    else:
        logging.info("No se procesaron símbolos o no hubo condiciones fallidas para analizar.")
        print("No symbols processed or no failed conditions.")

    logging.info("Trading ejecutado correctamente en segundo plano para los activos establecidos")
    print("Trading cycle completed.")
    return True

def analyze_trade_outcome(trade_id):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT t1.symbol, t1.price AS buy_price, t1.amount, t1.timestamp AS buy_time,
                   t1.rsi, t1.adx, t1.atr, t1.relative_volume, t1.divergence, t1.bb_position,
                   t1.confidence, t1.has_macd_crossover, t1.candles_since_crossover,
                   t1.volume_trend, t1.price_trend, t1.short_volume_trend, t1.support_level,
                   t1.spread, t1.imbalance, t1.depth,
                   t2.price AS sell_price, t2.timestamp AS sell_time
            FROM transactions_new t1
            LEFT JOIN transactions_new t2 ON t1.trade_id = t2.trade_id AND t2.action = 'sell'
            WHERE t1.trade_id = ? AND t1.action = 'buy'
        """, (trade_id,))
        trade_data = cursor.fetchone()
        
        if not trade_data:
            logging.warning(f"No se encontraron datos para el trade_id: {trade_id}")
            print(f"No data for trade_id: {trade_id}")
            return
        
        if trade_data[20] is None:
            logging.info(f"Trade {trade_id} no tiene venta registrada aún")
            print(f"Trade {trade_id} no sale yet")
            return

        buy_data = {
            'symbol': trade_data[0],
            'buy_price': trade_data[1],
            'amount': trade_data[2],
            'timestamp': trade_data[3],
            'rsi': trade_data[4],
            'adx': trade_data[5],
            'atr': trade_data[6],
            'relative_volume': trade_data[7],
            'divergence': trade_data[8],
            'bb_position': trade_data[9],
            'confidence': trade_data[10],
            'has_macd_crossover': trade_data[11],
            'candles_since_crossover': trade_data[12],
            'volume_trend': trade_data[13],
            'price_trend': trade_data[14],
            'short_volume_trend': trade_data[15],
            'support_level': trade_data[16],
            'spread': trade_data[17],
            'imbalance': trade_data[18],
            'depth': trade_data[19]
        }
        sell_price = trade_data[20]

        profit_loss = (sell_price - buy_data['buy_price']) * buy_data['amount']
        is_profitable = profit_loss > 0

        with open("trade_stats.csv", "a") as f:
            f.write(f"{trade_id},{buy_data['symbol']},{buy_data['buy_price']},{sell_price},{buy_data['amount']},{profit_loss:.2f},{buy_data['rsi'] or 'N/A'},{datetime.now()}\n")

        global RSI_THRESHOLD
        if 'RSI_THRESHOLD' not in globals():
            RSI_THRESHOLD = 70
        if profit_loss > 0 and buy_data['rsi'] and buy_data['rsi'] < 80:
            RSI_THRESHOLD = max(65, RSI_THRESHOLD - 5)
            logging.info(f"RSI_THRESHOLD reducido a {RSI_THRESHOLD} por operación rentable con RSI temprano")
            print(f"RSI_THRESHOLD reduced to {RSI_THRESHOLD}")
        elif profit_loss < 0 and buy_data['rsi'] and buy_data['rsi'] > 75:
            RSI_THRESHOLD = min(85, RSI_THRESHOLD + 5)
            logging.info(f"RSI_THRESHOLD aumentado a {RSI_THRESHOLD} por operación perdedora con RSI tardío")
            print(f"RSI_THRESHOLD increased to {RSI_THRESHOLD}")

        telegram_message = f"📊 *Análisis de Resultado* `{buy_data['symbol']}` (ID: {trade_id})\n" \
                          f"Resultado: {'Éxito' if is_profitable else 'Fracaso'}\n" \
                          f"Ganancia/Pérdida: {profit_loss:.2f} USDT\n" \
                          f"Confianza: {buy_data['confidence'] if buy_data['confidence'] else 'N/A'}%"
        send_telegram_message(telegram_message)
        logging.info(f"Análisis de transacción {trade_id}: {'Éxito' if is_profitable else 'Fracaso'} - P/L {profit_loss:.2f}")

    except Exception as e:
        logging.error(f"Error al analizar el resultado de la transacción {trade_id}: {e}")
        print(f"Error analyzing trade {trade_id}: {e}")
    finally:
        conn.close()

def send_periodic_summary():
    while True:
        try:
            with open("trading.log", "r") as log_file:
                lines = log_file.readlines()[-100:]
                buys_attempted = sum(1 for line in lines if "Intentando compra para" in line)
                buys_executed = sum(1 for line in lines if "Compra ejecutada para" in line)
                errors = sum(1 for line in lines if "Error" in line)
                symbols = set(line.split("para ")[1].split(":")[0] for line in lines if "Procesando" in line)

            message = (f"📈 *Resumen del Bot* ({get_colombia_timestamp()})\n"
                      f"Compras intentadas: `{buys_attempted}`\n"
                      f"Compras ejecutadas: `{buys_executed}`\n"
                      f"Errores recientes: `{errors}`\n"
                      f"Símbolos evaluados: `{len(symbols)}` (e.g., {', '.join(list(symbols)[:3])}...)")
            send_telegram_message(message)
            print(f"Periodic summary sent: Attempts={buys_attempted}, Executed={buys_executed}, Errors={errors}")
        except Exception as e:
            logging.error(f"Error en resumen periódico: {e}")
            print(f"Error in periodic summary: {e}")
            send_telegram_message(f"⚠️ *Error en Resumen* `{str(e)}`")
        time.sleep(3600)

def daily_summary():
    while True:
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            # Total P/L from closed trades
            cursor.execute("""
                SELECT SUM((t2.price - t1.price) * t1.amount) FROM transactions_new t1
                JOIN transactions_new t2 ON t1.trade_id = t2.trade_id AND t2.action = 'sell'
                WHERE t1.action = 'buy' AND t1.status = 'closed'
            """)
            total_pl = cursor.fetchone()[0] or 0.0

            # Win rate: profitable closed trades / total closed
            cursor.execute("""
                SELECT COUNT(*) FROM transactions_new t1
                JOIN transactions_new t2 ON t1.trade_id = t2.trade_id AND t2.action = 'sell'
                WHERE t1.action = 'buy' AND t1.status = 'closed' AND (t2.price - t1.price) > 0
            """)
            wins = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM transactions_new t1
                JOIN transactions_new t2 ON t1.trade_id = t2.trade_id AND t2.action = 'sell'
                WHERE t1.action = 'buy' AND t1.status = 'closed'
            """)
            total_closed = cursor.fetchone()[0]
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0

            # Total trade volume (executed buys)
            cursor.execute("SELECT COUNT(*) FROM transactions_new WHERE action='buy'")
            total_trades = cursor.fetchone()[0]

            # Missed opps count from CSV (or DB if logged)
            missed_count = sum(1 for line in open("missed_opportunities.csv", "r") if line.strip()) if os.path.exists("missed_opportunities.csv") else 0

            message = (f"📊 *Resumen Diario del Bot* ({get_colombia_timestamp()})\n"
                       f"Total Ganancia/Pérdida: {total_pl:.2f} USDT\n"
                       f"Tasa de Ganancia: {win_rate:.1f}% ({wins}/{total_closed} trades)\n"
                       f"Volumen Total de Trades: {total_trades}\n"
                       f"Oportunidades Perdidas: {missed_count}")
            send_telegram_message(message)
            logging.info(message)
            print(message)
            conn.close()
        except Exception as e:
            logging.error(f"Error en resumen diario: {e}")
            print(f"Error in daily summary: {e}")
            send_telegram_message(f"⚠️ *Error en Resumen Diario* `{str(e)}`")
        time.sleep(86400)  # 24 hours

if __name__ == "__main__":
    threading.Thread(target=send_periodic_summary, daemon=True).start()
    threading.Thread(target=daily_summary, daemon=True).start()
    while True:  # Loop infinito para re-evaluar continuamente
        demo_trading()
        time.sleep(30)  # Espera 30 segundos para la siguiente iteración
    logging.shutdown()