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
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import pytz
import pandas_ta as ta
from elegir_cripto import choose_best_cryptos
from scipy.stats import linregress

# Configuración e Inicialización
load_dotenv()
GPT_MODEL = "gpt-4o-mini"
DB_NAME = "trading_real.db"
ESTABLISHED_COINS = ['BTC', 'ETH', 'BNB']

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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logger = logging.getLogger("inversion_binance")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(os.path.expanduser("~/hobbies/trading.log"))
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.info("Prueba de escritura en trading.log al iniciar")

# Constantes
MAX_OPEN_TRADES = 3
MIN_NOTIONAL = 10
RSI_THRESHOLD = 70
ADX_THRESHOLD = 25
VOLUME_GROWTH_THRESHOLD = 0.5

# Cache de decisiones
decision_cache = {}
CACHE_EXPIRATION = 300

def detect_support_level(data, price_series, window=15, max_threshold_multiplier=1.0):
    if len(price_series) < window:
        logging.warning(f"Series too short for {price_series.name}: {len(price_series)} < {window}")
        return None

    recent_prices = price_series[-window:]
    local_mins = [
        recent_prices.iloc[i] for i in range(1, len(recent_prices) - 1)
        if recent_prices.iloc[i] < recent_prices.iloc[i - 1] and recent_prices.iloc[i] < recent_prices.iloc[i + 1]
    ]

    if len(local_mins) < 2:
        logging.warning(f"Insufficient local minima (< 2) for {price_series.name}")
        return None

    support_level = np.mean(local_mins)
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
    threshold = min(threshold, 1.02)

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

def fetch_order_book_data(symbol, limit=20):
    try:
        order_book = exchange.fetch_order_book(symbol, limit=limit)
        bids = order_book['bids']
        asks = order_book['asks']
        spread = asks[0][0] - bids[0][0] if bids and asks else None
        bid_volume = sum([volume for _, volume in bids])
        ask_volume = sum([volume for _, volume in asks])
        imbalance = bid_volume / ask_volume if ask_volume > 0 else None
        depth = bid_volume + ask_volume
        return {
            'spread': spread,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            'depth': depth
        }
    except Exception as e:
        logging.error(f"Error al obtener order book para {symbol}: {e}")
        return None

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram no configurado.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        logging.error(f"Error al enviar a Telegram: {e}")

def get_colombia_timestamp():
    colombia_tz = pytz.timezone("America/Bogota")
    return datetime.now(colombia_tz).strftime("%Y-%m-%d %H:%M:%S")

def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error al obtener precio de {symbol}: {e}")
        return None

def fetch_volume(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['quoteVolume']
    except Exception as e:
        logging.error(f"Error al obtener volumen de {symbol}: {e}")
        return None

def fetch_and_prepare_data(symbol, atr_length=7, rsi_length=14, bb_length=20, roc_length=7, limit=100):
    timeframes = ['15m', '1h', '4h', '1d']
    data = {}
    logging.debug(f"Inicio de fetch_and_prepare_data para {symbol}")

    for tf in timeframes:
        try:
            logging.debug(f"Iniciando fetch_ohlcv para {symbol} en {tf} con limit={limit}")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if not ohlcv or len(ohlcv) == 0:
                logging.warning(f"Datos vacíos para {symbol} en {tf}")
                continue

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            logging.debug(f"DataFrame para {symbol} en {tf} creado con {len(df)} velas.")

            if len(df) < max(atr_length, rsi_length, bb_length, roc_length, 15):
                logging.warning(f"Datos insuficientes (<{max(atr_length, rsi_length, bb_length, roc_length, 15)} velas) para {symbol} en {tf}")
                continue

            expected_freq = pd.Timedelta('15m') if tf == '15m' else pd.Timedelta('1h') if tf == '1h' else pd.Timedelta('4h') if tf == '4h' else pd.Timedelta('1d')
            expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq)
            if len(expected_index) > len(df.index):
                gap_ratio = (len(expected_index) - len(df.index)) / len(expected_index)
                if gap_ratio > 0.1:
                    logging.error(f"Demasiados gaps en {symbol} en {tf} (ratio: {gap_ratio:.2f}), omitiendo timeframe")
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

        except Exception as e:
            logging.error(f"Error procesando {symbol} en {tf}: {e}")
            continue

    if not data:
        logging.error(f"No se obtuvieron datos válidos para {symbol} en ningún timeframe")
        return None, None

    has_valid_indicators = any(
        not df[['RSI', 'ATR', 'MACD']].isna().all().all()
        for df in data.values()
    )
    if not has_valid_indicators:
        logging.error(f"No hay indicadores válidos para {symbol}")
        return None, None

    for tf in ['15m', '1h', '4h', '1d']:
        if tf in data and not data[tf].empty:
            price_series = data[tf]['close']
            logging.debug(f"Seleccionada serie de precios para soporte: {tf} con {len(price_series)} velas")
            break
    else:
        logging.error(f"No se pudieron obtener series de precios para {symbol}")
        return None, None

    return data, price_series

def calculate_adx(df):
    try:
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        return adx['ADX_14'].iloc[-1] if not pd.isna(adx['ADX_14'].iloc[-1]) else None
    except Exception as e:
        logging.error(f"Error al calcular ADX: {e}")
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
        return "bullish" if any(d[0] == "bullish" for d in divergences) else "bearish" if any(d[0] == "bearish" for d in divergences) else "none"
    except Exception as e:
        logging.error(f"Error en divergencias: {e}")
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
    conn.close()
    return count

def execute_order_buy(symbol, amount, indicators, confidence):
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        price = order.get("price", fetch_price(symbol))
        executed_amount = order.get("filled", amount)
        if price is None:
            logging.error(f"No se pudo obtener precio para {symbol} después de la orden")
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
        send_telegram_message(f"✅ *Compra* `{symbol}`\nPrecio: `{price}`\nCantidad: `{executed_amount}`\nConfianza: `{confidence}%`")
        return {"price": price, "filled": executed_amount, "trade_id": trade_id, "indicators": indicators}
    except Exception as e:
        logging.error(f"Error al ejecutar orden de compra para {symbol}: {e}")
        send_telegram_message(f"❌ *Fallo en Compra* `{symbol}`\nError: `{str(e)}`\nCantidad intentada: `{amount}`")
        return None

def sell_symbol(symbol, amount, trade_id):
    try:
        base_asset = symbol.split('/')[0]
        balance_info = exchange.fetch_balance()
        available = balance_info['free'].get(base_asset, 0)
        if available < amount:
            logging.warning(f"Balance insuficiente para {symbol}: se intenta vender {amount} pero disponible es {available}. Ajustando cantidad.")
            amount = available
            try:
                amount = float(exchange.amount_to_precision(symbol, amount))
            except Exception as ex:
                logging.warning(f"No se pudo redondear la cantidad para {symbol}: {ex}")

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
        send_telegram_message(f"✅ *Venta Ejecutada* para `{symbol}`\nPrecio: `{sell_price}`\nCantidad: `{amount}`")
        analyze_trade_outcome(trade_id)
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")

def dynamic_trailing_stop(symbol, amount, purchase_price, trade_id, indicators):
    def trailing_logic():
        try:
            highest_price = purchase_price
            take_profit_price = purchase_price * 1.04  # 4% above entry (2:1 risk-reward)
            stop_loss_price = purchase_price * 0.98    # 2% below entry
            data, price_series = fetch_and_prepare_data(symbol)
            if data is None or price_series is None:
                logging.error(f"No data para trailing stop de {symbol}, forzando venta inmediata")
                sell_symbol(symbol, amount, trade_id)
                return

            atr = data['1h']['ATR'].iloc[-1] if '1h' in data and 'ATR' in data['1h'] and not pd.isna(data['1h']['ATR'].iloc[-1]) else purchase_price * 0.02
            volatility = atr / purchase_price * 100 if purchase_price > 0 else 3.0
            support_level = indicators.get('support_level', None)

            while True:
                current_price = fetch_price(symbol)
                if current_price is None:
                    logging.warning(f"No se pudo obtener precio para {symbol}, reintentando en 60s")
                    time.sleep(60)
                    continue

                if current_price <= stop_loss_price:
                    sell_symbol(symbol, amount, trade_id)
                    logging.info(f"Stop-loss alcanzado para {symbol} a {current_price}")
                    break

                if current_price >= take_profit_price:
                    sell_symbol(symbol, amount, trade_id)
                    logging.info(f"Take-profit alcanzado para {symbol} a {current_price}")
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

                if current_price <= trailing_stop_price:
                    sell_symbol(symbol, amount, trade_id)
                    break
                time.sleep(60)

        except Exception as e:
            logging.error(f"Error en trailing stop para {symbol}: {e}")
            sell_symbol(symbol, amount, trade_id)

    threading.Thread(target=trailing_logic, daemon=True).start()

def evaluate_trading_strategy(symbol, data, current_price):
    support_level = detect_support_level(data, data['1h']['close']) if '1h' in data else None
    rsi = data['1h']['RSI'].iloc[-1] if '1h' in data and 'RSI' in data['1h'] else None
    bullish_candle = detect_bullish_candlestick(data, '1h')
    macd_crossover, candles_since = has_recent_macd_crossover(
        data['1h']['MACD'] if '1h' in data and 'MACD' in data['1h'] else pd.Series(),
        data['1h']['MACD_signal'] if '1h' in data and 'MACD_signal' in data['1h'] else pd.Series()
    )
    volume_behavior = calculate_volume_behavior(data, '1h')
    atr = data['1h']['ATR'].iloc[-1] if '1h' in data and 'ATR' in data['1h'] else None

    support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else 1.0
    rsi_oversold = rsi is not None and rsi < 30

    conditions = [
        support_distance <= 0.02,
        rsi_oversold,
        bullish_candle,
        macd_crossover,
        volume_behavior in ["decreasing", "increasing"]
    ]
    met_conditions = sum(1 for c in conditions if c)

    if met_conditions >= 3:
        action = "comprar"
        confidence = 80 + (met_conditions - 3) * 5
        explanation = (f"Compra en {symbol}: {met_conditions}/5 condiciones cumplidas "
                       f"(Soporte: {support_distance:.2%}, RSI: {rsi}, "
                       f"Vela alcista: {bullish_candle}, MACD cruce: {macd_crossover}, "
                       f"Volumen: {volume_behavior})")
    else:
        action = "mantener"
        confidence = 60
        explanation = (f"Mantener {symbol}: Solo {met_conditions}/5 condiciones cumplidas "
                       f"(Soporte: {support_distance:.2%}, RSI: {rsi}, "
                       f"Vela alcista: {bullish_candle}, MACD cruce: {macd_crossover}, "
                       f"Volumen: {volume_behavior})")

    indicators = {
        "rsi": rsi,
        "atr": atr,
        "has_macd_crossover": macd_crossover,
        "candles_since_crossover": candles_since,
        "volume_trend": volume_behavior,
        "support_level": support_level,
        "symbol": symbol
    }
    return action, confidence, explanation, indicators

def calculate_adaptive_strategy(indicators, data=None):
    rsi = indicators.get('rsi')
    current_price = indicators.get('current_price', 0)
    support_level = indicators.get('support_level')
    symbol = indicators.get('symbol', 'desconocido')

    support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else 1.0
    rsi_oversold = rsi is not None and rsi < 30
    bullish_candle = detect_bullish_candlestick(data, '1h') if data else False
    macd_crossover = indicators.get('has_macd_crossover', False)
    volume_behavior = calculate_volume_behavior(data, '1h') if data else "insufficient_data"

    conditions = [
        support_distance <= 0.02,
        rsi_oversold,
        bullish_candle,
        macd_crossover,
        volume_behavior in ["decreasing", "increasing"]
    ]
    met_conditions = sum(1 for c in conditions if c)

    if met_conditions >= 3:
        action = "comprar"
        confidence = 80 + (met_conditions - 3) * 5
        explanation = (f"Compra en {symbol}: {met_conditions}/5 condiciones cumplidas "
                       f"(Soporte: {support_distance:.2%}, RSI: {rsi}, "
                       f"Vela alcista: {bullish_candle}, MACD cruce: {macd_crossover}, "
                       f"Volumen: {volume_behavior})")
    else:
        action = "mantener"
        confidence = 60
        explanation = (f"Mantener {symbol}: Solo {met_conditions}/5 condiciones cumplidas "
                       f"(Soporte: {support_distance:.2%}, RSI: {rsi}, "
                       f"Vela alcista: {bullish_candle}, MACD cruce: {macd_crossover}, "
                       f"Volumen: {volume_behavior})")

    return action, confidence, explanation

def calculate_established_strategy(indicators, data=None):
    rsi = indicators.get('rsi', None)
    relative_volume = indicators.get('relative_volume', None)
    price_trend = indicators.get('price_trend', 'insufficient_data')
    short_volume_trend = indicators.get('short_volume_trend', 'insufficient_data')
    current_price = indicators.get('current_price', 0)
    support_level = indicators.get('support_level', None)
    adx = indicators.get('adx', None)
    symbol = indicators.get('symbol', 'desconocido')

    # Adjusted thresholds for established coins
    MIN_ADX = 20
    MIN_RELATIVE_VOLUME = 1.0  # Lowered to allow more trades
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
        3 * (relative_volume > 1.0 if relative_volume else False),
        2 * (short_volume_trend == "increasing"),
        1 * (price_trend == "increasing"),
        3 * (support_distance <= 0.03),
        1 * (adx > 30 if adx else False),
        -1 * (rsi > 70 if rsi else False),  # Penalty for overbought
        2 * oversold,
        1 * volume_spike
    ]
    signals_score = sum(weighted_signals)

    # Decision with adjusted threshold
    if signals_score >= 5:  # Lowered to allow more trades
        action = "comprar"
        base_confidence = 80 if signals_score < 7 else 90
        explanation = f"Compra fuerte (establecido): Volumen={relative_volume}, ADX={adx}, soporte_dist={support_distance:.2%}, RSI={rsi}, Sobrevendido={oversold}, Pico de volumen={volume_spike} para {symbol}"
    else:
        action = "mantener"
        base_confidence = 60
        explanation = f"Insuficiente (establecido): Volumen={relative_volume}, ADX={adx}, soporte_dist={support_distance:.2%}, RSI={rsi}, Sobrevendido={oversold}, Pico de volumen={volume_spike}, puntaje={signals_score} para {symbol}"

    return action, base_confidence, explanation

def evaluate_missed_opportunity(symbol, initial_price, confidence, explanation, indicators):
    time.sleep(18000)
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
                return ohlcv
            logging.warning(f"Datos vacíos o None para {symbol} en {timeframe}, intento {attempt + 1}/{max_retries}")
        except Exception as e:
            logging.error(f"Error al obtener OHLCV para {symbol} en {timeframe}, intento {attempt + 1}/{max_retries}: {e}")
        time.sleep(2 ** attempt)
    return None

buy_lock = threading.Lock()

def demo_trading(high_volume_symbols=None):
    logging.info("Iniciando trading en segundo plano para todos los activos USDT relevantes...")
    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
    logging.info(f"Saldo USDT disponible: {usdt_balance}")
    if usdt_balance < MIN_NOTIONAL:
        logging.warning("Saldo insuficiente en USDT para alcanzar MIN_NOTIONAL.")
        return False

    reserve = 50
    available_for_trading = max(usdt_balance - reserve, 0)
    logging.info(f"Disponible para trading: {available_for_trading}, se deja una reserva de {reserve}")

    open_trades = get_open_trades()
    logging.info(f"Operaciones abiertas actualmente: {open_trades}")
    if open_trades >= MAX_OPEN_TRADES:
        logging.info("Límite de operaciones abiertas alcanzado.")
        return False

    if high_volume_symbols is None:
        high_volume_symbols = choose_best_cryptos(base_currency="USDT", top_n=300)

    budget_per_trade = available_for_trading / (MAX_OPEN_TRADES - open_trades) if (MAX_OPEN_TRADES - open_trades) > 0 else available_for_trading
    selected_cryptos = [s for s in high_volume_symbols if s.split('/')[0] in ESTABLISHED_COINS]
    logging.info(f"Presupuesto por operación: {budget_per_trade}")
    balance = exchange.fetch_balance()['free']
    logging.info(f"Balance actual: {balance}")

    failed_conditions_count = {}
    symbols_processed = 0

    for i in range(0, len(selected_cryptos), 10):
        batch = selected_cryptos[i:i+10]
        for symbol in batch:
            try:
                open_trades = get_open_trades()
                logging.info(f"Operaciones abiertas antes de procesar {symbol}: {open_trades}")
                if open_trades >= MAX_OPEN_TRADES:
                    logging.info("Límite de operaciones abiertas alcanzado.")
                    return False

                logging.info(f"Procesando {symbol}...")
                base_asset = symbol.split('/')[0]
                if base_asset in balance and balance[base_asset] > 0:
                    logging.info(f"Se omite {symbol} porque ya tienes una posición abierta.")
                    continue

                conditions = {}
                daily_volume = fetch_volume(symbol)
                conditions['daily_volume >= 250000'] = daily_volume is not None and daily_volume >= 250000
                if not conditions['daily_volume >= 250000']:
                    logging.info(f"Se omite {symbol} por volumen insuficiente: {daily_volume}")
                    failed_conditions_count['daily_volume >= 250000'] = failed_conditions_count.get('daily_volume >= 250000', 0) + 1
                    continue

                order_book_data = fetch_order_book_data(symbol)
                conditions['order_book_available'] = order_book_data is not None
                if not conditions['order_book_available']:
                    logging.warning(f"Se omite {symbol} por fallo en datos del libro de órdenes")
                    failed_conditions_count['order_book_available'] = failed_conditions_count.get('order_book_available', 0) + 1
                    continue

                conditions['depth >= 1000'] = order_book_data['depth'] >= 1000
                if not conditions['depth >= 1000']:
                    logging.warning(f"Profundidad baja para {symbol}: {order_book_data['depth']}, pero se evalúa de todos modos")
                else:
                    logging.info(f"Profundidad aceptable para {symbol}: {order_book_data['depth']}")

                current_price = fetch_price(symbol)
                conditions['price_available'] = current_price is not None
                if not conditions['price_available']:
                    logging.warning(f"Se omite {symbol} por no obtener precio")
                    failed_conditions_count['price_available'] = failed_conditions_count.get('price_available', 0) + 1
                    continue

                try:
                    current_price = float(current_price)
                    spread = float(order_book_data['spread']) if order_book_data['spread'] is not None else float('inf')
                    imbalance = float(order_book_data['imbalance']) if order_book_data['imbalance'] is not None else 0
                    depth = float(order_book_data['depth'])
                except (ValueError, TypeError) as e:
                    logging.error(f"Error al convertir datos numéricos para {symbol}: {e}")
                    continue

                conditions['spread <= 0.005 * price'] = spread <= 0.005 * current_price
                if not conditions['spread <= 0.005 * price']:
                    logging.info(f"Se omite {symbol} por spread alto: {spread}")
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
                    failed_conditions_count['data_available'] = failed_conditions_count.get('data_available', 0) + 1
                    continue

                df_slice = data.get('1h', data.get('4h', data.get('1d')))
                if df_slice.empty:
                    logging.warning(f"No hay datos válidos en ningún timeframe para {symbol}")
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
                    except Exception as e:
                        logging.error(f"Error al calcular volume_trend para {symbol}: {e}", exc_info=True)
                        volume_trend = "insufficient_data"
                else:
                    logging.debug(f"No hay suficientes datos para calcular volume_trend para {symbol}: {len(volume_series)} velas")

                if len(price_series) >= 10:
                    last_10_price = price_series[-10:]
                    try:
                        slope_price, _, _, _, _ = linregress(range(10), last_10_price)
                        price_trend = "increasing" if slope_price > 0.01 else "decreasing" if slope_price < -0.01 else "stable"
                        logging.debug(f"Price trend calculado para {symbol}: slope={slope_price}, trend={price_trend}")
                    except Exception as e:
                        logging.error(f"Error al calcular price_trend para {symbol}: {e}", exc_info=True)
                        price_trend = "insufficient_data"
                else:
                    logging.debug(f"No hay suficientes datos para calcular price_trend para {symbol}: {len(price_series)} velas")

                support_level = detect_support_level(data, price_series, window=15, max_threshold_multiplier=1.0)
                support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else None

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

                conditions_str = "\n".join([f"{key}: {'Sí' if value is True else 'No' if value is False else 'Desconocido'}" for key, value in sorted(conditions.items())])
                logging.info(f"Condiciones evaluadas para {symbol}:\n{conditions_str}")

                for key, value in conditions.items():
                    logging.debug(f"Condición {key} para {symbol}: valor={value}, tipo={type(value)}")

                base_currency = symbol.split('/')[0]
                if base_currency in ESTABLISHED_COINS:
                    action, confidence, explanation = calculate_established_strategy(indicators, data)
                else:
                    action, confidence, explanation = calculate_adaptive_strategy(indicators, data)
                logging.info(f"Decisión inicial para {symbol}: {action} (Confianza: {confidence}%) - {explanation}")

                gpt_conditions = {
                    'action is "mantener"': action == "mantener",
                    'Relative volume is None or low (< 1.8)': relative_volume is None or (relative_volume is not None and relative_volume < 1.8),
                    'No recent MACD crossover': not has_crossover,
                    'Far from support (> 0.03)': support_distance is None or support_distance > 0.03
                }

                gpt_conditions_str = "\n".join([f"{key}: {'Sí' if value else 'No'}" for key, value in gpt_conditions.items()])
                total_gpt_conditions = len(gpt_conditions)
                passed_gpt_conditions = sum(1 for value in gpt_conditions.values() if value)
                failed_gpt_conditions = [key for key, value in gpt_conditions.items() if not value]
                failed_gpt_conditions_str = ", ".join(failed_gpt_conditions) if failed_gpt_conditions else "Ninguna"
                logging.info(f"Condiciones para llamar a gpt_decision_buy en {symbol}: Pasadas {passed_gpt_conditions} de {total_gpt_conditions}:\n{gpt_conditions_str}\nNo se cumplieron: {failed_gpt_conditions_str}")

                if passed_gpt_conditions == total_gpt_conditions:
                    prepared_text = gpt_prepare_data(data, indicators)
                    action, confidence, explanation = gpt_decision_buy(prepared_text)
                    logging.info(f"Resultado de gpt_decision_buy para {symbol}: Acción={action}, Confianza={confidence}%, Explicación={explanation}")

                logging.info(f"Decisión final para {symbol}: {action} (Confianza: {confidence}%) - {explanation}")
                logging.debug(f"Verificación post-decisión para {symbol}: action={action}, confidence={confidence}, explanation={explanation}, conditions={conditions}")

                if action == "comprar" and confidence >= 80:
                    with buy_lock:
                        open_trades = get_open_trades()
                        if open_trades >= MAX_OPEN_TRADES:
                            logging.info(f"Límite de operaciones abiertas alcanzado, no se ejecuta compra para {symbol}.")
                            return False

                        confidence_factor = confidence / 100
                        if current_price is None or current_price <= 0:
                            logging.error(f"Precio actual inválido para {symbol} ({current_price}), omitiendo operación")
                            continue

                        atr_value = atr if atr is not None else 0.02 * current_price
                        if atr_value <= 0 or current_price <= 0:
                            volatility_factor = 1.0
                            logging.warning(f"ATR o precio inválido para {symbol}, usando volatility_factor por defecto: {volatility_factor}")
                        else:
                            volatility_factor = min(2.0, (atr_value / current_price * 100))
                        size_multiplier = confidence_factor * volatility_factor
                        adjusted_budget = budget_per_trade * size_multiplier
                        min_amount_for_notional = MIN_NOTIONAL / current_price
                        target_amount = max(adjusted_budget / current_price, min_amount_for_notional)
                        amount = min(target_amount, 0.10 * usdt_balance / current_price)
                        trade_value = amount * current_price

                        logging.info(f"Intentando compra para {symbol}: amount={amount}, trade_value={trade_value}, confidence={confidence}%, volatility_factor={volatility_factor:.2f}x")
                        if trade_value >= MIN_NOTIONAL or (trade_value < MIN_NOTIONAL and trade_value >= usdt_balance):
                            order = execute_order_buy(symbol, amount, indicators, confidence)
                            if order:
                                logging.info(f"Compra ejecutada para {symbol}: {explanation}")
                                send_telegram_message(f"✅ *Compra* `{symbol}`\nConfianza: `{confidence}%`\nCantidad: `{amount}`\nExplicación: `{explanation}`")
                                dynamic_trailing_stop(symbol, order['filled'], order['price'], order['trade_id'], indicators)
                                usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
                                available_for_trading = max(usdt_balance - reserve, 0)
                                budget_per_trade = available_for_trading / (MAX_OPEN_TRADES - get_open_trades()) if (MAX_OPEN_TRADES - get_open_trades()) > 0 else available_for_trading
                                logging.info(f"Rebalanceo: Nuevo saldo USDT={usdt_balance}, presupuesto por trade={budget_per_trade}")
                            else:
                                logging.error(f"Error al ejecutar compra para {symbol}: orden no completada")
                        else:
                            logging.info(f"Compra no ejecutada para {symbol}: valor de la operación ({trade_value}) < MIN_NOTIONAL ({MIN_NOTIONAL}) y saldo insuficiente")

                failed_conditions = [key for key, value in conditions.items() if not value]
                for condition in failed_conditions:
                    failed_conditions_count[condition] = failed_conditions_count.get(condition, 0) + 1
                symbols_processed += 1

            except Exception as e:
                logging.error(f"Error en demo_trading para {symbol}: {e}", exc_info=True)
                continue

        if 'data' in locals():
            del data
        time.sleep(30)

    if symbols_processed > 0 and failed_conditions_count:
        most_common_condition = max(failed_conditions_count, key=failed_conditions_count.get)
        most_common_count = failed_conditions_count[most_common_condition]
        summary_message = (f"Resumen final: Después de procesar {symbols_processed} símbolos, "
                           f"la condición más común que impidió operaciones fue '{most_common_condition}' "
                           f"con {most_common_count} ocurrencias ({(most_common_count / symbols_processed) * 100:.1f}%).")
        logging.info(summary_message)

        with open("trade_blockers_summary.txt", "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {summary_message}\n")
            f.write(f"Detalles de condiciones fallidas: {dict(failed_conditions_count)}\n\n")
    else:
        logging.info("No se procesaron símbolos o no hubo condiciones fallidas para analizar.")

    logging.info("Trading ejecutado correctamente en segundo plano para todos los activos USDT")
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
            return
        
        if trade_data[20] is None:
            logging.info(f"Trade {trade_id} no tiene venta registrada aún")
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
        elif profit_loss < 0 and buy_data['rsi'] and buy_data['rsi'] > 75:
            RSI_THRESHOLD = min(85, RSI_THRESHOLD + 5)
            logging.info(f"RSI_THRESHOLD aumentado a {RSI_THRESHOLD} por operación perdedora con RSI tardío")

        gpt_prompt = f"""
        Analiza esta transacción de `{buy_data['symbol']}` (ID: {trade_id}) para determinar por qué fue un éxito o un fracaso, si es exito dime cual fue el acierto y si fue un fracaso dime que se deberia corregir en los datos agregar para tomar un amejor decision:
        - Precio de compra: {buy_data['buy_price']}
        - Precio de venta: {sell_price}
        - Cantidad: {buy_data['amount']}
        - Ganancia/Pérdida: {profit_loss:.2f} USDT
        - RSI: {buy_data['rsi'] if buy_data['rsi'] else 'N/A'}
        - ADX: {buy_data['adx'] if buy_data['adx'] else 'N/A'}
        - ATR: {buy_data['atr'] if buy_data['atr'] else 'N/A'}
        - Volumen relativo: {buy_data['relative_volume'] if buy_data['relative_volume'] else 'N/A'}
        - Divergencia: {buy_data['divergence'] if buy_data['divergence'] else 'N/A'}
        - Posición BB: {buy_data['bb_position'] if buy_data['bb_position'] else 'N/A'}
        - Confianza: {buy_data['confidence'] if buy_data['confidence'] else 'N/A'}
        - Cruce MACD: {'Sí' if buy_data['has_macd_crossover'] else 'No'}
        - Velas desde cruce MACD: {buy_data['candles_since_crossover'] if buy_data['candles_since_crossover'] else 'N/A'}
        - Tendencia de volumen: {buy_data['volume_trend'] if buy_data['volume_trend'] else 'N/A'}
        - Tendencia de precio: {buy_data['price_trend'] if buy_data['price_trend'] else 'N/A'}
        - Tendencia de volumen corto: {buy_data['short_volume_trend'] if buy_data['short_volume_trend'] else 'N/A'}
        - Nivel de soporte: {buy_data['support_level'] if buy_data['support_level'] else 'N/A'}
        - Spread: {buy_data['spread'] if buy_data['spread'] else 'N/A'}
        - Imbalance: {buy_data['imbalance'] if buy_data['imbalance'] else 'N/A'}
        - Profundidad: {buy_data['depth'] if buy_data['depth'] else 'N/A'}
        Responde SOLO con un JSON válido sin etiqueta '''json''':
        {{"resultado": "éxito", "razon": "Compramos {buy_data['symbol']} a {buy_data['buy_price']} con RSI {buy_data['rsi']} y volumen fuerte {buy_data['relative_volume']}, vendimos a {sell_price} por una ganancia de {profit_loss:.2f} USDT gracias a un timing efectivo.", "confianza": 85}}
        o
        {{"resultado": "fracaso", "razon": "Compramos {buy_data['symbol']} a {buy_data['buy_price']} con RSI {buy_data['rsi']} y volumen débil {buy_data['relative_volume']}, vendimos a {sell_price} por una pérdida de {profit_loss:.2f} USDT por falta de momentum.", "confianza": 75}}
        """

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": gpt_prompt}],
            temperature=0
        )
        raw_response = response.choices[0].message.content.strip()
        send_telegram_message(raw_response)
        outcome = json.loads(raw_response)
        
        resultado = outcome.get("resultado", "desconocido").lower()
        razon = outcome.get("razon", "Sin análisis disponible")
        confianza = outcome.get("confianza", 50)

        telegram_message = f"📊 *Análisis de Resultado* `{buy_data['symbol']}` (ID: {trade_id})\n" \
                          f"Resultado: {'Éxito' if is_profitable else 'Fracaso'}\n" \
                          f"Ganancia/Pérdida: {profit_loss:.2f} USDT\n" \
                          f"Razón: {razon}\n" \
                          f"Confianza: {confianza}%"
        send_telegram_message(telegram_message)
        logging.info(f"Análisis de transacción {trade_id}: {resultado} - {razon} (Confianza: {confianza}%)")

    except Exception as e:
        logging.error(f"Error al analizar el resultado de la transacción {trade_id}: {e}")
    finally:
        conn.close()

def gpt_prepare_data(data, indicators):
    combined_data = ""
    for tf, df in data.items():
        if not df.empty:
            combined_data += f"\nDatos de {tf} (últimos 3):\n{df.tail(3).to_string(index=False)}\n"
    prompt = f"""
    Analiza los datos y decide si comprar esta criptomoneda para maximizar ganancias a corto plazo en cualquier activo USDT en Binance:
    {combined_data}
    Indicadores actuales:
    - RSI: {indicators.get('rsi', 'No disponible')}
    - ADX: {indicators.get('adx', 'No disponible')}
    - Divergencias: {indicators.get('divergence', 'No disponible')}
    - Volumen relativo: {indicators.get('relative_volume', 'No disponible')}
    - Precio actual: {indicators.get('current_price', 'No disponible')}
    - Cruce alcista reciente de MACD: {indicators.get('has_macd_crossover', 'No disponible')}
    - Velas desde el cruce: {indicators.get('candles_since_crossover', 'No disponible')}
    - Spread: {indicators.get('spread', 'No disponible')}
    - Imbalance (bids/asks): {indicators.get('imbalance', 'No disponible')}
    - Profundidad del libro: {indicators.get('depth', 'No disponible')}
    - Tendencia de volumen: {indicators.get('volume_trend', 'No disponible')}
    - Tendencia de precio: {indicators.get('price_trend', 'No disponible')}
    - Tendencia de volumen corto: {indicators.get('short_volume_trend', 'No disponible')}
    - Nivel de soporte: {indicators.get('support_level', 'No disponible')}
    - ROC: {indicators.get('roc', 'No disponible')}
    """
    return prompt
def gpt_decision_buy(prepared_text):
    prompt = f"""
    Eres un experto en trading de criptomonedas de alto riesgo. Basándote en los datos para un activo USDT en Binance:
    {prepared_text}
    Decide si "comprar" o "mantener" para maximizar ganancias a corto plazo. Prioriza tendencias fuertes con volumen relativo alto (> 3.0) y proximidad al soporte (<= 0.15). Responde SOLO con un JSON válido sin '''json''' asi:
    {{"accion": "comprar", "confianza": 85, "explicacion": "Volumen relativo > 3.0, short_volume_trend 'increasing', price_trend 'increasing', distancia al soporte <= 0.15, indican oportunidad de ganancia rápida"}}
    Criterios ajustados:
    - Compra si: volumen relativo > 3.0, short_volume_trend es 'increasing' o 'stable', price_trend es 'increasing', distancia relativa al soporte <= 0.15, profundidad > 3000, y spread <= 0.5% del precio (0.005 * precio). RSI > 70 es un bono, no un requisito.
    - Mantener si: volumen relativo <= 3.0, short_volume_trend es 'decreasing', price_trend no es 'increasing', distancia relativa al soporte > 0.15, profundidad <= 3000, o spread > 0.5% del precio.
    - Evalúa liquidez con profundidad (>3000) y spread (<=0.5% del precio).
    - Asigna confianza >80 solo si volumen relativo > 3.0, soporte cercano (<= 0.15), y al menos 3 condiciones se cumplen; usa 60-79 para riesgos moderados (al menos 2 condiciones); de lo contrario, usa 50. Suma 10 a la confianza si RSI > 70.
    - Ignora el cruce MACD como requisito; prioriza momentum y soporte.
    """

    max_retries = 2
    timeout_sec = 5

    for attempt in range(max_retries + 1):
        try:
            response = with_timeout(
                client.chat.completions.create,
                kwargs={
                    "model": GPT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0
                },
                timeout_sec=timeout_sec
            )
            raw_response = response.choices[0].message.content.strip()
            decision = json.loads(raw_response)

            accion = decision.get("accion", "mantener").lower()
            confianza = decision.get("confianza", 50)
            explicacion = decision.get("explicacion", "Respuesta incompleta")

            if accion not in ["comprar", "mantener"]:
                accion = "mantener"
            if not isinstance(confianza, (int, float)) or confianza < 50 or confianza > 100:
                confianza = 50
                explicacion = "Confianza fuera de rango, ajustada a 50"

            try:
                if "short_volume_trend" in prepared_text and "increasing" not in prepared_text.lower() and "stable" not in prepared_text.lower():
                    return "mantener", 50, "Short volume trend no es 'increasing' ni 'stable', overriding GPT"
                if "relative_volume" in prepared_text:
                    rel_vol_str = prepared_text.split("Volumen relativo: ")[1].split("\n")[0]
                    relative_volume = float(rel_vol_str) if rel_vol_str.replace('.', '').replace('-', '').isdigit() else 0
                    if relative_volume <= 3.0:
                        return "mantener", 50, "Relative volume demasiado bajo (<= 3.0), overriding GPT"
                if "price_trend" in prepared_text and "increasing" not in prepared_text.lower():
                    return "mantener", 50, "Price trend no es 'increasing', overriding GPT"
                if "distancia relativa al soporte" in prepared_text.lower():
                    dist_str = prepared_text.split("distancia relativa al soporte: ")[1].split("\n")[0] if "distancia relativa al soporte: " in prepared_text.lower() else "1.0"
                    support_distance = float(dist_str) if dist_str.replace('.', '').replace('-', '').isdigit() else 1.0
                    if support_distance > 0.15:
                        return "mantener", 50, "Lejos del soporte (> 0.15), overriding GPT"
                if "profundidad" in prepared_text:
                    depth_str = prepared_text.split("Profundidad del libro: ")[1].split("\n")[0]
                    depth = float(depth_str) if depth_str.replace('.', '').replace('-', '').isdigit() else 0
                    if depth <= 3000:
                        return "mantener", 50, "Profundidad demasiado baja (<= 3000), overriding GPT"
                if "spread" in prepared_text:
                    spread_str = prepared_text.split("Spread: ")[1].split("\n")[0]
                    spread = float(spread_str) if spread_str.replace('.', '').replace('-', '').isdigit() else float('inf')
                    current_price_str = prepared_text.split("Precio actual: ")[1].split("\n")[0]
                    current_price = float(current_price_str) if current_price_str.replace('.', '').replace('-', '').isdigit() else 1
                    if spread > 0.005 * current_price:
                        return "mantener", 50, "Spread demasiado alto (> 0.5% del precio), overriding GPT"
            except (ValueError, IndexError) as e:
                logging.warning(f"Error al validar condiciones en prepared_text: {e}, usando decisión predeterminada")
                return "mantener", 50, "Error en validación de condiciones, manteniendo por seguridad"

            return accion, confianza, explicacion

        except json.JSONDecodeError as e:
            logging.error(f"Intento {attempt + 1} fallido: Respuesta de GPT no es JSON válido - {raw_response}")
            if attempt == max_retries:
                return "mantener", 50, f"Error en formato JSON tras {max_retries + 1} intentos"
        except requests.Timeout:
            logging.error(f"Intento {attempt + 1} fallido: Timeout de {timeout_sec} segundos en GPT")
            if attempt == max_retries:
                return "mantener", 50, f"Timeout tras {max_retries + 1} intentos"
        except Exception as e:
            logging.error(f"Error en GPT (intento {attempt + 1}): {e}")
            if attempt == max_retries:
                return "mantener", 50, "Error al procesar respuesta de GPT"
        time.sleep(2 ** attempt)

def with_timeout(func, kwargs, timeout_sec):
    start = time.time()
    result = [None]
    def target():
        result[0] = func(**kwargs)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_sec)
    if thread.is_alive():
        raise requests.Timeout(f"Function timed out after {timeout_sec} seconds")
    return result[0]

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
        except Exception as e:
            logging.error(f"Error en resumen periódico: {e}")
            send_telegram_message(f"⚠️ *Error en Resumen* `{str(e)}`")
        time.sleep(3600)

if __name__ == "__main__":
    high_volume_symbols = choose_best_cryptos(base_currency="USDT", top_n=300)
    threading.Thread(target=send_periodic_summary, daemon=True).start()
    demo_trading(high_volume_symbols)
    logging.shutdown()