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
import pandas_ta as ta  # Nueva librería reemplazando ta-lib
from elegir_cripto import choose_best_cryptos
from scipy.stats import linregress

# Configuración e Inicialización
load_dotenv()
GPT_MODEL = "gpt-4o-mini"
DB_NAME = "trading_real.db"

# Inicializar la base de datos (crear tabla si no existe)
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
            has_macd_crossover INTEGER,  -- 1 si hay cruce alcista reciente, 0 si no
            candles_since_crossover INTEGER,  -- Número de velas desde el cruce
            volume_trend TEXT,  -- Nueva columna para tendencia de volumen
            price_trend TEXT,   -- Nueva columna para tendencia de precio
            short_volume_trend TEXT,  -- Nueva columna para tendencia corta de volumen
            support_level REAL,  -- Nueva columna para nivel de soporte
            spread REAL,        -- Nueva columna para spread del libro
            imbalance REAL,     -- Nueva columna para imbalance del libro
            depth REAL          -- Nueva columna para profundidad del libro
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

logging.basicConfig(
    level=logging.INFO,
    filename="trading_real.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constantes actualizadas
MAX_DAILY_BUYS = 5  # Reducido de 10 para memoria baja
MIN_NOTIONAL = 10
RSI_THRESHOLD = 80  # Aumentado para permitir compras en sobrecompra (cryptos alcistas)
ADX_THRESHOLD = 15
VOLUME_GROWTH_THRESHOLD = 0.3  # Reducido para capturar volumen moderado en criptos

# Cache de decisiones
decision_cache = {}
CACHE_EXPIRATION = 300  # Reducido a 5 minutos para volatilidad

def detect_support_level(data, price_series, window=15, max_threshold_multiplier=2.5):
    """
    Detecta un nivel de soporte usando precios históricos y ajusta con ATR para la volatilidad.
    
    Args:
        data: Diccionario con DataFrames para timeframes '1h', '4h' y '1d' (resultado de fetch_and_prepare_data).
        price_series: Serie de precios de cierre ('close') para análisis.
        window: Ventana para detectar el mínimo (por defecto 15).
        
    Returns:
        float con el nivel de soporte o None si no se detecta.
    """
    if len(price_series) < window:
        logging.warning(f"Series too short for {price_series.name}: {len(price_series)} < {window}")
        return None

    recent_prices = price_series[-window:]
    min_price = recent_prices.min()
    current_price = price_series.iloc[-1]

    # Se intentará calcular ATR usando distintos timeframes
    timeframes = ['1h', '4h', '1d']
    atr_value = None
    used_tf = None

    for tf in timeframes:
        if tf not in data or data[tf].empty:
            logging.warning(f"No hay datos en {tf} para {price_series.name}")
            continue

        df = data[tf]
        if len(df) < 14:  # ATR requiere al menos 14 velas
            logging.warning(f"Datos insuficientes para ATR en {price_series.name} ({tf}): {len(df)} < 14")
            continue

        try:
            # Ordenar y eliminar duplicados si es necesario
            if not df.index.is_monotonic_increasing:
                df = df.sort_index()
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='first')]
                if len(df) < 14:
                    logging.warning(f"Datos insuficientes tras limpiar duplicados en {price_series.name} en {tf}")
                    continue

            # Para timeframes intradía, rellenar gaps
            if tf != '1d':
                expected_freq = pd.Timedelta('1h') if tf == '1h' else pd.Timedelta('4h')
                expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq)
                if len(expected_index) > len(df.index):
                    logging.warning(
                        f"Gaps detectados en {price_series.name} en {tf}: "
                        f"esperados={len(expected_index)}, reales={len(df.index)}"
                    )
                    df = df.reindex(expected_index, method='ffill').dropna(how='all')
                    if len(df) < 14:
                        logging.warning(
                            f"Datos insuficientes al llenar gaps para {price_series.name} en {tf}"
                        )
                        continue

            # Calcular ATR y rellenar NaN tanto hacia adelante como hacia atrás
            atr_series = ta.atr(df['high'], df['low'], df['close'], length=14)

            # *** AQUÍ ESTÁ LA DIFERENCIA: usamos ffill() y bfill() en vez de solo ffill() ***
            atr_series_filled = atr_series.ffill().bfill()

            logging.debug(
                f"Serie ATR para {price_series.name} en {tf} (después de ffill & bfill):\n{atr_series_filled}"
            )

            if atr_series_filled.isna().all():
                logging.error(f"ATR no calculado para {price_series.name} en {tf}: {atr_series}")
                continue

            atr_value = atr_series_filled.iloc[-1]
            used_tf = tf
            logging.debug(f"ATR calculado para {price_series.name} en {tf}: {atr_value}")
            break  # Usamos el primer timeframe con datos válidos

        except Exception as e:
            logging.error(f"Error al calcular ATR para {price_series.name} en {tf}: {e}")
            continue

    if atr_value is None:
        logging.warning(f"No se pudo calcular ATR para {price_series.name} en ningún timeframe, usando 2% por defecto")
        atr_value = current_price * 0.02

    # Umbral dinámico basado en ATR (capado a 5%)
    threshold = 1 + (atr_value * max_threshold_multiplier / current_price) if current_price > 0 else 1.02
    threshold = min(threshold, 1.05)

    logging.debug(
        f"Umbral soporte {price_series.name}: precio={current_price}, min_reciente={min_price}, "
        f"umbral={threshold:.3f}, timeframe={used_tf}"
    )
    return min_price if min_price < current_price * threshold else None

def calculate_short_volume_trend(volume_series, window=3):
    if len(volume_series) < window:
        return "insufficient_data"
    last_volume = volume_series.iloc[-1]
    avg_volume = volume_series[-window:].mean()
    if last_volume > avg_volume * 1.05:  # Ajustado de 1.1 a 1.05 (5% superior)
        return "increasing"
    elif last_volume < avg_volume * 0.95:  # Ajustado de 0.9 a 0.95 (5% inferior)
        return "decreasing"
    else:
        return "stable"

def fetch_order_book_data(symbol, limit=10):
    try:
        order_book = exchange.fetch_order_book(symbol, limit=limit)
        bids = order_book['bids']
        asks = order_book['asks']
        spread = asks[0][0] - bids[0][0] if bids and asks else None
        bid_volume = sum([volume for _, volume in bids])
        ask_volume = sum([volume for _, volume in asks])
        imbalance = bid_volume / ask_volume if ask_volume > 0 else None
        depth = bid_volume + ask_volume  # Profundidad total
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

def fetch_and_prepare_data(symbol, atr_length=7, rsi_length=14, bb_length=20, roc_length=7, limit=50):
    """
    Obtiene datos OHLCV para los timeframes '1h', '4h' y '1d' y calcula indicadores usando
    un número reducido de velas para ATR, RSI, Bollinger Bands y ROC, de modo que se pueda operar
    con el histórico limitado que permite la API de Binance.

    Args:
        symbol (str): Símbolo a consultar (ej. 'DOGE/USDT').
        atr_length (int): Número de velas para calcular el ATR (default 7).
        rsi_length (int): Número de velas para calcular el RSI (default 14, ajustado para criptos).
        bb_length (int): Número de velas para calcular Bollinger Bands (default 20, alineado con TradingView).
        roc_length (int): Número de velas para calcular ROC (default 7).
        limit (int): Número de velas a solicitar por timeframe (default 50).

    Returns:
        tuple(dict, pd.Series): 
          - Un diccionario con DataFrames para cada timeframe ('1h', '4h', '1d').
          - La serie de precios de cierre preferida (prioridad '1h', luego '4h', luego '1d').
          Si no hay datos válidos, retorna (None, None).
    """
    timeframes = ['1h', '4h', '1d']
    data = {}
    logging.debug(f"Inicio de fetch_and_prepare_data para {symbol}")

    for tf in timeframes:
        try:
            logging.debug(f"Iniciando fetch_ohlcv para {symbol} en {tf} con limit={limit}")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
            if not ohlcv or len(ohlcv) == 0:
                logging.warning(f"Datos vacíos para {symbol} en {tf}")
                continue

            # Crear DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            logging.debug(f"DataFrame para {symbol} en {tf} creado con {len(df)} velas.")

            if len(df) < 5:
                logging.warning(f"Datos insuficientes (<5 velas) para {symbol} en {tf}")
                continue

            # Rellenar gaps en el índice
            expected_freq = pd.Timedelta('1h') if tf == '1h' else pd.Timedelta('4h') if tf == '4h' else pd.Timedelta('1d')
            expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=expected_freq)
            if len(expected_index) > len(df.index):
                logging.warning(f"Gaps detectados en {symbol} en {tf}: esperados={len(expected_index)}, reales={len(df.index)}")
                df = df.reindex(expected_index, method='ffill').dropna(how='all')
                if len(df) < 5:
                    logging.warning(f"Datos insuficientes tras rellenar gaps para {symbol} en {tf}")
                    continue

            # Convertir columnas a numérico y manejar valores inválidos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # NaN si no es numérico
                if df[col].isna().any():
                    logging.warning(f"Valores NaN detectados en {col} para {symbol} en {tf}, rellenando")
                    df[col] = df[col].ffill().bfill()  # Rellenar hacia adelante y atrás

            # Asegurar índices únicos y ordenados
            if not df.index.is_unique:
                logging.warning(f"Índices duplicados para {symbol} en {tf}: {df.index[df.index.duplicated()].tolist()}")
                df = df[~df.index.duplicated(keep='first')]
                if len(df) < 5:
                    logging.warning(f"Datos insuficientes tras eliminar duplicados para {symbol} en {tf}")
                    continue
            if not df.index.is_monotonic_increasing:
                logging.warning(f"Índices no monótonos para {symbol} en {tf}, ordenando")
                df = df.sort_index()
                if len(df) < 5:
                    logging.warning(f"Datos insuficientes tras ordenar índices para {symbol} en {tf}")
                    continue

            logging.debug(f"Calculando indicadores para {symbol} en {tf}")

            # Calcular ATR
            if len(df) >= atr_length:
                atr_series = ta.atr(df['high'], df['low'], df['close'], length=atr_length)
                if atr_series is None or atr_series.isna().all():
                    logging.warning(f"ATR no calculado para {symbol} en {tf}, asignando NaN")
                    df['ATR'] = np.nan
                else:
                    df['ATR'] = atr_series.ffill().bfill()
                    logging.debug(f"ATR para {symbol} en {tf}: {df['ATR'].iloc[-1]}")
            else:
                logging.warning(f"Datos insuficientes para ATR (<{atr_length} velas) para {symbol} en {tf}")
                df['ATR'] = np.nan

            # Calcular RSI
            if len(df) >= rsi_length:
                rsi_series = ta.rsi(df['close'], length=rsi_length)
                if rsi_series is None or rsi_series.isna().all():
                    logging.warning(f"RSI no calculado para {symbol} en {tf}, asignando NaN")
                    df['RSI'] = np.nan
                else:
                    df['RSI'] = rsi_series.ffill().bfill()
                    logging.debug(f"RSI para {symbol} en {tf}: {df['RSI'].iloc[-1]}")
            else:
                logging.warning(f"Datos insuficientes para RSI (<{rsi_length} velas) para {symbol} en {tf}")
                df['RSI'] = np.nan

            # Calcular Bollinger Bands
            if len(df) >= bb_length:
                bb = ta.bbands(df['close'], length=bb_length, std=2)
                if bb is None or bb.empty:
                    logging.warning(f"Bollinger Bands no calculadas para {symbol} en {tf}")
                    df['BB_upper'] = df['BB_middle'] = df['BB_lower'] = np.nan
                else:
                    df['BB_upper'] = bb.get(f'BBU_{bb_length}_2.0', pd.Series(np.nan, index=df.index))
                    df['BB_middle'] = bb.get(f'BBM_{bb_length}_2.0', pd.Series(np.nan, index=df.index))
                    df['BB_lower'] = bb.get(f'BBL_{bb_length}_2.0', pd.Series(np.nan, index=df.index))
                    if df['BB_upper'].isna().all() or df['BB_lower'].isna().all():
                        logging.warning(f"Bollinger Bands sin valores válidos para {symbol} en {tf}")
                        df['BB_upper'] = df['BB_middle'] = df['BB_lower'] = np.nan
                    else:
                        logging.debug(f"Bollinger Bands para {symbol} en {tf}: Upper={df['BB_upper'].iloc[-1]}, Middle={df['BB_middle'].iloc[-1]}, Lower={df['BB_lower'].iloc[-1]}")
            else:
                logging.warning(f"Datos insuficientes para Bollinger Bands (<{bb_length} velas) para {symbol} en {tf}")
                df['BB_upper'] = df['BB_middle'] = df['BB_lower'] = np.nan

            # Calcular MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd is None or macd.empty:
                logging.warning(f"MACD no calculado para {symbol} en {tf}")
                df['MACD'] = df['MACD_signal'] = np.nan
            else:
                df['MACD'] = macd.get('MACD_12_26_9', pd.Series(np.nan, index=df.index))
                df['MACD_signal'] = macd.get('MACDs_12_26_9', pd.Series(np.nan, index=df.index))
                if df['MACD'].isna().all() or df['MACD_signal'].isna().all():
                    logging.warning(f"MACD sin valores válidos para {symbol} en {tf}")
                    df['MACD'] = df['MACD_signal'] = np.nan
                else:
                    logging.debug(f"MACD para {symbol} en {tf}: MACD={df['MACD'].iloc[-1]}, Signal={df['MACD_signal'].iloc[-1]}")

            # Calcular ROC
            if len(df) >= roc_length:
                roc_series = ta.roc(df['close'], length=roc_length)
                if roc_series is None or roc_series.isna().all():
                    logging.warning(f"ROC no calculado para {symbol} en {tf}, asignando NaN")
                    df['ROC'] = np.nan
                else:
                    df['ROC'] = roc_series.ffill().bfill()
                    logging.debug(f"ROC para {symbol} en {tf}: {df['ROC'].iloc[-1]}")
            else:
                logging.warning(f"Datos insuficientes para ROC (<{roc_length} velas) para {symbol} en {tf}")
                df['ROC'] = np.nan

            data[tf] = df
            logging.debug(f"{symbol} en {tf}: DataFrame final con {len(df)} velas obtenido.")

        except Exception as e:
            logging.error(f"Error procesando {symbol} en {tf}: {e}")
            continue  # Continúa con el siguiente timeframe

    if not data:
        last_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
        logging.error(f"No se obtuvieron datos válidos para {symbol} en ningún timeframe. Última respuesta OHLCV: {last_ohlcv[:2] if last_ohlcv else 'None'}")
        return None, None

    # Verificar si hay suficientes datos para al menos un indicador útil
    has_valid_data = any(
        len(df) >= max(atr_length, rsi_length, bb_length, roc_length) and 
        not df[['ATR', 'RSI', 'BB_upper', 'MACD', 'ROC']].isna().all().all() 
        for df in data.values()
    )
    if not has_valid_data:
        logging.warning(f"Datos insuficientes para indicadores en {symbol} a pesar de timeframes válidos")
        return None, None

    # Seleccionar serie de precios preferida
    logging.debug(f"Seleccionando serie de precios para {symbol}: se prefiere '1h'")
    if '1h' in data and not data['1h'].empty:
        price_series = data['1h']['close']
    elif '4h' in data and not data['4h'].empty:
        price_series = data['4h']['close']
    elif '1d' in data and not data['1d'].empty:
        price_series = data['1d']['close']
    else:
        logging.error(f"No se pudieron obtener series de precios para {symbol}")
        return None, None

    if price_series.empty or price_series.isna().all():
        logging.warning(f"Serie de precios vacía o inválida para {symbol} después de procesar timeframes")
        return None, None

    logging.debug(f"Datos finales para {symbol}: Último volumen={data[list(data.keys())[0]]['volume'].iloc[-1]}, Último precio={price_series.iloc[-1]}")
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
    if len(macd_series) < 2 or len(signal_series) < 2:  # Necesitamos al menos 2 velas para comparar
        return False, None
    for i in range(-1, -lookback-1, -1):
        if i < -len(macd_series) or i-1 < -len(macd_series):  # Evitar índices fuera de rango
            break
        if macd_series.iloc[i-1] <= signal_series.iloc[i-1] and macd_series.iloc[i] > signal_series.iloc[i]:
            return True, abs(i)
    return False, None

def get_daily_buys():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) FROM transactions_new WHERE action='buy' AND timestamp LIKE ?", (f"{today}%",))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def execute_order_buy(symbol, amount, indicators, confidence):
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        price = order.get("price", fetch_price(symbol))
        executed_amount = order.get("filled", amount)
        if price is None:
            logging.error(f"No se pudo obtener precio para {symbol} después de la orden")
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
                support_level, spread, imbalance, depth
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, "buy", price, executed_amount, timestamp, trade_id,
            indicators.get('rsi'), indicators.get('adx'), indicators.get('atr'),
            indicators.get('relative_volume'), indicators.get('divergence'), indicators.get('bb_position'), 
            confidence, 1 if indicators.get('has_macd_crossover') else 0, 
            indicators.get('candles_since_crossover'),
            indicators.get('volume_trend'), indicators.get('price_trend'), indicators.get('short_volume_trend'),
            indicators.get('support_level'), indicators.get('spread'), indicators.get('imbalance'), 
            indicators.get('depth')
        ))
        conn.commit()
        conn.close()
        
        logging.info(f"Compra ejecutada: {symbol} a {price} por {executed_amount} (ID: {trade_id})")
        return {"price": price, "filled": executed_amount, "trade_id": trade_id, "indicators": indicators}
    except Exception as e:
        logging.error(f"Error al ejecutar orden de compra para {symbol}: {e}")
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
                INSERT INTO transactions_new (symbol, action, price, amount, timestamp, trade_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, "sell", sell_price, amount, timestamp, trade_id))
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
            INSERT INTO transactions_new (symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, "sell", sell_price, amount, timestamp, trade_id, rsi, adx, atr))
        conn.commit()
        conn.close()
        
        logging.info(f"Venta ejecutada: {symbol} a {sell_price} (ID: {trade_id})")
        send_telegram_message(f"✅ *Venta Ejecutada* para `{symbol}`\nPrecio: `{sell_price}`\nCantidad: `{amount}`")
        analyze_trade_outcome(trade_id)  # Feedback al cerrar la operación
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")

def dynamic_trailing_stop(symbol, amount, purchase_price, trade_id, indicators):
    def trailing_logic():
        try:
            data, price_series = fetch_and_prepare_data(symbol)
            if data is None:
                logging.error(f"No se pudieron obtener datos para trailing stop de {symbol}")
                sell_symbol(symbol, amount, trade_id)
                return
            
            atr = data['1h']['ATR'].iloc[-1] if not pd.isna(data['1h']['ATR'].iloc[-1]) else 0
            volatility = atr / purchase_price * 100 if purchase_price else 0
            trailing_percent = max(1, min(5, volatility * 1.0))  # Dinámico, stop más ajustado
            stop_loss_percent = trailing_percent * 0.5
            highest_price = purchase_price
            activated = False

            entry_msg = f"🔄 *Trailing Stop Dinámico* para `{symbol}`\nPrecio Compra: `{purchase_price}`\nCantidad: `{amount}`\nTrailing: `{trailing_percent:.2f}%`\nStop Loss: `{stop_loss_percent:.2f}%`"
            logging.info(entry_msg)
            send_telegram_message(entry_msg)

            while True:
                current_price = fetch_price(symbol)
                if not current_price:
                    time.sleep(5)
                    continue

                if not activated and current_price < purchase_price * (1 - stop_loss_percent / 100):
                    sell_symbol(symbol, amount, trade_id)
                    logging.info(f"Stop loss ejecutado para {symbol} a {current_price}")
                    break

                if current_price >= purchase_price * 1.01:
                    activated = True

                if activated:
                    if current_price > highest_price:
                        highest_price = current_price
                    stop_price = highest_price * (1 - trailing_percent / 100)
                    if current_price < stop_price:
                        sell_symbol(symbol, amount, trade_id)
                        logging.info(f"Trailing stop ejecutado para {symbol} a {current_price}")
                        break
                time.sleep(5)
        except Exception as e:
            logging.error(f"Error en trailing stop de {symbol}: {e}")
            sell_symbol(symbol, amount, trade_id)

    threading.Thread(target=trailing_logic, daemon=True).start()

def get_cached_decision(symbol, indicators):
    if symbol in decision_cache:
        cached_time, cached_decision, cached_indicators = decision_cache[symbol]
        if time.time() - cached_time < CACHE_EXPIRATION:
            if all(abs(indicators.get(key, 0) - cached_indicators.get(key, 0)) / (cached_indicators.get(key, 1) or 1) < 0.05 for key in indicators):
                return cached_decision
    return None

def calculate_adaptive_strategy(indicators):
    rsi = indicators.get('rsi', None)
    adx = indicators.get('adx', None)
    relative_volume = indicators.get('relative_volume', None)
    has_macd_crossover = indicators.get('has_macd_crossover', False)
    macd = indicators.get('macd', None)
    macd_signal = indicators.get('macd_signal', None)
    roc = indicators.get('roc', None)
    price_trend = indicators.get('price_trend', 'insufficient_data')
    short_volume_trend = indicators.get('short_volume_trend', 'insufficient_data')

    is_trending = adx > 25 if adx is not None else False

    # Regla basada en tendencia corta de volumen
    if short_volume_trend == "increasing" and price_trend != "decreasing" and relative_volume > VOLUME_GROWTH_THRESHOLD:
        return "comprar", 80, "Volumen creciente a corto plazo con tendencia de precio no bajista"

    if is_trending:
        if (rsi is not None and rsi <= RSI_THRESHOLD) or (has_macd_crossover and macd > macd_signal and macd_signal > 0) or (roc is not None and roc > 0):
            return "comprar", 85, "Tendencia alcista confirmada por RSI bajo, cruce MACD, o ROC positivo"
        if rsi is not None and rsi > 70 and relative_volume > VOLUME_GROWTH_THRESHOLD and price_trend == "increasing":
            return "comprar", 75, "Compra en sobrecompra con volumen creciente y tendencia alcista"
        return "mantener", 50, "Sin señales claras de tendencia alcista"
    else:
        # Aquí se elimina la combinación de RSI y Bollinger Bands
        if rsi is not None and rsi < 30:
            return "comprar", 80, "Reversión confirmada por RSI en sobreventa"
        elif rsi is not None and rsi > 70:
            return "mantener", 50, "Evitar compra en sobrecompra extrema"
        return "mantener", 50, "Mercado en rango sin señales claras"


def fetch_ohlcv_with_retry(symbol, timeframe, limit=50, max_retries=3):
    for attempt in range(max_retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if ohlcv and len(ohlcv) > 0:
                return ohlcv
            logging.warning(f"Datos vacíos o None para {symbol} en {timeframe}, intento {attempt + 1}/{max_retries}")
        except Exception as e:
            logging.error(f"Error al obtener OHLCV para {symbol} en {timeframe}, intento {attempt + 1}/{max_retries}: {e}")
        time.sleep(2 ** attempt)  # Exponential backoff
    return None

def demo_trading(high_volume_symbols=None):
    logging.info("Iniciando trading en segundo plano para todos los activos USDT relevantes...")
    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
    logging.info(f"Saldo USDT disponible: {usdt_balance}")
    if usdt_balance < MIN_NOTIONAL:
        logging.warning("Saldo insuficiente en USDT.")
        return False

    reserve = 150  # Reserva para comisiones y posibles pérdidas
    available_for_trading = usdt_balance - reserve
    logging.info(f"Disponible para trading: {available_for_trading}, se deja una reserva de {reserve}")
    daily_buys = get_daily_buys()
    logging.info(f"Compras diarias realizadas: {daily_buys}")
    if daily_buys >= MAX_DAILY_BUYS:
        logging.info("Límite diario de compras alcanzado.")
        return False
    if high_volume_symbols is None:
        high_volume_symbols = choose_best_cryptos(base_currency="USDT", top_n=100)

    budget_per_trade = available_for_trading / (MAX_DAILY_BUYS - daily_buys)
    selected_cryptos = high_volume_symbols
    logging.info(f"Presupuesto por operación: {budget_per_trade}")
    balance = exchange.fetch_balance()['free']
    logging.info(f"Balance actual: {balance}")

    # Diccionario para rastrear condiciones fallidas por símbolo
    failed_conditions_count = {}
    symbols_processed = 0

    # Batch processing
    for i in range(0, len(selected_cryptos), 10):
        batch = selected_cryptos[i:i+10]
        for symbol in batch:
            logging.info(f"Procesando {symbol}...")
            base_asset = symbol.split('/')[0]
            # Corrección: Eliminamos la duplicación en la condición
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

            conditions['depth >= 2000'] = order_book_data['depth'] >= 2000
            if not conditions['depth >= 2000']:
                logging.info(f"Se omite {symbol} por profundidad insuficiente: {order_book_data['depth']}")
                failed_conditions_count['depth >= 2000'] = failed_conditions_count.get('depth >= 2000', 0) + 1
                continue

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

            conditions['spread <= 0.01 * price'] = spread <= 0.01 * current_price
            if not conditions['spread <= 0.01 * price']:
                logging.info(f"Se omite {symbol} por spread alto: {spread}")
                failed_conditions_count['spread <= 0.01 * price'] = failed_conditions_count.get('spread <= 0.01 * price', 0) + 1
                continue

            conditions['imbalance >= 1.0'] = imbalance >= 1.0
            if not conditions['imbalance >= 1.0']:
                logging.info(f"Se omite {symbol} por imbalance bajo: {imbalance}")
                failed_conditions_count['imbalance >= 1.0'] = failed_conditions_count.get('imbalance >= 1.0', 0) + 1
                continue

            data, price_series = fetch_and_prepare_data(symbol)
            conditions['data_available'] = data is not None and price_series is not None
            if not conditions['data_available']:
                logging.warning(f"Se omite {symbol} por datos insuficientes")
                failed_conditions_count['data_available'] = failed_conditions_count.get('data_available', 0) + 1
                continue

            df_slice = data.get('1h', data.get('4h', data.get('1d')))
            conditions['timeframe_available'] = not df_slice.empty
            if not conditions['timeframe_available']:
                logging.warning(f"No hay datos válidos en ningún timeframe para {symbol}")
                failed_conditions_count['timeframe_available'] = failed_conditions_count.get('timeframe_available', 0) + 1
                continue
            df_slice = df_slice.iloc[-100:]

            rsi = df_slice['RSI'].iloc[-1] if 'RSI' in df_slice and not pd.isna(df_slice['RSI'].iloc[-1]) else None
            adx = calculate_adx(df_slice) if not df_slice.empty else None
            atr = df_slice['ATR'].iloc[-1] if 'ATR' in df_slice and not pd.isna(df_slice['ATR'].iloc[-1]) else None
            volume_series = df_slice['volume']
            relative_volume = volume_series.iloc[-1] / volume_series[-10:].mean() if len(volume_series) >= 10 and volume_series[-10:].mean() != 0 else None
            divergence = detect_momentum_divergences(price_series, df_slice['RSI']) if 'RSI' in df_slice else "none"
            bb_position = get_bb_position(
                current_price,
                df_slice['BB_upper'].iloc[-1] if 'BB_upper' in df_slice and not pd.isna(df_slice['BB_upper'].iloc[-1]) else None,
                df_slice['BB_middle'].iloc[-1] if 'BB_middle' in df_slice and not pd.isna(df_slice['BB_middle'].iloc[-1]) else None,
                df_slice['BB_lower'].iloc[-1] if 'BB_lower' in df_slice and not pd.isna(df_slice['BB_lower'].iloc[-1]) else None
            )
            macd = df_slice['MACD'].iloc[-1] if 'MACD' in df_slice and not pd.isna(df_slice['MACD'].iloc[-1]) else None
            macd_signal = df_slice['MACD_signal'].iloc[-1] if 'MACD_signal' in df_slice and not pd.isna(df_slice['MACD_signal'].iloc[-1]) else None
            roc = df_slice['ROC'].iloc[-1] if 'ROC' in df_slice and not pd.isna(df_slice['ROC'].iloc[-1]) else None
            has_crossover, candles_since = has_recent_macd_crossover(
                df_slice['MACD'] if 'MACD' in df_slice else pd.Series(),
                df_slice['MACD_signal'] if 'MACD_signal' in df_slice else pd.Series(),
                lookback=5
            )
            short_volume_trend = calculate_short_volume_trend(volume_series) if len(volume_series) >= 3 else "insufficient_data"
            volume_trend = "insufficient_data"
            price_trend = "insufficient_data"
            if len(volume_series) >= 10:
                last_10_volume = volume_series[-10:]
                slope_volume, _, _, _, _ = linregress(range(10), last_10_volume)
                volume_trend = "increasing" if slope_volume > 0.01 else "decreasing" if slope_volume < -0.01 else "stable"
            if len(price_series) >= 10:
                last_10_price = price_series[-10:]
                slope_price, _, _, _, _ = linregress(range(10), last_10_price)
                price_trend = "increasing" if slope_price > 0.01 else "decreasing" if slope_price < -0.01 else "stable"
            short_price_trend = price_trend
            short_volume_trend_1h = calculate_short_volume_trend(volume_series, window=1) if len(volume_series) >= 1 else "insufficient_data"
            support_level = detect_support_level(data, price_series, window=15, max_threshold_multiplier=3.0)
            support_threshold = 1 + (atr * 3.0 / current_price) if atr and current_price > 0 else 1.10
            support_threshold = min(support_threshold, 1.15)

            conditions['support_near'] = support_level is not None and current_price <= support_level * support_threshold
            conditions['short_volume_trend_increasing'] = short_volume_trend == "increasing"
            conditions['price_trend_not_decreasing'] = price_trend != "decreasing"
            conditions['relative_volume > VOLUME_GROWTH_THRESHOLD'] = relative_volume is not None and relative_volume > VOLUME_GROWTH_THRESHOLD
            conditions['adx > 25'] = adx is not None and adx > 25
            conditions['rsi <= RSI_THRESHOLD'] = rsi is not None and rsi <= RSI_THRESHOLD
            conditions['macd_crossover'] = has_crossover and macd is not None and macd_signal is not None and macd > macd_signal and macd_signal > 0
            conditions['roc > 0'] = roc is not None and roc > 0
            conditions['rsi < 30 and bb_below_lower'] = rsi is not None and rsi < 30 and bb_position == "below_lower"
            conditions['rsi > 70 and bb_above_upper'] = rsi is not None and rsi > 70 and bb_position == "above_upper"

            conditions_str = "\n".join([f"{key}: {'Sí' if value else 'No'}" for key, value in sorted(conditions.items())])
            logging.info(f"Condiciones evaluadas para {symbol}:\n{conditions_str}\nValores: RSI={rsi}, ADX={adx}, RelVol={relative_volume}, ShortVolTrend={short_volume_trend}, PriceTrend={price_trend}, Support={support_level}, MACD={macd}, Signal={macd_signal}, Crossover={has_crossover}")

            indicators = {
                "rsi": rsi,
                "adx": adx,
                "atr": atr,
                "relative_volume": relative_volume,
                "divergence": divergence,
                "bb_position": bb_position,
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
                "short_price_trend": short_price_trend,
                "short_volume_trend_1h": short_volume_trend_1h,
                "roc": roc,
                "current_price": current_price
            }

            if not conditions['support_near']:
                logging.info(f"Precio de {symbol} ({current_price}) está por encima del umbral de soporte ({support_level * support_threshold:.3f})")
            if not conditions['short_volume_trend_increasing']:
                logging.info(f"Tendencia de volumen decreciente a corto plazo para {symbol}")

            action, confidence, explanation = calculate_adaptive_strategy(indicators)
            logging.info(f"Decisión inicial para {symbol}: {action} (Confianza: {confidence}%) - {explanation}")

            gpt_conditions = {
                'action == "mantener"': action == "mantener",
                'rsi is None or 30 < rsi < 70': rsi is None or (rsi is not None and 30 < rsi < 70),
                'relative_volume is None or < 0.5': relative_volume is None or (relative_volume is not None and relative_volume < 0.5),
                'no macd_crossover': not has_crossover
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
            else:
                logging.info(f"No se llamó a gpt_decision_buy para {symbol} debido a condiciones excluyentes no cumplidas")

            logging.info(f"Decisión final para {symbol}: {action} (Confianza: {confidence}%) - {explanation}")

            total_conditions = len(conditions)
            passed_conditions = sum(1 for value in conditions.values() if value)
            failed_conditions = [key for key, value in conditions.items() if not value]
            failed_conditions_str = ", ".join(failed_conditions) if failed_conditions else "Ninguna"
            logging.info(f"Resumen para {symbol}: Pasadas {passed_conditions} condiciones de {total_conditions}, no se cumplieron: {failed_conditions_str}")

            # Actualizar conteo de condiciones fallidas
            for condition in failed_conditions:
                failed_conditions_count[condition] = failed_conditions_count.get(condition, 0) + 1
            symbols_processed += 1

            if action == "comprar" and confidence >= 70:
                amount = min(budget_per_trade / current_price, 0.005 * usdt_balance / current_price)
                if amount * current_price >= MIN_NOTIONAL:
                    order = execute_order_buy(symbol, amount, indicators, confidence)
                    if order:
                        logging.info(f"Compra ejecutada para {symbol}: {explanation}")
                        send_telegram_message(f"✅ *Compra* `{symbol}`\nConfianza: `{confidence}%`\nExplicación: `{explanation}`")
                        dynamic_trailing_stop(symbol, order['filled'], order['price'], order['trade_id'], indicators)

        # Liberar memoria
        del data
        time.sleep(30)

    # Resumen final de condiciones fallidas más comunes
    if symbols_processed > 0 and failed_conditions_count:
        most_common_condition = max(failed_conditions_count, key=failed_conditions_count.get)
        most_common_count = failed_conditions_count[most_common_condition]
        summary_message = (f"Resumen final: Después de procesar {symbols_processed} símbolos, "
                           f"la condición más común que impidió operaciones fue '{most_common_condition}' "
                           f"con {most_common_count} ocurrencias ({(most_common_count / symbols_processed) * 100:.1f}%).")
        logging.info(summary_message)

        # Escribir en un archivo
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
            SELECT t1.*, t2.*
            FROM transactions_new t1
            LEFT JOIN transactions_new t2 ON t1.trade_id = t2.trade_id AND t2.action = 'sell'
            WHERE t1.trade_id = ? AND t1.action = 'buy'
        """, (trade_id,))
        trade_data = cursor.fetchone()
        
        if not trade_data:
            logging.warning(f"No se encontraron datos para el trade_id: {trade_id}")
            return

        buy_data = {
            'symbol': trade_data[1],  # symbol
            'buy_price': trade_data[3],  # price
            'amount': trade_data[4],  # amount
            'timestamp': trade_data[5],  # timestamp
            'rsi': trade_data[7],  # rsi
            'adx': trade_data[8],  # adx
            'atr': trade_data[9],  # atr
            'relative_volume': trade_data[10],  # relative_volume
            'divergence': trade_data[11],  # divergence
            'bb_position': trade_data[12],  # bb_position
            'confidence': trade_data[13],  # confidence
            'has_macd_crossover': trade_data[14],  # has_macd_crossover
            'candles_since_crossover': trade_data[15],  # candles_since_crossover
            'volume_trend': trade_data[16],  # volume_trend
            'price_trend': trade_data[17],  # price_trend
            'short_volume_trend': trade_data[18],  # short_volume_trend
            'support_level': trade_data[19],  # support_level
            'spread': trade_data[20],  # spread
            'imbalance': trade_data[21],  # imbalance
            'depth': trade_data[22]  # depth
        }

        sell_data = {
            'sell_price': trade_data[23] if trade_data[23] else None,  # price de la venta
            'sell_timestamp': trade_data[25] if trade_data[25] else None,  # timestamp de la venta
            'rsi_sell': trade_data[28],  # rsi de la venta
            'adx_sell': trade_data[29],  # adx de la venta
            'atr_sell': trade_data[30]  # atr de la venta
        }

        profit_loss = (sell_data['sell_price'] - buy_data['buy_price']) * buy_data['amount'] if sell_data['sell_price'] else 0
        is_profitable = profit_loss > 0

        gpt_prompt = f"""
        Analiza los datos de la transacción de `{buy_data['symbol']}` (ID: {trade_id}) para determinar por qué fue un éxito o un fracaso, proporcionando detalles específicos sobre qué hicimos bien o mal según nuestra estrategia. Responde SOLO con un JSON válido sin etiqueta '''json''':
        {{"resultado": "éxito", "razon": "Compramos {buy_data['symbol']} a {buy_data['buy_price']} debido a un cruce alcista de MACD y volumen creciente de {buy_data['relative_volume']}, vendimos a {sell_data['sell_price']} por una ganancia de {profit_loss:.2f} USDT, confirmando una estrategia de momentum efectiva.", "confianza": 85}}
        o
        {{"resultado": "fracaso", "razon": "Compramos {buy_data['symbol']} a {buy_data['buy_price']} con RSI {buy_data['rsi']} y volumen bajo de {buy_data['relative_volume']}, pero vendimos a {sell_data['sell_price']} por una pérdida de {profit_loss:.2f} USDT debido a una tendencia decreciente no detectada.", "confianza": 75}}

        Datos de compra:
        - Símbolo: {buy_data['symbol']}
        - Precio de compra: {buy_data['buy_price']}
        - Cantidad: {buy_data['amount']}
        - Timestamp de compra: {buy_data['timestamp']}
        - RSI: {buy_data['rsi']}
        - ADX: {buy_data['adx']}
        - ATR: {buy_data['atr']}
        - Volumen relativo: {buy_data['relative_volume']}
        - Divergencia: {buy_data['divergence']}
        - Posición BB: {buy_data['bb_position']}
        - Confianza: {buy_data['confidence']}
        - Cruce MACD: {'Sí' if buy_data['has_macd_crossover'] else 'No'}
        - Velas desde cruce MACD: {buy_data['candles_since_crossover']}
        - Tendencia de volumen: {buy_data['volume_trend']}
        - Tendencia de precio: {buy_data['price_trend']}
        - Tendencia de volumen corto: {buy_data['short_volume_trend']}
        - Nivel de soporte: {buy_data['support_level']}
        - Spread: {buy_data['spread']}
        - Imbalance: {buy_data['imbalance']}
        - Profundidad: {buy_data['depth']}

        Datos de venta (si aplica):
        - Precio de venta: {sell_data['sell_price']}
        - Timestamp de venta: {sell_data['sell_timestamp']}
        - RSI venta: {sell_data['rsi_sell']}
        - ADX venta: {sell_data['adx_sell']}
        - ATR venta: {sell_data['atr_sell']}

        Ganancia/Pérdida: {profit_loss:.2f} USDT
        """

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": gpt_prompt}],
            temperature=0
        )
        raw_response = response.choices[0].message.content.strip()
        outcome = json.loads(raw_response)
        
        resultado = outcome.get("resultado", "desconocido").lower()
        razon = outcome.get("razon", "Sin análisis disponible")
        confianza = outcome.get("confianza", 50)

        telegram_message = f"📊 *Análisis Detallado de Resultado* para `{buy_data['symbol']}` (ID: {trade_id})\n" \
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
    - Cruce alcista reciente de MACD: {indicators.get('macd_crossover', 'No disponible')}
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
    Eres un experto en trading de criptomonedas de alto riesgo. Basándote en los datos para cualquier activo USDT en Binance:
    {prepared_text}
    Decide si "comprar" o "mantener" para maximizar ganancias a corto plazo. Prioriza activos con alta volatilidad, volumen creciente (>0.2), o cruces alcistas recientes de MACD, incluso si RSI es > 70. Acepta riesgos moderados si el volumen y momentum a corto plazo son fuertes, especialmente cuando short_volume_trend es 'increasing' y price_trend no es 'decreasing'. Responde SOLO con un JSON válido sin '''json''' asi:
    {{"accion": "comprar", "confianza": 85, "explicacion": "Volumen creciente a corto plazo (>0.2) y tendencia de precio no bajista indican oportunidad de ganancia rápida en alta volatilidad"}}
    Criterios:
    - Compra si volumen relativo > 0.2, short_volume_trend es 'increasing', precio tendencia 'increasing' o 'stable', o cruce alcista MACD reciente, incluso con RSI > 70.
    - Mantener solo si todas las señales son débiles, negativas (e.g., short_volume_trend 'decreasing' o price_trend 'decreasing'), o hay sobre venta extrema (RSI < 20).
    - Evalúa profundidad (>2000) y spread (<1% del precio) para liquidez.
    - Ignora tendencias largas si short_volume_trend y volumen relativo indican momentum fuerte.
    """
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw_response = response.choices[0].message.content.strip()
            decision = json.loads(raw_response)
            
            accion = decision.get("accion", "mantener").lower()
            confianza = decision.get("confianza", 50)
            explicacion = decision.get("explicacion", "Respuesta incompleta")

            if accion not in ["comprar", "mantener"]:
                accion = "mantener"
            if not isinstance(confianza, (int, float)) or not 0 <= confianza <= 100:
                confianza = 50
                explicacion = "Confianza inválida, ajustada a 50"

            return accion, confianza, explicacion
        except json.JSONDecodeError as e:
            logging.error(f"Intento {attempt + 1} fallido: Respuesta de GPT no es JSON válido - {raw_response}")
            if attempt == max_retries:
                return "mantener", 50, f"Error en formato JSON tras {max_retries + 1} intentos"
        except Exception as e:
            logging.error(f"Error en GPT (intento {attempt + 1}): {e}")
            if attempt == max_retries:
                return "mantener", 50, "Error al procesar respuesta de GPT"
        time.sleep(1)

if __name__ == "__main__":
    high_volume_symbols = choose_best_cryptos(base_currency="USDT", top_n=100)
    demo_trading(high_volume_symbols)