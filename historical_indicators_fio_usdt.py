import ccxt
import pandas as pd
import numpy as np
import sqlite3
import os
import time
import logging
import pytz
from datetime import datetime
from dotenv import load_dotenv
import pandas_ta as ta
from scipy.stats import linregress

# Configuración
load_dotenv()
DB_NAME = "trading_analysis_ern.db"
logging.basicConfig(level=logging.INFO, filename="historical_analysis_ern.log", filemode="a", format="%(asctime)s - %(levelname)s - %(message)s")

exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY_REAL"),
    'secret': os.getenv("BINANCE_SECRET_KEY_REAL"),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})

# Constantes del programa principal
RSI_THRESHOLD = 80
ADX_THRESHOLD = 15
VOLUME_GROWTH_THRESHOLD = 0.3  # Ajustado para capturar crecimiento temprano de volumen

# Inicializar base de datos para almacenar resultados
def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price REAL,
            rsi REAL,
            adx REAL,
            atr REAL,
            relative_volume REAL,
            divergence TEXT,
            bb_position TEXT,
            macd REAL,
            macd_signal REAL,
            has_macd_crossover INTEGER,
            candles_since_crossover INTEGER,
            volume_trend TEXT,
            price_trend TEXT,
            short_volume_trend TEXT,
            support_level REAL,
            roc REAL,
            decision TEXT,
            confidence REAL,
            explanation TEXT,
            rejecting_variable TEXT
        )
    ''')
    conn.commit()
    conn.close()

initialize_db()

def fetch_historical_data(symbol, timeframe='5m', start_time=None, end_time=None, limit=1000):
    colombia_tz = pytz.timezone("America/Bogota")
    if start_time:
        adjusted_start_time = start_time - pd.Timedelta(minutes=70)  # 14 velas previas
        start_timestamp = int(adjusted_start_time.timestamp() * 1000)
    else:
        start_timestamp = None
    end_timestamp = int(end_time.timestamp() * 1000) if end_time else None
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=start_timestamp, limit=limit)
    if not ohlcv or len(ohlcv) == 0:
        logging.error(f"No data fetched for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert(colombia_tz)
    df = df[(df.index >= start_time) & (df.index <= end_time)]
    
    if df.empty:
        logging.warning(f"Empty DataFrame for {symbol} after filtering")
        return df

    # Calcular indicadores
    df['RSI'] = ta.rsi(df['close'], length=7).ffill().bfill()
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=7).ffill().bfill()
    bb = ta.bbands(df['close'], length=14, std=2)
    df['BB_upper'] = bb.get('BBU_14_2.0')
    df['BB_middle'] = bb.get('BBM_14_2.0')
    df['BB_lower'] = bb.get('BBL_14_2.0')
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd.get('MACD_12_26_9')
    df['MACD_signal'] = macd.get('MACDs_12_26_9')
    df['ROC'] = ta.roc(df['close'], length=7).ffill().bfill()
    
    return df

def calculate_adx(df):
    try:
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is None or 'ADX_14' not in adx or adx['ADX_14'].iloc[-1] is None:
            return None
        return adx['ADX_14'].iloc[-1] if not pd.isna(adx['ADX_14'].iloc[-1]) else None
    except Exception as e:
        logging.error(f"Error al calcular ADX: {e}")
        return None

def detect_momentum_divergences(price_series, rsi_series):
    price = np.array(price_series)
    rsi = np.array(rsi_series)
    divergences = []
    window = 5
    for i in range(window, len(price) - window):
        if (price[i] > max(price[i-window:i]) and price[i] > max(price[i+1:i+window+1]) and rsi[i] < rsi[i-window]):
            divergences.append(("bearish", i))
        elif (price[i] < min(price[i-window:i]) and price[i] < min(price[i+1:i+window+1]) and rsi[i] > rsi[i-window]):
            divergences.append(("bullish", i))
    return "bullish" if any(d[0] == "bullish" for d in divergences) else "bearish" if any(d[0] == "bearish" for d in divergences) else "none"

def get_bb_position(price, bb_upper, bb_middle, bb_lower):
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
        if macd_series.iloc[i-1] <= signal_series.iloc[i-1] and macd_series.iloc[i] > signal_series[i]:
            return True, abs(i)
    return False, None

def detect_support_level(df, price_series, window=15):
    recent_prices = price_series[-window:]
    min_price = recent_prices.min()
    current_price = price_series.iloc[-1]
    atr = df['ATR'].iloc[-1]
    threshold = 1 + (atr / current_price) if atr and current_price > 0 else 1.02
    threshold = min(threshold, 1.05)
    return min_price if min_price < current_price * threshold else None

def calculate_short_volume_trend(volume_series, window=3):
    if len(volume_series) < window:
        return "insufficient_data"
    last_volume = volume_series.iloc[-1]
    avg_volume = volume_series[-window:].mean()
    return "increasing" if last_volume > avg_volume * 1.05 else "decreasing" if last_volume < avg_volume * 0.95 else "stable"  # Ajustado a ±5%

def calculate_volume_trend(volume_series, window=10):
    if len(volume_series) < window:
        return "insufficient_data"
    slope_volume, _, _, _, _ = linregress(range(window), volume_series[-window:])
    return "increasing" if slope_volume > 0.01 else "decreasing" if slope_volume < -0.01 else "stable"

def calculate_price_trend(price_series, window=10):
    if len(price_series) < window:
        return "insufficient_data"
    slope_price, _, _, _, _ = linregress(range(window), price_series[-window:])
    return "increasing" if slope_price > 0.01 else "decreasing" if slope_price < -0.01 else "stable"

def calculate_adaptive_strategy(indicators):
    rsi = indicators.get('rsi')
    adx = indicators.get('adx')
    relative_volume = indicators.get('relative_volume')
    has_macd_crossover = indicators.get('has_macd_crossover', False)
    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')
    roc = indicators.get('roc')
    bb_position = indicators.get('bb_position', 'unknown')
    price_trend = indicators.get('price_trend', 'insufficient_data')
    short_volume_trend = indicators.get('short_volume_trend', 'insufficient_data')

    is_trending = adx > 25 if adx is not None else False

    # Nueva regla basada en tendencia corta de volumen y precio
    if short_volume_trend == "increasing" and price_trend != "decreasing" and relative_volume > VOLUME_GROWTH_THRESHOLD:
        return "comprar", 80, "Volumen creciente a corto plazo con tendencia de precio no bajista"

    if is_trending:
        if (rsi is not None and rsi <= RSI_THRESHOLD) or (has_macd_crossover and macd > macd_signal and macd_signal > 0) or (roc is not None and roc > 0):
            return "comprar", 85, "Tendencia alcista confirmada por RSI bajo, cruce MACD, o ROC positivo"
        if rsi is not None and rsi > 70 and relative_volume > VOLUME_GROWTH_THRESHOLD and price_trend == "increasing":
            return "comprar", 75, "Compra en sobrecompra con volumen creciente y tendencia alcista"
        return "mantener", 50, "Sin señales claras de tendencia alcista"
    else:
        if (rsi is not None and rsi < 30 and bb_position == "below_lower") or (rsi is not None and rsi > 70 and bb_position == "above_upper"):
            return "comprar" if bb_position == "below_lower" else "mantener", 80 if bb_position == "below_lower" else 50, "Reversión media confirmada por RSI y Bollinger Bands" if bb_position == "below_lower" else "Sin oportunidad de reversión"
        return "mantener", 50, "Mercado en rango sin señales claras"

def analyze_historical_data(symbol, start_time, end_time):
    df = fetch_historical_data(symbol, start_time=start_time, end_time=end_time)
    if df.empty:
        logging.error(f"No se obtuvieron datos para {symbol}")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Búfer mínimo de 26 velas (para MACD slow=26) o todo el DataFrame si es más corto
    buffer_size = 26

    for timestamp, row in df.iterrows():
        idx = df.index.get_loc(timestamp)
        start_idx = max(0, idx - buffer_size)  # Asegurar al menos 26 velas previas si es posible
        df_slice = df.iloc[start_idx:idx + 1]

        price = row['close']
        rsi = row['RSI']
        atr = row['ATR']
        macd = row['MACD']
        macd_signal = row['MACD_signal']
        roc = row['ROC']
        volume_series = df['volume'].iloc[start_idx:idx + 1]
        price_series = df['close'].iloc[start_idx:idx + 1]

        indicators = {
            'rsi': rsi,
            'adx': calculate_adx(df_slice),
            'atr': atr,
            'relative_volume': volume_series.iloc[-1] / volume_series[-10:].mean() if len(volume_series) >= 10 and volume_series[-10:].mean() != 0 else 1.0,  # Ajustado a 10 velas
            'divergence': detect_momentum_divergences(price_series, df['RSI'].iloc[start_idx:idx + 1]),
            'bb_position': get_bb_position(price, row['BB_upper'], row['BB_middle'], row['BB_lower']),
            'macd': macd,
            'macd_signal': macd_signal,
            'has_macd_crossover': has_recent_macd_crossover(df['MACD'].iloc[start_idx:idx + 1], df['MACD_signal'].iloc[start_idx:idx + 1])[0],
            'candles_since_crossover': has_recent_macd_crossover(df['MACD'].iloc[start_idx:idx + 1], df['MACD_signal'].iloc[start_idx:idx + 1])[1],
            'volume_trend': calculate_volume_trend(volume_series),
            'price_trend': calculate_price_trend(price_series),
            'short_volume_trend': calculate_short_volume_trend(volume_series),
            'support_level': detect_support_level(df_slice, price_series),  # Opcional, no bloqueante
            'roc': roc
        }

        # Decisión de compra según estrategia
        action, confidence, explanation = calculate_adaptive_strategy(indicators)

        # Filtros adicionales (sin requerir soporte estricto)
        rejecting_variable = None
        if action == "comprar" and confidence >= 70:
            if indicators['short_volume_trend'] == "decreasing":
                action, rejecting_variable = "mantener", "short_volume_trend decreasing"
            elif indicators['price_trend'] == "decreasing":
                action, rejecting_variable = "mantener", "price_trend decreasing"

        # Guardar en la base de datos
        cursor.execute('''
            INSERT INTO historical_analysis (
                timestamp, symbol, price, rsi, adx, atr, relative_volume, divergence, bb_position,
                macd, macd_signal, has_macd_crossover, candles_since_crossover, volume_trend,
                price_trend, short_volume_trend, support_level, roc, decision, confidence,
                explanation, rejecting_variable
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp.isoformat(), symbol, price, rsi, indicators['adx'], atr, indicators['relative_volume'],
            indicators['divergence'], indicators['bb_position'], macd, macd_signal,
            1 if indicators['has_macd_crossover'] else 0, indicators['candles_since_crossover'],
            indicators['volume_trend'], indicators['price_trend'], indicators['short_volume_trend'],
            indicators['support_level'], roc, action, confidence, explanation, rejecting_variable
        ))
        conn.commit()

        logging.info(f"{timestamp}: {symbol} - Decision: {action}, Confidence: {confidence}%, Rejecting: {rejecting_variable}")

    conn.close()

if __name__ == "__main__":
    # Ajustar el rango de tiempo para coincidir con el gráfico (March 3, 2025, alrededor de 16:30)
    start_time = datetime(2025, 3, 3, 15, 0, tzinfo=pytz.timezone("America/Bogota"))  # 15:00 para capturar datos previos
    end_time = datetime(2025, 3, 3, 17, 0, tzinfo=pytz.timezone("America/Bogota"))   # 17:00 para incluir 16:30
    analyze_historical_data('ERN/USDT', start_time, end_time)
    logging.info("Análisis histórico completado para ERN/USDT.")