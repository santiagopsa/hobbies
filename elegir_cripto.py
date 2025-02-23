import ccxt
import pandas as pd
import numpy as np
import time
from ta.momentum import RSIIndicator
from dotenv import load_dotenv
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cargar variables de entorno
load_dotenv()

# Configuración del exchange para Binance global
exchange = ccxt.binance({
    "rateLimit": 1200,
    "enableRateLimit": True,
    "apiKey": os.getenv("BINANCE_API_KEY_REAL"),
    "secret": os.getenv("BINANCE_SECRET_KEY_REAL"),
    'options': {'defaultType': 'spot'}
})

def fetch_ohlcv_safe(symbol, timeframe='1h', limit=24):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        if len(df) < 14:
            logging.warning(f"Datos insuficientes para {symbol}: solo {len(df)} velas")
            return None
        return df
    except Exception as e:
        logging.error(f"Error al obtener datos OHLCV para {symbol}: {e}")
        return None

def get_top_symbols_by_volume(base_currency="USDT", min_volume=10000):
    try:
        markets = exchange.load_markets()
        symbols = [
            symbol for symbol, market in markets.items()
            if market['quote'] == base_currency and market['type'] == 'spot' and market['active']
        ]
        logging.info(f"Total de símbolos con {base_currency}: {len(symbols)}")
        
        tickers = exchange.fetch_tickers(symbols)
        volumes = [
            {'symbol': symbol, 'volume': ticker['quoteVolume']}
            for symbol, ticker in tickers.items()
            if 'quoteVolume' in ticker and ticker['quoteVolume'] >= min_volume
        ]
        
        sorted_symbols = sorted(volumes, key=lambda x: x['volume'], reverse=True)
        logging.info(f"Símbolos con volumen > {min_volume}: {len(sorted_symbols)}")
        return [item['symbol'] for item in sorted_symbols]
    except Exception as e:
        logging.error(f"Error al obtener símbolos por volumen: {e}")
        return []

def choose_best_cryptos(base_currency="USDT", top_n=100):
    symbols = get_top_symbols_by_volume(base_currency, min_volume=10000)
    if not symbols:
        logging.error("No se encontraron símbolos válidos.")
        return []

    crypto_data = []
    max_symbols_to_analyze = min(600, len(symbols))
    processed_count = 0

    logging.info(f"Analizando hasta {max_symbols_to_analyze} de {len(symbols)} símbolos disponibles")
    for symbol in symbols[:max_symbols_to_analyze]:
        df = fetch_ohlcv_safe(symbol, timeframe='1h', limit=24)
        if df is None or df.empty:
            continue

        # Calcular cambio porcentual en las últimas 24 horas
        pct_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100

        # Calcular volatilidad y volumen promedio
        volatility = ((df['high'] - df['low']) / df['close']).mean() * 100
        avg_volume = (df['volume'] * df['close']).mean()

        # Calcular RSI (se usa para información; podrías incorporar un ajuste en el score si lo deseas)
        rsi = RSIIndicator(df['close'], window=14).rsi()
        rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        # Puntaje compuesto: mientras mayor sea el score, más atractiva se considera la criptomoneda
        score = avg_volume * volatility * (1 + pct_change / 100)

        crypto_data.append({
           'symbol': symbol,
           'volatility': volatility,
           'avg_volume': avg_volume,
           'rsi': rsi_value,
           'pct_change': pct_change,
           'score': score
        })

        processed_count += 1
        time.sleep(0.02)

    if not crypto_data:
        logging.error("No se encontraron criptos viables.")
        return []

    # Ordenar por score de mayor a menor y seleccionar las top_n
    df_data = pd.DataFrame(crypto_data)
    df_sorted = df_data.sort_values(by='score', ascending=False)
    selected_symbols = df_sorted['symbol'].head(top_n).tolist()

    logging.info(f"Símbolos analizados: {len(crypto_data)}, seleccionados: {len(selected_symbols)}")
    return selected_symbols


if __name__ == "__main__":
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=200)
    print(f"Top 200 criptos seleccionadas: {len(selected_cryptos)} símbolos")
    print(selected_cryptos)