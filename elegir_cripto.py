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
    """
    Obtiene datos OHLCV de manera segura para un símbolo.
    
    Args:
        symbol (str): Símbolo del par (e.g., "BTC/USDT").
        timeframe (str): Marco temporal (default: '1h').
        limit (int): Número de velas (default: 24 horas).
    
    Returns:
        pd.DataFrame: Datos OHLCV o None si falla.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        if len(df) < 14:  # Mínimo para RSI
            logging.warning(f"Datos insuficientes para {symbol}: solo {len(df)} velas")
            return None
        return df
    except Exception as e:
        logging.error(f"Error al obtener datos OHLCV para {symbol}: {e}")
        return None

def get_top_symbols_by_volume(base_currency="USDT", min_volume=10000):
    """
    Obtiene pares de trading con volumen significativo.
    
    Args:
        base_currency (str): Moneda base (e.g., "USDT").
        min_volume (float): Volumen mínimo en USD (ajustado a 10,000).
    
    Returns:
        list: Lista de símbolos filtrados.
    """
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

def choose_best_cryptos(base_currency="USDT", top_n=200):
    """
    Selecciona las mejores criptomonedas para invertir basadas en volumen y viabilidad básica.
    
    Args:
        base_currency (str): Moneda base (e.g., "USDT").
        top_n (int): Número máximo de símbolos a devolver (default: 200).
    
    Returns:
        list: Lista de símbolos seleccionados.
    """
    symbols = get_top_symbols_by_volume(base_currency, min_volume=10000)
    if not symbols:
        logging.error("No se encontraron símbolos válidos.")
        return []

    crypto_data = []
    max_symbols_to_analyze = min(600, len(symbols))
    processed_count = 0
    discarded_by_data = 0
    discarded_by_rsi = 0

    logging.info(f"Analizando hasta {max_symbols_to_analyze} de {len(symbols)} símbolos disponibles")
    for symbol in symbols[:max_symbols_to_analyze]:
        df = fetch_ohlcv_safe(symbol, timeframe='1h', limit=24)
        if df is None or df.empty:
            discarded_by_data += 1
            continue

        volatility = ((df['high'] - df['low']) / df['close']).mean() * 100
        avg_volume = (df['volume'] * df['close']).mean()
        rsi = RSIIndicator(df['close'], window=14).rsi()
        rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        if rsi_value >= 80:  # Solo descartar sobrecompra extrema
            discarded_by_rsi += 1
            logging.debug(f"{symbol} descartado por RSI: {rsi_value}")
            continue

        # Puntaje simple basado en volumen y volatilidad
        score = avg_volume * volatility
        crypto_data.append({
            'symbol': symbol,
            'volatility': volatility,
            'avg_volume': avg_volume,
            'rsi': rsi_value,
            'score': score
        })

        processed_count += 1
        if processed_count % 50 == 0:
            logging.info(f"Progreso: {processed_count} procesados, {len(crypto_data)} válidos, "
                         f"Descartados - Datos: {discarded_by_data}, RSI: {discarded_by_rsi}")

        time.sleep(0.02)

    if not crypto_data:
        logging.error("No se encontraron criptos viables. Devolviendo top por volumen.")
        return symbols[:top_n]

    df = pd.DataFrame(crypto_data)
    df = df.sort_values(by='score', ascending=False)
    selected_symbols = df['symbol'].head(top_n).tolist()
    
    logging.info(f"Símbolos analizados: {len(crypto_data)}, seleccionados: {len(selected_symbols)}")
    logging.info(f"Descartados totales - Datos: {discarded_by_data}, RSI: {discarded_by_rsi}")
    return selected_symbols

if __name__ == "__main__":
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=200)
    print(f"Top 200 criptos seleccionadas: {len(selected_cryptos)} símbolos")
    print(selected_cryptos)