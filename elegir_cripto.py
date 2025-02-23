import ccxt
import pandas as pd
import numpy as np
import time
from ta.momentum import RSIIndicator
from dotenv import load_dotenv
import os
import logging

# Configurar logging a nivel INFO para evitar demasiada salida
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
            logging.info(f"Datos insuficientes para {symbol}: solo {len(df)} velas")
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

def process_symbol(symbol):
    """
    Procesa un símbolo: obtiene datos OHLCV, calcula indicadores y retorna un diccionario con los resultados.
    """
    df = fetch_ohlcv_safe(symbol, timeframe='1h', limit=24)
    if df is None or df.empty:
        return None

    pct_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
    volatility = ((df['high'] - df['low']) / df['close']).mean() * 100
    avg_volume = (df['volume'] * df['close']).mean()
    rsi = RSIIndicator(df['close'], window=14).rsi()
    rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    score = avg_volume * volatility * (1 + pct_change / 100)
    
    return {
        'symbol': symbol,
        'volatility': volatility,
        'avg_volume': avg_volume,
        'rsi': rsi_value,
        'pct_change': pct_change,
        'score': score
    }

def choose_best_cryptos(base_currency="USDT", top_n=100):
    logging.info("Iniciando elección de las mejores criptomonedas.")
    symbols = get_top_symbols_by_volume(base_currency, min_volume=10000)
    if not symbols:
        logging.error("No se encontraron símbolos válidos.")
        return []

    crypto_data = []
    max_symbols_to_analyze = min(600, len(symbols))
    symbols_to_process = symbols[:max_symbols_to_analyze]
    total_symbols = len(symbols_to_process)
    logging.info(f"Analizando {total_symbols} de {len(symbols)} símbolos disponibles")

    processed_count = 0
    for symbol in symbols_to_process:
        processed_count += 1
        result = process_symbol(symbol)
        if result:
            crypto_data.append(result)
        # Imprimir avance aproximado (se actualiza en la misma línea)
        print(f"Revisando {processed_count} de {total_symbols} monedas...", end="\r")
    print("")  # Nueva línea después de la impresión de avance

    if not crypto_data:
        logging.error("No se encontraron criptos viables. Devolviendo lista vacía.")
        return []

    df_data = pd.DataFrame(crypto_data)
    df_sorted = df_data.sort_values(by='score', ascending=False)
    selected_symbols = df_sorted['symbol'].head(top_n).tolist()

    # Asegurar que cada símbolo tenga el formato "BASE/QUOTE"
    formatted_symbols = []
    for sym in selected_symbols:
        if "/" not in sym:
            if sym.endswith(base_currency):
                formatted = sym[:-len(base_currency)] + "/" + base_currency
                formatted_symbols.append(formatted)
            else:
                formatted_symbols.append(sym)
        else:
            formatted_symbols.append(sym)

    logging.info(f"Total de criptos analizadas: {len(crypto_data)}; Criptos seleccionadas: {len(formatted_symbols)}")
    logging.info(f"Lista final de criptos seleccionadas: {formatted_symbols}")
    return formatted_symbols

if __name__ == "__main__":
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=100)
    print(f"Top 100 criptos seleccionadas: {len(selected_cryptos)} símbolos")
    print(selected_cryptos)
