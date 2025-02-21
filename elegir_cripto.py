import ccxt
import pandas as pd
import numpy as np
import time
from ta.trend import ADXIndicator
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

def fetch_ohlcv_safe(symbol, timeframe='1h', limit=48):
    """
    Obtiene datos OHLCV de manera segura para un símbolo.
    
    Args:
        symbol (str): Símbolo del par (e.g., "BTC/USDT").
        timeframe (str): Marco temporal (default: '1h').
        limit (int): Número de velas (default: 48 horas).
    
    Returns:
        pd.DataFrame: Datos OHLCV o None si falla.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        if len(df) < limit:
            logging.warning(f"Datos incompletos para {symbol}: solo {len(df)} velas")
        return df
    except Exception as e:
        logging.error(f"Error al obtener datos OHLCV para {symbol}: {e}")
        return None

def get_top_symbols_by_volume(base_currency="USDT", min_volume=50000):
    """
    Obtiene pares de trading con volumen significativo.
    
    Args:
        base_currency (str): Moneda base (e.g., "USDT").
        min_volume (float): Volumen mínimo en USD (ajustado a 50,000).
    
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
    Selecciona las mejores criptomonedas para invertir basadas en volumen, volatilidad,
    tendencia y momentum.
    
    Args:
        base_currency (str): Moneda base (e.g., "USDT").
        top_n (int): Número máximo de símbolos a devolver (default: 200).
    
    Returns:
        list: Lista de símbolos seleccionados.
    """
    symbols = get_top_symbols_by_volume(base_currency, min_volume=50000)
    if not symbols:
        logging.error("No se encontraron símbolos válidos.")
        return []

    crypto_data = []
    max_symbols_to_analyze = min(600, len(symbols))

    logging.info(f"Analizando hasta {max_symbols_to_analyze} de {len(symbols)} símbolos disponibles")
    for symbol in symbols[:max_symbols_to_analyze]:
        df = fetch_ohlcv_safe(symbol, timeframe='1h', limit=48)
        if df is None or df.empty or len(df) < 48:
            continue

        volatility = ((df['high'] - df['low']) / df['close']).mean() * 100
        avg_volume = (df['volume'] * df['close']).mean()
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        adx_value = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        price_change_6h = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] * 100
        rsi = RSIIndicator(df['close'], window=14).rsi()
        rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        if (adx_value > 20 and volatility > 0.5 and rsi_value < 70 and price_change_6h > -2):
            score = (avg_volume * volatility * adx_value * (1 + price_change_6h / 100))
            crypto_data.append({
                'symbol': symbol,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'adx': adx_value,
                'price_change_6h': price_change_6h,
                'rsi': rsi_value,
                'score': score
            })
        else:
            logging.debug(f"{symbol} descartado - ADX: {adx_value}, Volatility: {volatility}, RSI: {rsi_value}, Price Change 6h: {price_change_6h}")
        
        time.sleep(0.05)

    if not crypto_data:
        logging.error("No se encontraron criptos que cumplan los criterios.")
        return []

    df = pd.DataFrame(crypto_data)
    df = df.sort_values(by='score', ascending=False)
    selected_symbols = df['symbol'].head(top_n).tolist()
    
    logging.info(f"Símbolos analizados: {len(crypto_data)}, seleccionados: {len(selected_symbols)}")
    return selected_symbols

if __name__ == "__main__":
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=200)
    print(f"Top 200 criptos seleccionadas: {len(selected_cryptos)} símbolos")
    print(selected_cryptos)