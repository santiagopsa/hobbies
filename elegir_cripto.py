import ccxt
import pandas as pd
import os
import logging
from dotenv import load_dotenv
import time

# Configurar logging a nivel INFO
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

markets_cache = None
markets_cache_timestamp = None
CACHE_DURATION = 3600  # 1 hora en segundos

def get_markets():
    global markets_cache, markets_cache_timestamp
    current_time = time.time()
    if markets_cache is None or (markets_cache_timestamp and current_time - markets_cache_timestamp > CACHE_DURATION):
        logging.info("Actualizando caché de mercados...")
        markets_cache = exchange.load_markets()
        markets_cache_timestamp = current_time
    return markets_cache

def choose_best_cryptos(base_currency="USDT", top_n=100, min_volume=500000):
    start_time = time.time()
    logging.info("Iniciando elección de criptomonedas (versión optimizada).")
    
    # Cargar todos los mercados y filtrar los símbolos que cumplen los criterios
    markets = get_markets()
    valid_symbols = [
        symbol for symbol, market in markets.items()
        if market['quote'] == base_currency and market['type'] == 'spot' and market['active']
    ]
    logging.info(f"Total de símbolos con {base_currency}: {len(valid_symbols)}")
    
    # Obtener tickers de todos los mercados (en una sola llamada)
    tickers = exchange.fetch_tickers()
    
    crypto_data = []
    for symbol in valid_symbols:
        ticker = tickers.get(symbol)
        if not ticker:
            continue
        
        # Filtrar por volumen mínimo
        try:
            quote_volume = float(ticker.get('quoteVolume', 0))
        except ValueError as e:
            logging.error(f"Volumen no convertible para {symbol}: {ticker.get('quoteVolume')}, error: {e}")
            continue
        
        if quote_volume < min_volume:
            continue
        
        try:
            high = float(ticker.get('high', 0))
            low = float(ticker.get('low', 0))
            last = float(ticker.get('last', 1))  # evitar división por cero
            # Calcular volatilidad aproximada
            volatility = ((high - low) / last) * 100 if last != 0 else 0
            # Obtener el cambio porcentual (se adapta al campo disponible)
            pct_change = float(ticker.get('percentage', ticker.get('priceChangePercent', 0)))
            # Calcular el score (filtro grueso)
            score = quote_volume * volatility * (1 + pct_change / 100)
            
            crypto_data.append({
                'symbol': symbol,
                'volatility': volatility,
                'quote_volume': quote_volume,
                'pct_change': pct_change,
                'score': score
            })
        except Exception as e:
            logging.error(f"Error procesando {symbol}: {e}")
    
    if not crypto_data:
        logging.error("No se encontraron criptos viables. Devolviendo lista vacía.")
        return []
    
    # Ordenar y seleccionar los mejores símbolos
    df_data = pd.DataFrame(crypto_data)
    df_sorted = df_data.sort_values(by='score', ascending=False)
    selected_symbols = df_sorted['symbol'].head(top_n).tolist()
    
    # Formatear símbolos para asegurar el formato "BASE/QUOTE"
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
    logging.info(f"Tiempo de ejecución: {time.time() - start_time:.2f}s")
    return formatted_symbols

if __name__ == "__main__":
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=100)
    print(f"Top 100 criptos seleccionadas: {len(selected_cryptos)} símbolos")