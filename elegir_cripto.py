import ccxt
import pandas as pd

# Configuración del exchange para Binance US
exchange = ccxt.binanceus({
    "rateLimit": 1200,
    "enableRateLimit": True
})

def fetch_ohlcv_safe(symbol, timeframe='1d', limit=30):
    """
    Función segura para obtener datos OHLCV de un símbolo.
    Maneja excepciones y retorna None si hay un error.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        print(f"Error al obtener datos para {symbol}: {e}")
        return None

def get_top_symbols_by_volume(base_currency="USDT", limit=50):
    """
    Obtiene los pares de trading más relevantes basados en volumen.
    
    Args:
        base_currency (str): Moneda base para filtrar pares (e.g., 'USDT').
        limit (int): Número máximo de símbolos a devolver.
    
    Returns:
        list: Lista de símbolos ordenados por volumen.
    """
    try:
        markets = exchange.load_markets()
        # Filtrar solo mercados "spot"
        symbols = [
            symbol for symbol, market in markets.items()
            if market['quote'] == base_currency and market['type'] == 'spot' and market['active']
        ]
        # Obtener el volumen de las últimas 24 horas
        tickers = exchange.fetch_tickers(symbols)
        volumes = [
            {'symbol': symbol, 'volume': ticker['quoteVolume']}
            for symbol, ticker in tickers.items()
            if 'quoteVolume' in ticker
        ]
        # Ordenar por volumen y seleccionar los `limit` primeros
        sorted_symbols = sorted(volumes, key=lambda x: x['volume'], reverse=True)[:limit]
        return [item['symbol'] for item in sorted_symbols]
    except Exception as e:
        print(f"Error al obtener los símbolos principales por volumen: {e}")
        return []

def choose_best_cryptos(base_currency="USDT", top_n=10):
    """
    Elige las mejores criptos basándose en volumen y volatilidad.
    Retorna las `top_n` criptos con el mejor puntaje.
    
    Args:
        base_currency (str): Moneda base para filtrar pares (e.g., 'USDT').
        top_n (int): Número de criptos a seleccionar.
        
    Returns:
        list: Lista de símbolos seleccionados.
    """
    symbols = get_top_symbols_by_volume(base_currency)
    crypto_data = []

    # Analizar cada cripto
    for symbol in symbols:
        df = fetch_ohlcv_safe(symbol)
        if df is not None and not df.empty:
            # Calcular volatilidad (promedio del rango diario)
            volatility = (df['high'] - df['low']).mean()
            # Calcular volumen promedio
            avg_volume = df['volume'].mean()
            # Guardar datos
            crypto_data.append({
                'symbol': symbol,
                'volatility': volatility,
                'avg_volume': avg_volume
            })

    # Crear un DataFrame para calcular el puntaje
    if not crypto_data:
        print("No se encontraron datos válidos.")
        return []

    df = pd.DataFrame(crypto_data)
    # Calcular un puntaje basado en volumen y volatilidad
    df['score'] = df['avg_volume'] * df['volatility']
    # Ordenar y seleccionar las `top_n` criptos
    df = df.sort_values(by='score', ascending=False).head(top_n)
    
    return df['symbol'].tolist()

# Ejemplo de uso
selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=10)
print(selected_cryptos)
