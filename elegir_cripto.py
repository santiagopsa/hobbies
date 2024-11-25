import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Configurar el exchange (ejemplo: Binance)
exchange = ccxt.binance()

# Función para obtener la lista de criptos disponibles
def get_available_cryptos():
    markets = exchange.load_markets()
    cryptos = [
        market for market in markets.values()
        if market['active'] and '/USDT' in market['symbol']  # Solo pares con USDT
    ]
    return cryptos

# Función para filtrar criptos por tiempo de vida y volumen
def filter_cryptos(cryptos, min_days=180, min_volume=1_000_000):
    filtered_cryptos = []
    for crypto in cryptos:
        symbol = crypto['symbol']
        since = crypto['info'].get('listed_since', None)
        
        # Verificar tiempo de vida
        if since:
            days_active = (datetime.now() - datetime.fromtimestamp(int(since) / 1000)).days
            if days_active < min_days:
                continue  # Saltar criptos nuevas
        
        # Verificar volumen promedio
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=30)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            avg_volume = df['volume'].mean()
            if avg_volume < min_volume:
                continue  # Saltar criptos con bajo volumen
        except Exception as e:
            print(f"Error al obtener datos de {symbol}: {e}")
            continue
        
        filtered_cryptos.append(symbol)
    return filtered_cryptos

# Ejecutar filtro
cryptos = get_available_cryptos()
filtered_cryptos = filter_cryptos(cryptos)
print(f"Criptos seleccionadas: {filtered_cryptos}")
