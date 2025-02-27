import ccxt
import pandas as pd

binance = ccxt.binance()
symbol = 'PNUT/USDT'
timeframe = '1d'
all_ohlcv = []

# Pide datos desde 2021, por ejemplo
since_timestamp = binance.parse8601('2021-01-01T00:00:00Z')
limit = 200

while True:
    batch = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since_timestamp, limit=limit)
    if not batch:
        break
    all_ohlcv += batch
    # El "timestamp" de la última vela
    last_ts = batch[-1][0]
    # Avanza el "since" para la siguiente página
    since_timestamp = last_ts + (60_000 if timeframe=='1m' else 3600_000 if timeframe=='1h' else 86400_000)
    # Rompe si devuelven menos velas de las solicitadas -> no hay más datos
    if len(batch) < limit:
        break

df_all = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
print("Total velas conseguidas:", len(df_all))
