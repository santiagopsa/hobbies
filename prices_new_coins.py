import ccxt
import datetime
import logging
import time

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Inicializar el exchange con CCXT
exchange = ccxt.binance({
    'enableRateLimit': True,
})

symbol = 'LAYER/USDT'

def fetch_trades_first_10_seconds(symbol, start_time_iso):
    """
    Obtiene los trades del símbolo desde start_time (en ISO8601, por ejemplo "2025-02-11T14:00:00Z")
    y filtra los trades que ocurrieron en los primeros 10 segundos.
    
    Parámetros:
      - symbol: el par a analizar (ej. "LAYER/USDT")
      - start_time_iso: el momento de inicio en formato ISO8601 (UTC)
      
    Retorna una lista de trades (cada trade es un diccionario) que tienen un timestamp
    menor o igual a start_time + 10 segundos.
    """
    try:
        # Convertir la hora de inicio a timestamp en milisegundos
        since = exchange.parse8601(start_time_iso)
        # Llamar a fetch_trades con 'since'
        trades = exchange.fetch_trades(symbol, since=since)
        
        # Calcular el timestamp de 10 segundos después del start
        end_time = since + 10 * 1000  # 10 segundos en milisegundos
        
        # Filtrar los trades para quedarnos con los que ocurrieron en ese intervalo
        filtered_trades = [trade for trade in trades if trade['timestamp'] <= end_time]
        return filtered_trades
    except Exception as e:
        logging.error(f"Error fetching trades for {symbol}: {e}")
        return []

def print_trades(trades):
    """
    Imprime información de cada trade.
    """
    if not trades:
        logging.info("No se encontraron trades en el intervalo especificado.")
        return

    for trade in trades:
        # Convertir el timestamp (en ms) a formato datetime UTC
        dt = datetime.datetime.utcfromtimestamp(trade['timestamp'] / 1000)
        logging.info(f"Trade at {dt.isoformat()} - Price: {trade['price']} - Amount: {trade['amount']}")

if __name__ == "__main__":
    # Define el instante exacto en que quieres comenzar a analizar (por ejemplo, el listado de la moneda)
    # Debe estar en formato ISO8601, por ejemplo: "2025-02-11T14:00:00Z"
    start_time = "2025-02-11T14:00:00Z"
    logging.info(f"Obteniendo trades para {symbol} en los primeros 10 segundos a partir de {start_time}...")
    
    # Obtén los trades que ocurrieron en el intervalo [start_time, start_time + 10s]
    trades = fetch_trades_first_10_seconds(symbol, start_time)
    print_trades(trades)
