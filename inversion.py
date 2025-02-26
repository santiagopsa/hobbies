import time
import datetime
import ccxt
import os
import requests
import numpy as np
import threading
import logging
from inversion_binance import demo_trading
from dotenv import load_dotenv
from elegir_cripto import choose_best_cryptos

load_dotenv()

# Configurar Binance
exchange = ccxt.binance({
    "enableRateLimit": True,
    "apiKey": os.getenv("BINANCE_API_KEY_REAL"),
    "secret": os.getenv("BINANCE_SECRET_KEY_REAL"),
})

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    filename="inversion_monitor.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constantes
MONITOR_INTERVAL = 60  # 1 minuto (real-time monitoring)
EXECUTION_INTERVAL = 3600  # 1 hora (faster execution)
THRESHOLD_VOLUME_CHANGE = 0.5  # 50% (more sensitive to spikes)
THRESHOLD_PRICE_CHANGE = 0.05  # 5% (more sensitive to moves)
THRESHOLD_ATR = 0.02  # 2% de volatilidad
THRESHOLD_RSI_OVERBOUGHT = 70  # Ajustado a 70
THRESHOLD_RSI_OVERSOLD = 30
SYMBOLS_TO_MONITOR = 100  # Mantener 50 símbolos para memoria baja

# Variables globales
last_conditions = {}
next_execution_time = time.time() + EXECUTION_INTERVAL

def fetch_klines(symbol, interval="1h", limit=10):  # Reducido a 10 para memoria baja
    binance_symbol = symbol.replace("/", "")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code != 200:
            logging.error(f"Error al obtener velas para {symbol}: {response.text}")
            return []
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            logging.warning(f"Respuesta inesperada para {symbol}: {data}")
            return []
        return data
    except Exception as e:
        logging.error(f"Error al procesar datos de {symbol}: {e}")
        return []

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
    rsi = 100 - (100 / (1 + rs))
    rsis = [rsi]
    for i in range(period, len(prices) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsis.append(100 - (100 / (1 + rs)))
    return rsis[-1]

def calculate_atr(symbol, period=14):
    try:
        klines = fetch_klines(symbol, interval="1h", limit=period + 1)
        if len(klines) < period + 1:
            return None
        true_ranges = []
        for i in range(1, len(klines)):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_close = float(klines[i-1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        atr = np.mean(true_ranges)
        return atr / float(klines[-1][4])
    except Exception as e:
        logging.error(f"Error calculating ATR for {symbol}: {e}")
        return None

def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error al obtener precio para {symbol}: {e}")
        return None

def fetch_volume(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['quoteVolume']
    except Exception as e:
        logging.error(f"Error al obtener volumen para {symbol}: {e}")
        return None

def fetch_portfolio_symbols():
    try:
        balance = exchange.fetch_balance()
        return [f"{asset}/USDT" for asset, total in balance['total'].items() if total > 0 and asset != 'USDT']
    except Exception as e:
        logging.error(f"Error al obtener símbolos del portafolio: {e}")
        return []

def should_execute_trading(symbol):
    global last_conditions
    
    if symbol not in last_conditions:
        last_conditions[symbol] = {"price": None, "volume": None, "rsi": None}

    current_price = fetch_price(symbol)
    current_volume = fetch_volume(symbol)
    klines = fetch_klines(symbol)
    if not current_price or not current_volume or not klines:
        return False

    closing_prices = [float(k[4]) for k in klines]
    rsi = calculate_rsi(closing_prices)
    atr = calculate_atr(symbol)
    if rsi is None or atr is None:
        return False

    execute = False
    last = last_conditions[symbol]

    if rsi > THRESHOLD_RSI_OVERBOUGHT or rsi < THRESHOLD_RSI_OVERSOLD:
        logging.info(f"RSI extremo para {symbol}: {rsi}")
        execute = True
    if last["price"] and abs((current_price - last["price"]) / last["price"]) > THRESHOLD_PRICE_CHANGE:
        logging.info(f"Cambio significativo en precio para {symbol}: {current_price} vs {last['price']}")
        execute = True
    if last["volume"] and abs((current_volume - last["volume"]) / last["volume"]) > THRESHOLD_VOLUME_CHANGE:
        logging.info(f"Cambio significativo en volumen para {symbol}: {current_volume} vs {last['volume']}")
        execute = True
    if atr > THRESHOLD_ATR:
        logging.info(f"Alta volatilidad para {symbol}: ATR {atr*100:.2f}%")
        execute = True

    last_conditions[symbol] = {"price": current_price, "volume": current_volume, "rsi": rsi}
    return execute

def run_trading(high_volume_symbols=None):
    logging.info(f"Ejecutando demo_trading en segundo plano a las {datetime.datetime.now()}")
    try:
        # Run demo_trading in a separate thread to avoid blocking
        thread = threading.Thread(target=demo_trading, args=(high_volume_symbols,))
        thread.daemon = True  # Ensure the thread doesn’t prevent the main program from exiting
        thread.start()
        logging.info("Trading iniciado en segundo plano.")
        return True
    except Exception as e:
        logging.error(f"Error al iniciar demo_trading: {e}")
        return False

def monitor_realtime(symbols):
    global next_execution_time
    
    logging.info("Iniciando monitoreo en tiempo real para los 50 mejores activos USDT...")
    run_trading(symbols)  # Initial execution

    while True:
        current_time = time.time()
        portfolio_symbols = fetch_portfolio_symbols()
        
        # Llamada a choose_best_cryptos (puede ser bloqueante, pero reducida a 50 símbolos)
        logging.info("Esperando a que se complete la selección de criptomonedas (choose_best_cryptos)...")
        high_volume_symbols = choose_best_cryptos(base_currency="USDT", top_n=SYMBOLS_TO_MONITOR)
        logging.info(f"Símbolos de alto volumen pedidos: {SYMBOLS_TO_MONITOR}, obtenidos: {len(high_volume_symbols)}")

        symbols = list(set(high_volume_symbols + portfolio_symbols))
        logging.info(f"Monitoreando {len(symbols)} símbolos a las {datetime.datetime.now()}")
        execute_now = False
        
        for symbol in symbols:
            if should_execute_trading(symbol):
                execute_now = True
                break
        
        if execute_now or current_time >= next_execution_time:
            run_trading(symbols)
            next_execution_time = current_time + EXECUTION_INTERVAL
            logging.info(f"Próxima ejecución programada a las {datetime.datetime.fromtimestamp(next_execution_time)}")
        else:
            logging.info(f"No se cumplen condiciones. Próxima revisión en {MONITOR_INTERVAL//60} minutos.")

        time.sleep(MONITOR_INTERVAL)

if __name__ == "__main__":
    logging.info("Iniciando el proceso principal...")
    logging.info("Optimizando para revisar cada minuto y capturar oportunidades de trading en todos los activos USDT relevantes.")
    try:
        high_volume_symbols = choose_best_cryptos(base_currency="USDT", top_n=SYMBOLS_TO_MONITOR)
        monitor_realtime(high_volume_symbols)
    except KeyboardInterrupt:
        logging.info("\nPrograma detenido por el usuario.")