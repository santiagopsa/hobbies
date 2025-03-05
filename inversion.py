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
MONITOR_INTERVAL = 60  # 1 minuto por lote
EXECUTION_INTERVAL = 3600  # 1 hora
SYMBOL_UPDATE_INTERVAL = 600  # 10 minutos para actualizar símbolos
THRESHOLD_VOLUME_CHANGE = 0.5  # 50%
THRESHOLD_PRICE_CHANGE = 0.05  # 5%
THRESHOLD_RSI_OVERBOUGHT = 70
THRESHOLD_RSI_OVERSOLD = 30
SYMBOLS_TO_MONITOR = 200  # Total de símbolos a monitorear
SYMBOLS_PER_BATCH = 40  # Símbolos por iteración (200 / 5 minutos = 40 por minuto)

# Variables globales
last_conditions = {}
data_cache = {}  # Caché para precio, volumen y RSI
symbol_cache = {"symbols": [], "last_update": 0}
next_execution_time = time.time() + EXECUTION_INTERVAL
trading_running = False
lock = threading.Lock()
batch_index = 0  # Índice para rotar lotes

def fetch_klines(symbol, interval="1h", limit=10):
    binance_symbol = symbol.replace("/", "")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=3)
        return response.json() if response.status_code == 200 and isinstance(response.json(), list) else []
    except Exception:
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
    return 100 - (100 / (1 + rs))  # Versión simplificada

def fetch_symbol_data(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return {
            "price": ticker['last'],
            "volume": ticker['quoteVolume'],
            "klines": None  # Solo para RSI cuando sea necesario
        }
    except Exception as e:
        logging.error(f"Error al obtener datos para {symbol}: {e}")
        return None

def update_symbol_cache():
    global symbol_cache
    current_time = time.time()
    with lock:
        if current_time - symbol_cache["last_update"] >= SYMBOL_UPDATE_INTERVAL:
            try:
                logging.info("Actualizando lista de símbolos...")
                symbol_cache["symbols"] = choose_best_cryptos(base_currency="USDT", top_n=SYMBOLS_TO_MONITOR)
                symbol_cache["last_update"] = current_time
                logging.info(f"Lista actualizada: {len(symbol_cache['symbols'])} símbolos")
            except Exception as e:
                logging.error(f"Error al actualizar símbolos: {e}")

def fetch_portfolio_symbols():
    try:
        balance = exchange.fetch_balance()
        return [f"{asset}/USDT" for asset, total in balance['total'].items() if total > 0 and asset != 'USDT']
    except Exception:
        return []

def should_execute_trading(symbol):
    global last_conditions, data_cache

    with lock:
        current_time = time.time()
        if symbol not in data_cache or current_time - data_cache[symbol].get("timestamp", 0) > 300:  # Actualizar cada 5 minutos
            data = fetch_symbol_data(symbol)
            if data:
                data_cache[symbol] = {"data": data, "timestamp": current_time}
            else:
                return False
        else:
            data = data_cache[symbol]["data"]

    current_price = data["price"]
    current_volume = data["volume"]

    if symbol not in last_conditions:
        last_conditions[symbol] = {"price": None, "volume": None, "rsi": None, "last_rsi_update": 0}

    last = last_conditions[symbol]
    execute = False

    # RSI cada 5 minutos
    if current_time - last["last_rsi_update"] >= 300:
        klines = fetch_klines(symbol)
        if klines and len(klines) >= 10:
            closing_prices = [float(k[4]) for k in klines]
            rsi = calculate_rsi(closing_prices)
            if rsi is not None:
                if rsi > THRESHOLD_RSI_OVERBOUGHT or rsi < THRESHOLD_RSI_OVERSOLD:
                    logging.info(f"RSI extremo para {symbol}: {rsi}")
                    execute = True
                last["rsi"] = rsi
            last["last_rsi_update"] = current_time
    elif last["rsi"] and (last["rsi"] > THRESHOLD_RSI_OVERBOUGHT or last["rsi"] < THRESHOLD_RSI_OVERSOLD):
        execute = True

    # Cambios rápidos en precio y volumen
    if last["price"] and abs((current_price - last["price"]) / last["price"]) > THRESHOLD_PRICE_CHANGE:
        logging.info(f"Cambio significativo en precio para {symbol}: {current_price} vs {last['price']}")
        execute = True
    if last["volume"] and abs((current_volume - last["volume"]) / last["volume"]) > THRESHOLD_VOLUME_CHANGE:
        logging.info(f"Cambio significativo en volumen para {symbol}: {current_volume} vs {last['volume']}")
        execute = True

    with lock:
        last_conditions[symbol].update({"price": current_price, "volume": current_volume})
    return execute

def run_trading(high_volume_symbols=None):
    global trading_running
    if trading_running:
        logging.info("Trading ya en ejecución, omitiendo.")
        return False
    logging.info(f"Ejecutando demo_trading a las {datetime.datetime.now()}")
    trading_running = True
    def trading_thread():
        try:
            demo_trading(high_volume_symbols)
        except Exception as e:
            logging.error(f"Error en demo_trading: {e}")
        finally:
            global trading_running
            trading_running = False
    thread = threading.Thread(target=trading_thread)
    thread.daemon = True
    thread.start()
    logging.info("Trading iniciado en segundo plano.")
    return True

def monitor_realtime():
    global next_execution_time, batch_index

    logging.info(f"Iniciando monitoreo continuo para {SYMBOLS_TO_MONITOR} activos USDT...")
    update_symbol_cache()
    run_trading(symbol_cache["symbols"])  # Ejecución inicial

    while True:
        try:
            current_time = time.time()
            update_symbol_cache()
            portfolio_symbols = fetch_portfolio_symbols()
            all_symbols = list(set(symbol_cache["symbols"] + portfolio_symbols))

            # Rotar lotes de SYMBOLS_PER_BATCH (40) cada minuto
            start_idx = batch_index * SYMBOLS_PER_BATCH
            end_idx = min(start_idx + SYMBOLS_PER_BATCH, len(all_symbols))
            symbols = all_symbols[start_idx:end_idx]
            batch_index = (batch_index + 1) % ((len(all_symbols) + SYMBOLS_PER_BATCH - 1) // SYMBOLS_PER_BATCH)  # Ciclo completo

            logging.info(f"Monitoreando lote de {len(symbols)} símbolos (índice {start_idx}-{end_idx-1}) a las {datetime.datetime.now()}")

            execute_now = False
            for symbol in symbols:
                if should_execute_trading(symbol):
                    execute_now = True
                    break

            if execute_now or current_time >= next_execution_time:
                run_trading(all_symbols)  # Ejecutar con todos los símbolos si hay una condición
                next_execution_time = current_time + EXECUTION_INTERVAL
                logging.info(f"Próxima ejecución programada a las {datetime.datetime.fromtimestamp(next_execution_time)}")
            else:
                logging.info(f"No se cumplen condiciones en este lote. Próxima revisión en {MONITOR_INTERVAL//60} minutos.")

            time.sleep(MONITOR_INTERVAL)
        except Exception as e:
            logging.error(f"Error en monitor_realtime: {e}. Reintentando en 10 segundos...")
            time.sleep(10)

if __name__ == "__main__":
    logging.info("Iniciando el proceso principal para ejecución continua...")
    try:
        monitor_realtime()
    except KeyboardInterrupt:
        logging.info("\nPrograma detenido por el usuario.")
    except Exception as e:
        logging.error(f"Error crítico al iniciar: {e}. Reiniciando en 30 segundos...")
        time.sleep(30)
        monitor_realtime()  # Reinicio automático