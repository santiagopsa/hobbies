import ccxt
import websocket
import json
import threading
import logging
import time
import os
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
import queue
from datetime import timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    filename='new_coins_ws.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Cola para logging asíncrono
log_queue = queue.Queue()

def log_worker():
    while True:
        level, msg = log_queue.get()
        logging.log(level, msg)
        log_queue.task_done()

threading.Thread(target=log_worker, daemon=True).start()

# Cargar variables de entorno desde .env
load_dotenv()

# Configuración de la API de Binance para spot (por defecto)
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY_REAL'),
    'secret': os.getenv('BINANCE_SECRET_KEY_REAL'),
    'enableRateLimit': True,
    'timeout': 20000,  # en milisegundos
    'options': {
        'adjustForTimeDifference': True
    }
})

# Configuración de Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log_queue.put((logging.WARNING, "Telegram token o chat ID no configurado."))
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        log_queue.put((logging.ERROR, f"Error al enviar mensaje a Telegram: {e}"))

# Base de datos
DB_NAME = "trading_bot.db"
MAX_DAILY_PURCHASES = 1

def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions_new_coins (
            symbol TEXT,
            action TEXT,
            price REAL,
            amount REAL,
            timestamp TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_purchases (
            date TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def insert_transaction(symbol, action, price, amount, timestamp):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions_new_coins (symbol, action, price, amount, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, action, price, amount, timestamp))
        conn.commit()
        conn.close()
    except Exception as e:
        log_queue.put((logging.ERROR, f"Error al insertar transacción: {e}"))

def get_daily_purchases():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT count FROM daily_purchases WHERE date = ?", (today,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0

def increment_daily_purchases():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute(
        "INSERT INTO daily_purchases (date, count) VALUES (?, 1) "
        "ON CONFLICT(date) DO UPDATE SET count = count + 1", (today,)
    )
    conn.commit()
    conn.close()

# Global variables
market_cache = {}
active_orders = set()
MIN_NOTIONAL = 10  # Minimum notional set to $7 USD for new coins

# Función para obtener precio
def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except ccxt.RateLimitExceeded:
        log_queue.put((logging.WARNING, f"Límite de API excedido al obtener precio para {symbol}"))
        time.sleep(0.1)
        return None
    except Exception as e:
        log_queue.put((logging.ERROR, f"Error al obtener precio para {symbol}: {e}"))
        return None

# Cache markets
def update_market_cache():
    global market_cache
    while True:
        try:
            market_cache = exchange.load_markets()
            log_queue.put((logging.DEBUG, "Mercados actualizados en caché."))
            time.sleep(60)  # Update every minute
        except Exception as e:
            log_queue.put((logging.ERROR, f"Error al actualizar caché de mercados: {e}"))
            time.sleep(10)

threading.Thread(target=update_market_cache, daemon=True).start()

# Compra optimizada con reintentos
buying_lock = threading.Lock()

def buy_symbol_microsecond(symbol, budget=15, max_attempts=5, retry_delay=0.001):
    daily_purchases = get_daily_purchases()
    if daily_purchases >= MAX_DAILY_PURCHASES:
        log_queue.put((logging.INFO, f"Límite diario alcanzado ({MAX_DAILY_PURCHASES}). Ignorando {symbol}."))
        send_telegram_message(f"⚠️ Límite diario alcanzado. No se comprará `{symbol}`.")
        return None

    with buying_lock:
        if symbol in active_orders:
            return None
        active_orders.add(symbol)

    log_queue.put((logging.INFO, f"🚀 Comprando {symbol} inmediatamente..."))
    try:
        # Obtener precio rápidamente
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        if not price:
            raise Exception("No se pudo obtener precio")

        # Calcular cantidad mínima viable
        amount = budget / price
        market = market_cache.get(symbol, exchange.markets[symbol])
        lot_size = market['limits']['amount']['min']
        amount = max(amount, lot_size)
        amount = round(amount / lot_size) * lot_size

        # Ejecutar compra sin más verificaciones
        start_time = time.time()
        order = exchange.create_market_buy_order(symbol, amount)
        end_time = time.time()

        order_price = order.get('average', price)
        filled = order.get('filled', amount)
        timestamp = datetime.now(timezone.utc).isoformat()

        insert_transaction(symbol, 'buy', order_price, filled, timestamp)
        increment_daily_purchases()
        latency_ms = (end_time - start_time) * 1000
        log_queue.put((logging.INFO, f"✅ Compra ejecutada: {symbol} a {order_price} USDT | Cantidad: {filled} | Latencia: {latency_ms:.3f}ms"))
        send_telegram_message(f"✅ *Compra ejecutada*\nSímbolo: `{symbol}`\nPrecio: `{order_price}`\nCantidad: `{filled}`\nLatencia: `{latency_ms:.3f}ms`")

        active_orders.remove(symbol)
        return {'price': order_price, 'filled': filled}

    except Exception as e:
        log_queue.put((logging.ERROR, f"Error al comprar {symbol}: {e}"))
        send_telegram_message(f"❌ *Error*: `{symbol}` no se pudo comprar: {e}")
        active_orders.remove(symbol)
        return None

# Venta
def sell_symbol(symbol, amount):
    try:
        if symbol not in market_cache:
            exchange.load_markets(reload=True)
        market = market_cache.get(symbol, exchange.markets[symbol])
        base_asset = symbol.split('/')[0]
        balance = exchange.fetch_balance()
        available = balance.get(base_asset, {}).get('free', 0)
        if available < amount:
            amount = available
        current_price = fetch_price(symbol)
        if not current_price:
            return None
        notional = amount * current_price
        min_notional = market.get('limits', {}).get('cost', {}).get('min', MIN_NOTIONAL)  # $7 minimum
        if notional < min_notional:
            amount = min_notional / current_price  # Adjust amount to meet minimum
            amount = max(amount, market['limits']['amount']['min'])  # Ensure lot size
            amount = round(amount / market['limits']['amount']['stepSize']) * market['limits']['amount']['stepSize']
        safe_amount = amount * 0.999  # Safety margin
        order = exchange.create_market_sell_order(symbol, safe_amount)
        order_price = order.get('average', order.get('price', current_price))
        ts = datetime.now(timezone.utc).isoformat()
        insert_transaction(symbol, 'sell', order_price, safe_amount, ts)
        log_queue.put((logging.INFO, f"Venta ejecutada: {symbol} a {order_price} USDT, cantidad: {safe_amount}"))
        send_telegram_message(f"✅ *Venta ejecutada*\nSímbolo: `{symbol}`\nPrecio: `{order_price}`\nCantidad: `{safe_amount}`")
        return order
    except ccxt.RateLimitExceeded:
        log_queue.put((logging.ERROR, f"Límite de API excedido al vender {symbol}"))
        time.sleep(1)
        return None
    except Exception as e:
        log_queue.put((logging.ERROR, f"Error al vender {symbol}: {e}"))
        return None

# Trailing Stop
def set_trailing_stop(symbol, amount, purchase_price, trailing_percent=5, max_duration=24*3600):
    start_time = time.time()
    highest_price = purchase_price
    log_queue.put((logging.INFO, f"🔄 Trailing stop iniciado para {symbol}"))

    while time.time() - start_time < max_duration:
        try:
            current_price = fetch_price(symbol)
            if not current_price:
                time.sleep(0.1)
                continue

            if current_price > highest_price:
                highest_price = current_price
                log_queue.put((logging.INFO, f"📈 Nuevo máximo: {symbol} a {highest_price}"))

            stop_price = highest_price * (1 - trailing_percent / 100)
            if current_price <= stop_price:
                log_queue.put((logging.INFO, f"🔴 Trailing stop activado: {symbol} vendido a {current_price}"))
                sell_symbol(symbol, amount)
                break
            time.sleep(0.1)
        except ccxt.RateLimitExceeded:
            log_queue.put((logging.ERROR, f"Límite de API excedido en trailing stop para {symbol}"))
            time.sleep(0.1)
        except Exception as e:
            log_queue.put((logging.ERROR, f"Error en trailing stop para {symbol}: {e}"))
            time.sleep(0.1)
    else:
        log_queue.put((logging.INFO, f"⏰ Trailing stop expiró para {symbol}. Vendiendo..."))
        sell_symbol(symbol, amount)

def process_order(symbol, order_details):
    purchase_price = order_details.get('price')
    amount = order_details.get('filled')
    if not purchase_price or not amount or float(amount) <= 0:
        log_queue.put((logging.ERROR, f"Datos insuficientes para trailing stop en {symbol}"))
        return
    threading.Thread(target=set_trailing_stop, args=(symbol, amount, purchase_price), daemon=True).start()

# WebSocket para spot
known_symbols = set()
new_coin_detected = False

def on_message(ws, message):
    global known_symbols, new_coin_detected
    try:
        tickers = json.loads(message)
        for ticker in tickers:
            symbol_raw = ticker.get("s")
            if not symbol_raw or not symbol_raw.endswith("USDT"):
                continue

            formatted_symbol = f"{symbol_raw[:-4]}/{symbol_raw[-4:]}"
            if formatted_symbol not in known_symbols:
                known_symbols.add(formatted_symbol)
                new_coin_detected = True

                event_time_ms = ticker.get("E")
                detection_time = datetime.now(timezone.utc)
                latency = (detection_time.timestamp() - (event_time_ms / 1000)) if event_time_ms else None
                latency_str = f"{latency:.6f}" if latency is not None else "N/A"
                log_queue.put((logging.INFO, f"🚀 Nueva moneda detectada: {formatted_symbol} | Latencia: {latency_str}s"))
                send_telegram_message(f"🚀 *Nueva moneda detectada*: `{formatted_symbol}`")

                # Launch buy in a separate thread for immediate execution
                threading.Thread(target=buy_symbol_microsecond, args=(formatted_symbol,), daemon=True).start()
    except Exception as e:
        log_queue.put((logging.ERROR, f"Error en on_message: {e}"))

def on_error(ws, error):
    log_queue.put((logging.ERROR, f"WebSocket error: {error}"))

def on_close(ws, close_status_code, close_msg):
    log_queue.put((logging.WARNING, f"WebSocket cerrado: {close_status_code} - {close_msg}. Reconectando..."))
    ws.keep_running = False
    time.sleep(2)

def on_open(ws):
    log_queue.put((logging.INFO, "Conexión WebSocket abierta"))

def start_websocket():
    url = "wss://stream.binance.com:9443/ws/!ticker@arr"
    while True:
        ws = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        try:
            ws.run_forever()
        except Exception as e:
            log_queue.put((logging.ERROR, f"Excepción en WebSocket: {e}. Reintentando en 5 segundos..."))
            time.sleep(5)
        finally:
            ws.close()
            del ws
            time.sleep(5)

def start_ws_thread():
    threading.Thread(target=start_websocket, daemon=True).start()

def get_sleep_duration():
    now_dt = datetime.now()
    next_critical = now_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    sleep_time = (next_critical - now_dt).total_seconds() - 5  # Wake up 5 seconds early
    return max(0, sleep_time)

def keep_markets_updated():
    global new_coin_detected
    while True:
        now_dt = datetime.now()
        minute = now_dt.minute

        if minute >= 58 or minute < 2:
            try:
                exchange.load_markets(reload=True)
                log_queue.put((logging.DEBUG, "Mercados actualizados en rango crítico."))
                
                if new_coin_detected:
                    log_queue.put((logging.DEBUG, "Nueva moneda detectada, durmiendo para permitir la compra."))
                    time.sleep(0.1)  # Reduced sleep
                    new_coin_detected = False
                else:
                    time.sleep(0.05)  # Faster updates during critical window
            except ccxt.RateLimitExceeded:
                log_queue.put((logging.WARNING, "Límite de API excedido en actualización de mercados. Esperando..."))
                time.sleep(0.1)
            except Exception as e:
                log_queue.put((logging.ERROR, f"Error al actualizar mercados: {e}"))
                time.sleep(0.1)
        else:
            next_critical = now_dt.replace(minute=58, second=0, microsecond=0)
            if next_critical <= now_dt:
                next_critical += timedelta(hours=1)
            sleep_time = (next_critical - now_dt).total_seconds()
            log_queue.put((logging.DEBUG, f"Fuera del rango crítico. Durmiendo por {sleep_time:.1f} segundos hasta el inicio del rango crítico."))
            time.sleep(sleep_time)

# Función para obtener símbolos iniciales usando el endpoint de spot
def fetch_current_symbols_fast():
    try:
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=1)
        data = response.json()
        symbols = {f"{s['baseAsset']}/{s['quoteAsset']}" for s in data.get("symbols", [])
                   if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT"}
        return symbols
    except Exception as e:
        log_queue.put((logging.ERROR, f"Error en fetch_current_symbols_fast: {e}"))
        return set()

# Código principal
if __name__ == "__main__":
    initialize_db()
    log_queue.put((logging.INFO, "Iniciando bot de trading en vivo (Spot)."))
    known_symbols = fetch_current_symbols_fast()
    log_queue.put((logging.INFO, f"Símbolos iniciales: {len(known_symbols)}"))

    threading.Thread(target=keep_markets_updated, daemon=True).start()
    start_ws_thread()

    try:
        while True:
            time.sleep(60)  # Mantener el programa vivo
    except KeyboardInterrupt:
        log_queue.put((logging.INFO, "Bot detenido por el usuario."))