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
from binance.client import Client

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    filename='new_coins_ws.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Binance client (assumed to be defined globally in your code)
binance_client = Client(os.getenv('BINANCE_API_KEY_REAL'), os.getenv('BINANCE_SECRET_KEY_REAL'))
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

# Configuración de la API de Binance para spot
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

daily_purchases_count = 0
current_date = datetime.now().strftime('%Y-%m-%d')

def get_daily_purchases():
    global daily_purchases_count, current_date
    today = datetime.now().strftime('%Y-%m-%d')
    if today != current_date:
        daily_purchases_count = 0
        current_date = today
    return daily_purchases_count

def increment_daily_purchases():
    global daily_purchases_count
    daily_purchases_count += 1

# Variables globales
market_cache = {}
active_orders = set()
MIN_NOTIONAL = 20  # Mínimo notional en $10 USD para nuevas monedas

# Función para obtener precio (mejorada)
def fetch_price(symbol, ws_ticker=None):
    try:
        if ws_ticker and 'c' in ws_ticker:  # 'c' es el último precio en el stream de Binance
            log_queue.put((logging.DEBUG, f"Usando precio del WebSocket para {symbol}: {ws_ticker['c']}"))
            return float(ws_ticker['c'])
        ticker = exchange.fetch_ticker(symbol)
        log_queue.put((logging.DEBUG, f"Ticker obtenido para {symbol}: {ticker}"))
        return ticker['last'] if ticker.get('last') else ticker.get('bid')  # Fallback al precio de oferta
    except ccxt.RateLimitExceeded:
        log_queue.put((logging.WARNING, f"Límite de API excedido al obtener precio para {symbol}"))
        time.sleep(0.1)
        return None
    except Exception as e:
        log_queue.put((logging.ERROR, f"Error al obtener precio para {symbol}: {e}"))
        return None

# Cache de mercados
def update_market_cache():
    global market_cache
    while True:
        try:
            market_cache = exchange.load_markets()
            log_queue.put((logging.DEBUG, "Mercados actualizados en caché."))
            time.sleep(60)  # Actualizar cada minuto
        except Exception as e:
            log_queue.put((logging.ERROR, f"Error al actualizar caché de mercados: {e}"))
            time.sleep(10)

threading.Thread(target=update_market_cache, daemon=True).start()

# Compra optimizada con reintentos y verificación de mercado
buying_lock = threading.Lock()

def buy_symbol_microsecond(symbol, ws_ticker=None, budget=20, max_attempts=10, retry_delay=0.0005):
    """
    Execute a buy order for a new coin with minimal latency, handling delayed market info.
    
    Args:
        symbol (str): Trading pair, e.g., 'WCT/USDT'
        ws_ticker (dict): WebSocket ticker data, if available
        budget (float): Amount in USDT to spend (default: 20)
        max_attempts (int): Number of retry attempts (default: 10)
        retry_delay (float): Initial delay between retries in seconds (default: 0.0005)
    
    Returns:
        dict: Order details {'price': float, 'filled': float} or None if failed
    """
    # Check daily purchase limit
    daily_purchases = get_daily_purchases()
    if daily_purchases >= MAX_DAILY_PURCHASES:
        log_queue.put((logging.INFO, f"Límite diario alcanzado ({MAX_DAILY_PURCHASES}). Ignorando {symbol}."))
        send_telegram_message(f"⚠️ Límite diario alcanzado. No se comprará `{symbol}`.")
        return None

    # Prevent duplicate orders
    with buying_lock:
        if symbol in active_orders:
            log_queue.put((logging.WARNING, f"Orden ya activa para {symbol}. Ignorando."))
            return None
        active_orders.add(symbol)

    log_queue.put((logging.INFO, f"🚀 Comprando {symbol} inmediatamente..."))
    attempt = 0
    current_delay = retry_delay

    while attempt < max_attempts:
        try:
            # Fetch price from WebSocket or API
            price = ws_ticker.get("p") if ws_ticker and ws_ticker.get("p") else fetch_price(symbol)
            if not price:
                raise BinanceAPIException("Precio no disponible", None)

            # Calculate amount based on budget
            amount = budget / float(price)

            # Attempt to fetch market info
            try:
                market = binance_client.get_symbol_info(symbol.replace("/", ""))
                if not market:
                    raise BinanceAPIException("Información del mercado no disponible", None)
                lot_size = float(market["filters"][2]["minQty"])  # LOT_SIZE filter
                price_precision = int(market["filters"][0]["tickSize"].find("1") - 1) if market["filters"][0]["tickSize"].find("1") >= 0 else 8
                amount = max(amount, lot_size)
                amount = round(amount / lot_size) * lot_size  # Adjust to lot size
            except BinanceAPIException:
                # Fallback: Assume permissive lot size and precision
                log_queue.put((logging.WARNING, f"Market info no disponible para {symbol}. Usando valores predeterminados."))
                lot_size = 0.0001  # Minimal lot size
                price_precision = 8  # Common precision for USDT pairs
                amount = max(amount, lot_size)
                amount = round(amount / lot_size) * lot_size

            # Try limit order first
            try:
                limit_price = float(price) * 1.01  # 1% premium
                limit_price = round(limit_price, price_precision)  # Adjust to price precision
                start_time = time.time()
                order = binance_client.create_order(
                    symbol=symbol.replace("/", ""),
                    side="BUY",
                    type="LIMIT",
                    quantity=f"{amount:.8f}",
                    price=f"{limit_price:.8f}",
                    timeInForce="GTC"
                )
                end_time = time.time()
            except BinanceAPIException as e:
                log_queue.put((logging.WARNING, f"Orden límite fallida para {symbol}: {e}. Intentando orden de mercado..."))
                # Fallback to market order
                start_time = time.time()
                order = binance_client.create_order(
                    symbol=symbol.replace("/", ""),
                    side="BUY",
                    type="MARKET",
                    quantity=f"{amount:.8f}"
                )
                end_time = time.time()

            # Extract order details
            order_price = float(order["fills"][0]["price"]) if order.get("fills") else float(price)
            filled = float(order["executedQty"])
            timestamp = datetime.now(timezone.utc).isoformat()

            # Log transaction to database
            insert_transaction(symbol, 'buy', order_price, filled, timestamp)
            increment_daily_purchases()

            # Calculate and log latency
            latency_ms = (end_time - start_time) * 1000
            log_queue.put((logging.INFO, f"✅ Compra ejecutada: {symbol} a {order_price} USDT | Cantidad: {filled} | Latencia: {latency_ms:.3f}ms | Intento: {attempt + 1}"))
            send_telegram_message(
                f"✅ *Compra ejecutada*\n"
                f"Símbolo: `{symbol}`\n"
                f"Precio: `{order_price}`\n"
                f"Cantidad: `{filled}`\n"
                f"Latencia: `{latency_ms:.3f}ms\n"
                f"Intento: `{attempt + 1}`"
            )

            # Remove from active orders and trigger trailing stop
            active_orders.remove(symbol)
            order_details = {'price': order_price, 'filled': filled}
            process_order(symbol, order_details)

            return {'price': order_price, 'filled': filled}

        except BinanceAPIException as e:
            log_queue.put((logging.WARNING, f"Intento {attempt + 1}/{max_attempts} fallido para {symbol}: {e}"))
            attempt += 1
            if attempt < max_attempts:
                # Adaptive delay: increase slightly to avoid hammering API
                current_delay = min(current_delay * 1.5, 0.01)  # Cap at 10ms
                time.sleep(current_delay)
            continue
        except Exception as e:
            log_queue.put((logging.ERROR, f"Error inesperado al comprar {symbol} en intento {attempt + 1}: {e}"))
            send_telegram_message(f"❌ *Error*: `{symbol}` no se pudo comprar: {e}")
            attempt += 1
            if attempt < max_attempts:
                current_delay = min(current_delay * 1.5, 0.01)
                time.sleep(current_delay)
            continue

    # Cleanup on failure
    log_queue.put((logging.ERROR, f"Compra fallida para {symbol} tras {max_attempts} intentos"))
    send_telegram_message(f"❌ *Compra fallida*: `{symbol}` tras {max_attempts} intentos")
    if symbol in active_orders:
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
        min_notional = market.get('limits', {}).get('cost', {}).get('min', MIN_NOTIONAL)
        if notional < min_notional:
            amount = min_notional / current_price
            amount = max(amount, market['limits']['amount']['min'])
            amount = round(amount / market['limits']['amount']['stepSize']) * market['limits']['amount']['stepSize']
        safe_amount = amount * 0.999  # Margen de seguridad
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
    log_queue.put((logging.INFO, f"🔄 Trailing stop iniciado para {symbol} con precio inicial {purchase_price}"))

    while time.time() - start_time < max_duration:
        try:
            current_price = fetch_price(symbol)
            if not current_price:
                log_queue.put((logging.WARNING, f"Precio no disponible para {symbol}"))
                time.sleep(0.1)
                continue

            log_queue.put((logging.DEBUG, f"Precio actual de {symbol}: {current_price}, máximo: {highest_price}"))
            if current_price > highest_price:
                highest_price = current_price
                log_queue.put((logging.INFO, f"📈 Nuevo máximo: {symbol} a {highest_price}"))

            stop_price = highest_price * (1 - trailing_percent / 100)
            log_queue.put((logging.DEBUG, f"Stop price calculado: {stop_price}"))
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
        log_queue.put((logging.ERROR, f"Datos insuficientes para trailing stop en {symbol}: {order_details}"))
        return
    log_queue.put((logging.DEBUG, f"Iniciando trailing stop para {symbol} con precio {purchase_price} y cantidad {amount}"))
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

                threading.Thread(target=buy_symbol_microsecond, args=(formatted_symbol, ticker), daemon=True).start()
                threading.Thread(target=send_telegram_message, args=(f"🚀 *Nueva moneda detectada*: `{formatted_symbol}`",), daemon=True).start()
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
    sleep_time = (next_critical - now_dt).total_seconds() - 5  # Despertar 5 segundos antes
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
                    time.sleep(0.1)
                    new_coin_detected = False
                else:
                    time.sleep(0.05)
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

# Obtener símbolos iniciales
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