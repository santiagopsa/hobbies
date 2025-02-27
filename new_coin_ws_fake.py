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

def get_sleep_duration():
    now = datetime.now().time()
    # Definir la ventana crítica (ajusta las horas según tu zona horaria)
    critical_start = datetime.strptime("03:58", "%H:%M").time()
    critical_end = datetime.strptime("04:02", "%H:%M").time()
    if critical_start <= now <= critical_end:
        return 0.2  # Consultas rápidas en la ventana crítica (cada 0.2 segundos)
    else:
        return 5    # Consultas normales fuera de la ventana

new_coin_detected = False
# Pre-carga de mercados en segundo plano
def keep_markets_updated():
    global new_coin_detected
    while True:
        now_dt = datetime.now()
        minute = now_dt.minute

        # Estamos en la ventana crítica: de XX:58 a XX:01
        if minute >= 58 or minute < 2:
            try:
                exchange.load_markets(reload=True)
                log_queue.put((logging.DEBUG, "Mercados actualizados en rango crítico."))
                
                # Si se detectó una nueva moneda, se pausa para dar tiempo a que se ejecute la compra
                if new_coin_detected:
                    log_queue.put((logging.DEBUG, "Nueva moneda detectada, durmiendo para permitir la compra."))
                    time.sleep(10)
                    new_coin_detected = False
                else:
                    time.sleep(0.2)  # Actualizaciones rápidas dentro del rango crítico
            except ccxt.RateLimitExceeded:
                log_queue.put((logging.WARNING, "Límite de API excedido en actualización de mercados. Esperando..."))
                time.sleep(10)
            except Exception as e:
                log_queue.put((logging.ERROR, f"Error al actualizar mercados: {e}"))
                time.sleep(10)
        else:
            # Fuera de la ventana crítica, dormir hasta el próximo minuto 58
            next_critical = now_dt.replace(minute=58, second=0, microsecond=0)
            if next_critical <= now_dt:
                next_critical += timedelta(hours=1)
            sleep_time = (next_critical - now_dt).total_seconds()
            log_queue.put((logging.DEBUG, f"Fuera del rango crítico. Durmiendo por {sleep_time:.1f} segundos hasta el inicio del rango crítico."))
            time.sleep(sleep_time)


# Función para obtener precio
def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except ccxt.RateLimitExceeded:
        log_queue.put((logging.WARNING, f"Límite de API excedido al obtener precio para {symbol}"))
        time.sleep(1)
        return None
    except Exception as e:
        log_queue.put((logging.ERROR, f"Error al obtener precio para {symbol}: {e}"))
        return None

# Compra optimizada con reintentos hasta que el mercado esté disponible
def buy_symbol_microsecond(symbol, budget=5, max_attempts=100, retry_delay=0.01):
    daily_purchases = get_daily_purchases()
    if daily_purchases >= MAX_DAILY_PURCHASES:
        log_queue.put((logging.INFO, f"Límite diario alcanzado ({MAX_DAILY_PURCHASES}). Ignorando {symbol}."))
        send_telegram_message(f"⚠️ Límite diario alcanzado. No se comprará `{symbol}`.")
        return None

    log_queue.put((logging.INFO, f"🚀 Intentando comprar {symbol} tan pronto esté disponible..."))
    for attempt in range(max_attempts):
        try:
            exchange.load_markets(reload=True)
            if symbol not in exchange.markets:
                log_queue.put((logging.DEBUG, f"{symbol} no disponible aún. Intento {attempt+1}/{max_attempts}"))
                time.sleep(retry_delay)
                continue

            market = exchange.markets[symbol]
            if market.get('info', {}).get('status') != 'TRADING':
                log_queue.put((logging.DEBUG, f"{symbol} detectado pero no en estado TRADING. Intento {attempt+1}/{max_attempts}"))
                time.sleep(retry_delay)
                continue

            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
            if not price:
                log_queue.put((logging.ERROR, f"No se pudo obtener precio para {symbol}"))
                time.sleep(retry_delay)
                continue

            # Calcula la cantidad basada en el presupuesto
            calculated_amount = budget / price

            # Obtén el valor notional mínimo definido en el mercado
            min_notional = market.get('limits', {}).get('cost', {}).get('min', 0)
            # Calcula la cantidad mínima necesaria para cumplir con el notional
            min_amount = min_notional / price

            # Usa el máximo entre la cantidad calculada y la cantidad mínima
            amount = max(calculated_amount, min_amount)
            start_time = time.time()
            order = exchange.create_market_buy_order(symbol, amount)
            end_time = time.time()

            order_price = order.get('average', order.get('price', price))
            filled = order.get('filled', 0)
            timestamp = datetime.now(timezone.utc).isoformat()

            insert_transaction(symbol, 'buy', order_price, filled, timestamp)
            increment_daily_purchases()
            latency_ms = (end_time - start_time) * 1000
            log_queue.put((logging.INFO, f"✅ Compra ejecutada: {symbol} a {order_price} USDT | Cantidad: {filled} | Latencia: {latency_ms:.3f}ms"))
            send_telegram_message(f"✅ *Compra ejecutada*\nSímbolo: `{symbol}`\nPrecio: `{order_price}`\nCantidad: `{filled}`\nLatencia: `{latency_ms:.3f}ms`")

            return {'price': order_price, 'filled': filled}

        except ccxt.RateLimitExceeded:
            log_queue.put((logging.WARNING, f"Límite de API excedido al intentar comprar {symbol}. Esperando..."))
            time.sleep(1)
        except ccxt.ExchangeError as e:
            log_queue.put((logging.ERROR, f"Error de exchange al comprar {symbol}: {e}"))
            time.sleep(retry_delay)
        except Exception as e:
            log_queue.put((logging.ERROR, f"Excepción al comprar {symbol}: {e}"))
            time.sleep(retry_delay)

    log_queue.put((logging.ERROR, f"❌ {symbol} no estuvo disponible tras {max_attempts} intentos."))
    send_telegram_message(f"❌ *Error*: `{symbol}` no estuvo disponible tras varios intentos.")
    return None

# Venta
def sell_symbol(symbol, amount):
    try:
        if symbol not in exchange.markets:
            exchange.load_markets(reload=True)
        base_asset = symbol.split('/')[0]
        balance = exchange.fetch_balance()
        available = balance.get(base_asset, {}).get('free', 0)
        if available < amount:
            amount = available
        # Usar el mismo amount de compra sin convertir a precisión,
        # opcionalmente multiplicamos por 0.999 para evitar vender más de lo disponible.
        safe_amount = amount * 0.999  
        order = exchange.create_market_sell_order(symbol, safe_amount)
        order_price = order.get('average', order.get('price'))
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
                time.sleep(0.5)
                continue

            if current_price > highest_price:
                highest_price = current_price
                log_queue.put((logging.INFO, f"📈 Nuevo máximo: {symbol} a {highest_price}"))

            stop_price = highest_price * (1 - trailing_percent / 100)
            if current_price <= stop_price:
                log_queue.put((logging.INFO, f"🔴 Trailing stop activado: {symbol} vendido a {current_price}"))
                sell_symbol(symbol, amount)
                break
            time.sleep(0.5)
        except ccxt.RateLimitExceeded:
            log_queue.put((logging.ERROR, f"Límite de API excedido en trailing stop para {symbol}"))
            time.sleep(1)
        except Exception as e:
            log_queue.put((logging.ERROR, f"Error en trailing stop para {symbol}: {e}"))
            time.sleep(1)
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
                new_coin_detected = True  # Se detectó una nueva moneda

                event_time_ms = ticker.get("E")
                detection_time = datetime.now(timezone.utc)
                latency = (detection_time.timestamp() - (event_time_ms / 1000)) if event_time_ms else None
                latency_str = f"{latency:.6f}" if latency is not None else "N/A"
                log_queue.put((logging.INFO, f"🚀 Nueva moneda detectada: {formatted_symbol} | Latencia: {latency_str}s"))
                send_telegram_message(f"🚀 *Nueva moneda detectada*: `{formatted_symbol}`")

                # Intentar compra inmediata con reintentos
                order_details = buy_symbol_microsecond(formatted_symbol)
                if order_details:
                    process_order(formatted_symbol, order_details)
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
    log_queue.put((logging.INFO, "Iniciando bot de trading en vivo (Spot) - MODO DE PRUEBA REAL."))
    known_symbols = fetch_current_symbols_fast()
    log_queue.put((logging.INFO, f"Símbolos iniciales: {len(known_symbols)}"))
    
    # Inicia el thread que actualiza mercados durante la ventana crítica de cada hora
    threading.Thread(target=keep_markets_updated, daemon=True).start()
    # Inicia el thread del WebSocket real (puedes mantenerlo o no para este test)
    start_ws_thread()

    # -----------------------------
    # BLOQUE DE PRUEBA: Forzar mensaje de WebSocket
    # -----------------------------
    import json
    # Construimos un mensaje simulando la detección de una nueva moneda
    fake_message = json.dumps([{
        "s": "DOGEUSDT",  # Esto se convertirá en "TESTCOIN/USDT"
        "E": int(time.time() * 1000)
    }])
    # Llamamos a on_message directamente para activar la lógica de compra
    on_message(None, fake_message)
    # -----------------------------
    
    try:
        while True:
            time.sleep(60)  # Mantener el programa vivo
    except KeyboardInterrupt:
        log_queue.put((logging.INFO, "Bot detenido por el usuario."))