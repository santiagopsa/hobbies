import ccxt
import websocket
import json
import threading
import logging
import time
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from decimal import Decimal, getcontext
import requests

# Configurar la precisión decimal (ajústala según tus necesidades)
getcontext().prec = 10

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,  # Cambia a DEBUG para mayor detalle
    filename='new_coins_ws.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configurar API de Binance con CCXT (para trading y trailing stop)
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY_REAL'),
    'secret': os.getenv('BINANCE_SECRET_KEY_REAL'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
})

# Configurar API de Telegram (opcional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    """
    Envía un mensaje de texto a Telegram.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram token o chat ID no configurado.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload, timeout=3)
        if response.status_code == 200:
            logging.info("Mensaje enviado a Telegram.")
        else:
            logging.error(f"Error al enviar mensaje a Telegram: {response.text}")
    except Exception as e:
        logging.error(f"Excepción al enviar mensaje a Telegram: {e}")

# --- Base de Datos y Control de Compras ---
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
        logging.error(f"Error al insertar transacción: {e}")

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

# --- Funciones para Obtener Listado de Símbolos ---
def fetch_current_symbols():
    """
    Obtiene la lista completa de símbolos usando load_markets (método completo, más lento).
    """
    try:
        markets = exchange.load_markets(True)
        symbols = [symbol.upper() for symbol in markets.keys() if symbol.endswith('/USDT')]
        return list(set(symbols))
    except Exception as e:
        logging.error(f"Error al cargar mercados: {e}")
        return []

def fetch_current_symbols_fast():
    """
    Obtiene la lista de símbolos utilizando el endpoint directo de exchangeInfo.
    Es más rápido y retorna un conjunto.
    """
    try:
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=1)
        data = response.json()
        symbols = []
        for s in data.get("symbols", []):
            if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                base = s.get("baseAsset")
                quote = s.get("quoteAsset")
                symbols.append(f"{base}/{quote}")
        return set(symbols)
    except Exception as e:
        logging.error(f"Error en fetch_current_symbols_fast: {e}")
        return set()

def get_new_symbols(previous_symbols, current_symbols):
    """
    Compara el conjunto de símbolos previos con el actual para detectar nuevos.
    """
    return list(current_symbols - previous_symbols)

# --- Funciones de Trading: Precio, Compra, Venta, Trailing Stop ---

def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error al obtener el precio para {symbol}: {e}")
        return None

import time
import logging

def buy_symbol_fast(symbol):
    """
    Intenta comprar la moneda en el instante exacto en que esté disponible en Binance.
    Reintenta enviar la orden cada 50ms hasta que Binance acepte el símbolo.
    """
    max_attempts = 50  # Número máximo de intentos (ajusta según pruebas)
    retry_delay = 0.05  # 50ms entre intentos

    logging.info(f"🚀 Intentando comprar {symbol} en cuanto esté disponible...")

    for attempt in range(max_attempts):
        exchange.load_markets()  # Recargar mercados para detectar cuándo se habilita el símbolo
        if symbol in exchange.markets:  # Verificar si la moneda ya está en Binance
            logging.info(f"✅ {symbol} encontrado en Binance en intento {attempt+1}. Ejecutando compra...")

            # Obtener el precio actual
            price = fetch_price(symbol)
            if price is None:
                logging.error(f"⚠️ No se pudo obtener el precio de {symbol}.")
                return None

            price_limit = round(price * 1.02, 6)  # Precio límite 2% por encima del detectado
            budget = 5  # USDT a invertir
            amount = budget / price_limit
            amount = exchange.amount_to_precision(symbol, amount)

            # Crear orden límite con IOC
            order = exchange.create_order(
                symbol=symbol,
                type="limit",
                side="buy",
                amount=amount,
                price=price_limit,
                params={"timeInForce": "IOC"}
            )

            order_price = order.get('price')
            filled = order.get('filled', 0)
            timestamp = datetime.now(timezone.utc).isoformat()
            insert_transaction(symbol, 'buy', order_price, filled, timestamp)

            logging.info(f"✅ Orden ejecutada: {symbol} a {order_price} USDT, cantidad: {filled}")
            send_telegram_message(f"✅ *Orden de compra ejecutada*\nSímbolo: `{symbol}`\nPrecio: `{order_price} USDT`\nCantidad: `{filled}`")
            return {'price': order_price, 'filled': filled}

        else:
            logging.info(f"⏳ {symbol} aún no está disponible. Intento {attempt+1}/{max_attempts}...")
            time.sleep(retry_delay)  # Esperar 50ms antes de reintentar

    logging.error(f"❌ {symbol} no estuvo disponible después de {max_attempts} intentos.")
    send_telegram_message(f"❌ *Error*: `{symbol}` no estuvo disponible después de varios intentos.")
    return None



def sell_symbol(symbol, amount):
    """
    Antes de vender, se fuerza la actualización de mercados si el símbolo no está presente.
    """
    if symbol not in exchange.markets:
        logging.info(f"Símbolo {symbol} no encontrado en exchange.markets. Actualizando mercados...")
        exchange.load_markets()
    try:
        base_asset = symbol.split('/')[0]
        balance = exchange.fetch_balance()
        available = balance.get(base_asset, {}).get('free', 0)
        if available < amount:
            logging.warning(f"Balance insuficiente para {symbol}: disponible {available} vs pedido {amount}.")
            amount = available
        safe_amount = float(amount) * 0.999
        safe_amount = exchange.amount_to_precision(symbol, safe_amount)
        order = exchange.create_market_sell_order(symbol, safe_amount)
        order_price = order.get('average', order.get('price', None))
        ts = datetime.now(timezone.utc).isoformat()
        insert_transaction(symbol, 'sell', order_price, safe_amount, ts)
        send_telegram_message(f"✅ *Venta ejecutada*\nSímbolo: `{symbol}`\nPrecio: `{order_price} USDT`\nCantidad: `{safe_amount}`")
        logging.info(f"Venta ejecutada: {symbol} a {order_price} USDT, cantidad: {safe_amount}")
        return order
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")
        send_telegram_message(f"❌ *Error al vender* `{symbol}`\nDetalles: {e}")
        return None

def set_trailing_stop(symbol, amount, purchase_price, trailing_percent=5):
    try:
        logging.info(f"Configurando trailing stop para {symbol} con trailing del {trailing_percent}%")
        send_telegram_message(f"🔄 *Trailing Stop configurado* para `{symbol}`")
        highest_price = purchase_price
        # Espera a que el precio supere el precio de compra
        while True:
            current_price = fetch_price(symbol)
            if current_price is None:
                time.sleep(0.5)
                continue
            if current_price > purchase_price:
                highest_price = current_price
                break
            time.sleep(0.3)
        # Monitoreo activo
        while True:
            current_price = fetch_price(symbol)
            if current_price is None:
                time.sleep(0.5)
                continue
            if current_price > highest_price:
                highest_price = current_price
                logging.info(f"{symbol}: Nuevo máximo alcanzado: {highest_price} USDT")
                send_telegram_message(f"📈 *Nuevo máximo* para `{symbol}`: {highest_price} USDT")
            stop_price = highest_price * (1 - trailing_percent / 100)
            if current_price < stop_price:
                logging.info(f"{symbol}: Precio {current_price} USDT cayó por debajo del trailing stop ({stop_price} USDT). Ejecutando venta inmediata.")
                send_telegram_message(f"🔴 *Trailing Stop activado* para `{symbol}`. Ejecutando venta inmediata.")
                sell_symbol(symbol, amount)
                break
            time.sleep(0.5)
    except Exception as e:
        logging.error(f"Error en trailing stop para {symbol}: {e}")
        send_telegram_message(f"❌ *Error en trailing stop* `{symbol}`\nDetalles: {e}")

def process_order(symbol, order_details):
    if order_details:
        purchase_price = order_details.get('price')
        amount = order_details.get('filled')
        if purchase_price is None or amount is None or float(amount) <= 0:
            logging.error(f"Datos insuficientes para configurar el trailing stop en {symbol}.")
            send_telegram_message(f"❌ *Error*: Datos insuficientes para trailing stop en `{symbol}`.")
            return
        time.sleep(5)  # Breve espera para actualización de balances
        threading.Thread(target=set_trailing_stop, args=(symbol, amount, purchase_price, 5), daemon=True).start()

# --- Función de Sincronización para HTTP (para comparar) ---
def wait_for_next_hour_polling():
    """
    Espera hasta un instante antes de la hora exacta, tomando en cuenta un tiempo estimado de latencia HTTP.
    Finaliza el polling 'http_latency_estimate' segundos antes de la hora, para iniciar la consulta HTTP inmediatamente.
    """
    http_latency_estimate = 1  # Tiempo estimado en segundos para la llamada HTTP
    now = datetime.now()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    target_time = next_hour - timedelta(seconds=http_latency_estimate)
    sleep_until = target_time - timedelta(seconds=0.2)
    sleep_time = (sleep_until - datetime.now()).total_seconds()
    if sleep_time > 0:
        time.sleep(sleep_time)
    while datetime.now() < target_time:
        time.sleep(0.005)

# --- Función de WebSocket para detectar nuevos listados ---
import websocket

known_symbols = set()

def on_message(ws, message):
    global known_symbols
    try:
        tickers = json.loads(message)
        for ticker in tickers:
            symbol_raw = ticker.get("s")  # Ej: "HEIUSDT"
            if symbol_raw and symbol_raw.endswith("USDT"):
                formatted_symbol = f"{symbol_raw[:-4]}/{symbol_raw[-4:]}"
                if formatted_symbol not in known_symbols:
                    known_symbols.add(formatted_symbol)
                    event_time_ms = ticker.get("E")
                    event_time = event_time_ms / 1000.0 if event_time_ms else None
                    detection_time = datetime.now(timezone.utc)
                    latency = detection_time.timestamp() - event_time if event_time is not None else None
                    logging.info(f"Nueva moneda detectada vía WebSocket: {formatted_symbol} a las {detection_time.isoformat()}")
                    if latency is not None:
                        logging.info(f"Latencia de detección: {latency:.3f} s")
                    threading.Thread(target=execute_trade, args=(formatted_symbol,), daemon=True).start()
    except Exception as e:
        logging.error(f"Error en on_message: {e}")


def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket cerrado")

def on_open(ws):
    logging.info("Conexión WebSocket abierta")

def start_websocket():
    url = "wss://stream.binance.com:9443/ws/!ticker@arr"
    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()

def start_ws_thread():
    threading.Thread(target=start_websocket, daemon=True).start()

# --- Función para ejecutar la compra y trailing stop ---
def execute_trade(symbol):
    send_telegram_message(f"🚀 *Nueva moneda detectada*: `{symbol}`")
    order_details = buy_symbol_fast(symbol)
    if order_details:
        process_order(symbol, order_details)

# --- Código Principal ---
if __name__ == "__main__":
    import json
    initialize_db()
    logging.info("Iniciando bot de trading de nuevas monedas vía WebSocket.")
    # Carga inicial de símbolos a través del método rápido
    known_symbols = set(fetch_current_symbols_fast())
    logging.info(f"Símbolos iniciales (WS): {len(known_symbols)}")
    # Inicia el WebSocket en un hilo separado
    start_ws_thread()
    
    # También puedes, si lo deseas, ejecutar el método HTTP (polling) para comparar
    # Por ejemplo, ejecutar wait_for_next_hour_polling() y luego fetch_current_symbols_fast()
    try:
        while True:
            # Espera hasta un instante antes de la hora exacta, según la latencia estimada
            t_sync_start = time.time()
            wait_for_next_hour_polling()
            t_sync_end = time.time()
            logging.info(f"Tiempo de sincronización (wait_for_next_hour_polling): {t_sync_end - t_sync_start:.3f} s")
            
            t_fetch_start = time.time()
            current_symbols = set(fetch_current_symbols_fast())
            t_fetch_end = time.time()
            logging.info(f"Tiempo de fetch_current_symbols (HTTP): {t_fetch_end - t_fetch_start:.3f} s")
            
            # Comparar para detectar nuevos símbolos vía HTTP (para comparar)
            new_symbols = list(current_symbols - known_symbols)
            if new_symbols:
                logging.info(f"Nuevas monedas detectadas vía HTTP: {new_symbols}")
                for symbol in new_symbols:
                    threading.Thread(target=execute_trade, args=(symbol,), daemon=True).start()
                # Actualiza known_symbols para evitar duplicados
                known_symbols |= current_symbols
            else:
                logging.info("No se detectaron nuevas monedas vía HTTP en esta iteración.")
            
            time.sleep(1)  # Espera 1 s antes del siguiente ciclo (para fines de comparación)
    except KeyboardInterrupt:
        logging.info("Programa terminado por el usuario.")
