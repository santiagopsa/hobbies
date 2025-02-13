import ccxt
import time
import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
import sqlite3
import requests
import threading
from decimal import Decimal, ROUND_UP, getcontext

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

# Configurar API de Binance
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY_REAL'),
    'secret': os.getenv('BINANCE_SECRET_KEY_REAL'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'
    }
})

# Configurar API de Telegram
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
        response = requests.post(url, json=payload)
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
    """
    Inicializa la base de datos SQLite para registrar transacciones y contar compras diarias.
    """
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
    """
    Registra una transacción en la base de datos.
    """
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
    """
    Retorna la cantidad de compras realizadas hoy.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT count FROM daily_purchases WHERE date = ?", (today,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0

def increment_daily_purchases():
    """
    Incrementa el contador de compras diarias.
    """
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
        markets = exchange.load_markets(True)  # Forzar recarga completa
        symbols = [symbol.upper() for symbol in markets.keys() if symbol.endswith('/USDT')]
        return list(set(symbols))
    except Exception as e:
        logging.error(f"Error al cargar mercados: {e}")
        return []

def fetch_current_symbols_fast():
    """
    Obtiene la lista de símbolos disponibles utilizando el endpoint directo de exchangeInfo.
    Esto es más rápido y retorna un conjunto.
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
    """
    Retorna el precio actual del símbolo.
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error al obtener el precio para {symbol}: {e}")
        return None

def buy_symbol(symbol):
    """
    Realiza una orden de compra de mercado para la nueva moneda detectada,
    verificando el límite diario.
    """
    if get_daily_purchases() >= MAX_DAILY_PURCHASES:
        logging.info(f"⚠️ Límite de compras diarias alcanzado ({MAX_DAILY_PURCHASES}/día).")
        send_telegram_message(f"⚠️ *Límite de compras alcanzado*: No se comprará `{symbol}` hoy.")
        return None

    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        budget = 5  # Presupuesto en USDT para la compra
        amount = budget / price
        amount = exchange.amount_to_precision(symbol, amount)
        order = exchange.create_market_buy_order(symbol, amount)
        order_price = order.get('average', order.get('price', None))
        filled = order.get('filled', 0)
        timestamp = datetime.now(timezone.utc).isoformat()
        insert_transaction(symbol, 'buy', order_price, filled, timestamp)
        increment_daily_purchases()
        logging.info(f"✅ Compra realizada: {symbol} a {order_price} USDT, cantidad: {filled}")
        send_telegram_message(f"✅ *Compra realizada*\nSímbolo: `{symbol}`\nPrecio: `{order_price} USDT`\nCantidad: `{filled}`")
        return {'price': order_price, 'filled': filled}
    except Exception as e:
        logging.error(f"Error al comprar {symbol}: {e}")
        send_telegram_message(f"❌ *Error al comprar* `{symbol}`\nDetalles: {e}")
        return None

def sell_symbol(symbol, amount):
    """
    Ejecuta una orden de venta de mercado y registra la transacción.
    """
    try:
        base_asset = symbol.split('/')[0]
        balance = exchange.fetch_balance()
        available = balance.get(base_asset, {}).get('free', 0)
        if available < amount:
            logging.warning(f"Balance insuficiente para {symbol}: disponible {available} vs pedido {amount}.")
            amount = available
        safe_amount = float(amount) * 0.999  # Aplica margen de seguridad
        safe_amount = exchange.amount_to_precision(symbol, safe_amount)
        order = exchange.create_market_sell_order(symbol, safe_amount)
        order_price = order.get('average', order.get('price', None))
        timestamp = datetime.now(timezone.utc).isoformat()
        insert_transaction(symbol, 'sell', order_price, safe_amount, timestamp)
        send_telegram_message(f"✅ *Venta ejecutada*\nSímbolo: `{symbol}`\nPrecio: `{order_price} USDT`\nCantidad: `{safe_amount}`")
        logging.info(f"Venta ejecutada: {symbol} a {order_price} USDT, cantidad: {safe_amount}")
        return order
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")
        send_telegram_message(f"❌ *Error al vender* `{symbol}`\nDetalles: {e}")
        return None

def set_trailing_stop(symbol, amount, purchase_price, trailing_percent=5):
    """
    Monitorea el precio del símbolo y, una vez activado el trailing stop,
    vende inmediatamente si el precio cae por debajo del nivel calculado.
    """
    try:
        logging.info(f"Configurando trailing stop para {symbol} con trailing del {trailing_percent}%")
        send_telegram_message(f"🔄 *Trailing Stop configurado* para `{symbol}`")
        highest_price = purchase_price

        # Espera a que el precio supere el precio de compra para activar el trailing stop
        while True:
            current_price = fetch_price(symbol)
            if current_price is None:
                time.sleep(0.5)
                continue
            if current_price > purchase_price:
                highest_price = current_price
                break
            time.sleep(0.3)

        # Monitoreo activo del precio
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
    """
    Después de realizar la compra, espera unos segundos y lanza el trailing stop en un hilo.
    """
    if order_details:
        purchase_price = order_details.get('price')
        amount = order_details.get('filled')
        if purchase_price is None or amount is None or float(amount) <= 0:
            logging.error(f"Datos insuficientes para configurar el trailing stop en {symbol}.")
            send_telegram_message(f"❌ *Error*: Datos insuficientes para trailing stop en `{symbol}`.")
            return
        time.sleep(5)  # Pequeña pausa para que se actualicen balances, etc.
        threading.Thread(target=set_trailing_stop, args=(symbol, amount, purchase_price, 5), daemon=True).start()

# --- Función de Sincronización y Ejecución de la Lógica de Trading ---

def wait_for_next_hour_polling():
    """
    Espera de forma eficiente hasta un instante antes de la hora exacta,
    tomando en cuenta un tiempo estimado de latencia HTTP (por ejemplo, 1 segundo).

    Esto permite iniciar la adquisición de datos justo antes de la hora,
    de modo que la respuesta HTTP (que tarda aproximadamente 1 s)
    se reciba lo más cerca posible de la hora exacta.
    """
    http_latency_estimate = 1  # Tiempo estimado en segundos que tarda la llamada HTTP
    now = datetime.now()
    # Calcula el inicio de la siguiente hora
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    # Define el instante objetivo para finalizar el polling: 1 segundo antes de next_hour
    target_time = next_hour - timedelta(seconds=http_latency_estimate)

    # Dormir hasta 0.2 segundos antes del target para luego hacer polling fino
    sleep_until = target_time - timedelta(seconds=0.2)
    sleep_time = (sleep_until - datetime.now()).total_seconds()
    if sleep_time > 0:
        time.sleep(sleep_time)
    
    # Polling de alta frecuencia hasta alcanzar target_time
    while datetime.now() < target_time:
        time.sleep(0.005)


def execute_trade(symbol):
    """
    Ejecuta inmediatamente la compra y lanza el trailing stop para el símbolo detectado.
    """
    send_telegram_message(f"🚀 *Nueva moneda detectada*: `{symbol}`")
    order_details = buy_symbol(symbol)
    if order_details:
        process_order(symbol, order_details)

def main():
    initialize_db()
    logging.info("Iniciando bot de trading de nuevas monedas.")
    # Carga inicial de símbolos (puedes usar fetch_current_symbols o fetch_current_symbols_fast)
    previous_symbols = set(fetch_current_symbols())
    logging.info(f"Símbolos iniciales cargados: {len(previous_symbols)}")
    
    while True:
        try:
            # Sincroniza con el cambio de hora
            t_sync_start = time.time()
            wait_for_next_hour_polling()
            t_sync_end = time.time()
            logging.info(f"Tiempo de sincronización (wait_for_next_hour_polling): {t_sync_end - t_sync_start:.3f} s")
            
            # Usa la versión rápida para obtener el listado actualizado
            t_fetch_start = time.time()
            current_symbols = set(fetch_current_symbols_fast())
            t_fetch_end = time.time()
            logging.info(f"Tiempo de fetch_current_symbols: {t_fetch_end - t_fetch_start:.3f} s")
            
            # Comparar para detectar nuevos símbolos
            t_compare_start = time.time()
            new_symbols = get_new_symbols(previous_symbols, current_symbols)
            t_compare_end = time.time()
            logging.info(f"Tiempo de get_new_symbols: {t_compare_end - t_compare_start:.3f} s")
            
            if new_symbols:
                logging.info(f"Nuevas monedas detectadas: {new_symbols}")
                for symbol in new_symbols:
                    threading.Thread(target=execute_trade, args=(symbol,), daemon=True).start()
            else:
                logging.info("No se detectaron nuevas monedas en esta iteración.")

            previous_symbols = current_symbols
        except Exception as e:
            logging.error(f"Error en el loop principal: {e}")
            time.sleep(30)  # Espera antes de reintentar

if __name__ == "__main__":
    main()
