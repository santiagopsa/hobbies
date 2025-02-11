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
    level=logging.INFO,  # Puedes cambiar a DEBUG para más detalle
    filename='trading_bot.log',
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

# Nombre de la base de datos y límite de compras diarias
DB_NAME = "trading_bot.db"
MAX_DAILY_PURCHASES = 3

def initialize_db():
    """
    Inicializa la base de datos SQLite para registrar transacciones y contar compras diarias.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Tabla para registrar transacciones (compra/venta)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions_new_coins (
            symbol TEXT,
            action TEXT,
            price REAL,
            amount REAL,
            timestamp TEXT
        )
    ''')
    # Tabla para llevar el control de compras diarias
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

def fetch_current_symbols():
    """
    Obtiene la lista actual de símbolos disponibles en Binance que terminan en /USDT.
    """
    try:
        markets = exchange.load_markets(True)  # Forzar la recarga de mercados
        symbols = [symbol.upper() for symbol in markets.keys() if symbol.endswith('/USDT')]
        return list(set(symbols))  # Eliminar duplicados
    except Exception as e:
        logging.error(f"Error al cargar mercados: {e}")
        return []

def get_new_symbols(previous_symbols, current_symbols):
    """
    Compara las listas para detectar nuevas monedas.
    """
    return list(set(current_symbols) - set(previous_symbols))

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
    siempre y cuando no se supere el límite diario.
    """
    if get_daily_purchases() >= MAX_DAILY_PURCHASES:
        logging.info(f"⚠️ Límite de compras diarias alcanzado ({MAX_DAILY_PURCHASES}/día).")
        send_telegram_message(f"⚠️ *Límite de compras alcanzado*: No se comprará `{symbol}` hoy.")
        return None

    try:
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        budget = 10  # Presupuesto en USDT para la compra (ajusta este valor según prefieras)
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
    Ejecuta una venta inmediata (orden de mercado) y registra la transacción.
    """
    try:
        order = exchange.create_market_sell_order(symbol, amount)
        order_price = order.get('average', order.get('price', None))
        timestamp = datetime.now(timezone.utc).isoformat()
        insert_transaction(symbol, 'sell', order_price, amount, timestamp)
        send_telegram_message(f"✅ *Venta ejecutada*\nSímbolo: `{symbol}`\nPrecio: `{order_price} USDT`\nCantidad: `{amount}`")
        logging.info(f"Venta ejecutada: {symbol} a {order_price} USDT, cantidad: {amount}")
        return order
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")
        send_telegram_message(f"❌ *Error al vender* `{symbol}`\nDetalles: {e}")
        return None

def set_trailing_stop(symbol, amount, purchase_price, trailing_percent=5):
    """
    Monitorea el precio del símbolo y, una vez activado el trailing stop,
    vende inmediatamente si el precio cae por debajo del nivel calculado.

    Lógica:
      - Se espera hasta que el precio supere el precio de compra (activación).
      - Luego se guarda el precio máximo alcanzado.
      - Si el precio actual cae por debajo de (máximo * (1 - trailing_percent/100)),
        se ejecuta una venta inmediata.
    """
    try:
        logging.info(f"Configurando trailing stop para {symbol} con trailing del {trailing_percent}%")
        send_telegram_message(f"🔄 *Trailing Stop configurado* para `{symbol}`")
        highest_price = purchase_price

        # Esperar a que el precio suba para activar el trailing stop
        while True:
            current_price = fetch_price(symbol)
            if current_price is None:
                time.sleep(5)
                continue
            if current_price > purchase_price:
                highest_price = current_price
                break
            time.sleep(5)

        # Monitorear el precio y actualizar el máximo alcanzado
        while True:
            current_price = fetch_price(symbol)
            if current_price is None:
                time.sleep(5)
                continue

            if current_price > highest_price:
                highest_price = current_price
                logging.info(f"{symbol}: Nuevo máximo alcanzado: {highest_price} USDT")
                send_telegram_message(f"📈 *Nuevo máximo* para `{symbol}`: {highest_price} USDT")

            # Calcular el precio de stop
            stop_price = highest_price * (1 - trailing_percent / 100)
            if current_price < stop_price:
                logging.info(f"{symbol}: Precio {current_price} USDT cayó por debajo del trailing stop ({stop_price} USDT). Ejecutando venta inmediata.")
                send_telegram_message(f"🔴 *Trailing Stop activado* para `{symbol}`. Ejecutando venta inmediata.")
                sell_symbol(symbol, amount)
                break

            time.sleep(5)
    except Exception as e:
        logging.error(f"Error en trailing stop para {symbol}: {e}")
        send_telegram_message(f"❌ *Error en trailing stop* `{symbol}`\nDetalles: {e}")

def process_order(symbol, order_details):
    """
    Una vez realizada la compra, este método espera unos segundos y lanza el trailing stop en un hilo separado.
    """
    if order_details:
        purchase_price = order_details.get('price')
        amount = order_details.get('filled')
        if purchase_price is None or amount is None or float(amount) <= 0:
            logging.error(f"Datos insuficientes para configurar el trailing stop en {symbol}.")
            send_telegram_message(f"❌ *Error*: Datos insuficientes para trailing stop en `{symbol}`.")
            return
        # Pequeña pausa para que se actualicen balances, etc.
        time.sleep(5)
        threading.Thread(target=set_trailing_stop, args=(symbol, amount, purchase_price, 5), daemon=True).start()

def wait_for_next_hour():
    """
    Calcula y espera el tiempo restante hasta 5 segundos después de la siguiente hora en punto.
    """
    now = datetime.now()
    # Se configura la siguiente verificación a la hora en punto más 5 segundos
    next_hour = now.replace(minute=0, second=5, microsecond=0) + timedelta(hours=1)
    wait_time = (next_hour - now).total_seconds()
    logging.info(f"Esperando {wait_time:.2f} segundos hasta la siguiente verificación...")
    time.sleep(wait_time)

def main():
    initialize_db()
    logging.info("Iniciando bot de trading de nuevas monedas.")
    previous_symbols = set(fetch_current_symbols())
    logging.info(f"Símbolos iniciales cargados: {len(previous_symbols)}")
    
    while True:
        try:
            # Espera hasta 5 segundos después de la hora en punto
            wait_for_next_hour()

            current_symbols = set(fetch_current_symbols())
            new_symbols = get_new_symbols(previous_symbols, current_symbols)

            if new_symbols:
                logging.info(f"Nuevas monedas detectadas: {new_symbols}")
                for symbol in new_symbols:
                    send_telegram_message(f"🚀 *Nueva moneda detectada*: `{symbol}`")
                    order_details = buy_symbol(symbol)
                    if order_details:
                        process_order(symbol, order_details)
            else:
                logging.info("No se detectaron nuevas monedas en esta iteración.")

            # Actualiza la lista de símbolos para la siguiente iteración
            previous_symbols = current_symbols
        except Exception as e:
            logging.error(f"Error en el loop principal: {e}")
            time.sleep(30)  # Espera antes de reintentar

if __name__ == "__main__":
    main()
