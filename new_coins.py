import ccxt
import time
import os
import json
import logging
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
import requests

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    filename='trading_bot.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configurar API de Binance
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY_REAL'),
    'secret': os.getenv('BINANCE_SECRET_KEY_REAL'),
    'enableRateLimit': True
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

def initialize_db(db_name="trading_bot.db"):
    """
    Inicializa la base de datos SQLite para registrar transacciones.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions_new_coins (
            symbol TEXT,
            action TEXT,
            price REAL,
            amount REAL,
            timestamp TEXT,
            order_id TEXT,
            stop_order_id TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_transaction(symbol, action, price, amount, timestamp, order_id=None, stop_order_id=None, db_name="trading_bot.db"):
    """
    Inserta una transacción en la base de datos.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions_new_coins (symbol, action, price, amount, timestamp, order_id, stop_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, action, price, amount, timestamp, order_id, stop_order_id))
        conn.commit()
        conn.close()
        logging.info(f"Transacción registrada: {action} {symbol} a {price} por {amount}")
    except Exception as e:
        logging.error(f"Error al insertar transacción: {e}")

def fetch_balance():
    """
    Obtiene el balance disponible en USDT.
    """
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['free'].get('USDT', 0)
        return usdt_balance
    except Exception as e:
        logging.error(f"Error al obtener balance: {e}")
        return 0

def fetch_current_symbols():
    """
    Obtiene la lista actual de símbolos disponibles en Binance.
    """
    try:
        markets = exchange.load_markets()
        symbols = list(markets.keys())
        return symbols
    except Exception as e:
        logging.error(f"Error al cargar mercados: {e}")
        return []

def get_new_symbols(previous_symbols, current_symbols):
    """
    Identifica los símbolos que son nuevos en la lista actual.
    """
    new_symbols = [symbol for symbol in current_symbols if symbol not in previous_symbols and symbol.endswith('/USDT')]
    return new_symbols

def buy_symbol(symbol, budget=5):
    """
    Realiza una orden de compra de mercado para el símbolo especificado con el presupuesto dado.
    """
    try:
        order = exchange.create_market_buy_order(symbol, budget)
        price = order['price']
        amount = order['filled']
        order_id = order['id']
        timestamp = datetime.utcnow().isoformat()
        insert_transaction(symbol, 'buy', price, amount, timestamp, order_id)
        logging.info(f"Compra realizada: {symbol} - Precio: {price} - Cantidad: {amount}")
        send_telegram_message(f"✅ *Compra realizada*\nSímbolo: `{symbol}`\nPrecio: `{price} USDT`\nCantidad: `{amount}`")
        return order
    except Exception as e:
        logging.error(f"Error al comprar {symbol}: {e}")
        send_telegram_message(f"❌ *Error al comprar* `{symbol}`\nDetalles: {e}")
        return None

def sell_symbol(symbol, amount, target_price, stop_price):
    """
    Coloca una orden de venta limitada y un trailing stop para el símbolo especificado.
    """
    try:
        # Orden de venta limitada a 3x el precio de compra
        sell_order = exchange.create_limit_sell_order(symbol, amount, target_price)
        sell_order_id = sell_order['id']
        logging.info(f"Orden de venta limitada colocada para {symbol} a {target_price} USDT")
        send_telegram_message(f"📈 *Orden de venta limitada colocada*\nSímbolo: `{symbol}`\nPrecio objetivo: `{target_price} USDT`")

        # Configurar trailing stop cuando el precio alcance 2x
        # Nota: CCXT no soporta trailing stops directamente, se debe manejar manualmente
        # Aquí se colocará una orden de venta stop a un 15% por debajo del precio actual
        # Esta orden se actualizará a medida que el precio suba
        return sell_order_id
    except Exception as e:
        logging.error(f"Error al colocar orden de venta para {symbol}: {e}")
        send_telegram_message(f"❌ *Error al colocar orden de venta* `{symbol}`\nDetalles: {e}")
        return None

def set_trailing_stop(symbol, amount, purchase_price, trailing_percent=15):
    """
    Configura un trailing stop para el símbolo especificado.
    """
    try:
        # Calcular el precio de stop inicial al alcanzar 2x el precio de compra
        target_price = purchase_price * 3
        stop_activation_price = purchase_price * 2
        current_price = fetch_price(symbol)
        if current_price is None:
            logging.error(f"No se pudo obtener el precio actual para {symbol}.")
            return

        if current_price >= stop_activation_price:
            # Calcular el stop price con el margen del 15%
            stop_price = current_price * (1 - trailing_percent / 100)
            # Colocar una orden de venta stop
            stop_order = exchange.create_order(symbol, 'stop', 'sell', amount, stop_price)
            stop_order_id = stop_order['id']
            timestamp = datetime.utcnow().isoformat()
            insert_transaction(symbol, 'stop_sell', stop_price, amount, timestamp, stop_order_id)
            logging.info(f"Trailing stop colocado para {symbol} a {stop_price} USDT")
            send_telegram_message(f"🔄 *Trailing Stop colocado*\nSímbolo: `{symbol}`\nPrecio de stop: `{stop_price} USDT`")
    except Exception as e:
        logging.error(f"Error al configurar trailing stop para {symbol}: {e}")
        send_telegram_message(f"❌ *Error al configurar trailing stop* `{symbol}`\nDetalles: {e}")

def fetch_price(symbol):
    """
    Obtiene el precio actual de un par de criptomonedas en USDT.
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error al obtener el precio para {symbol}: {e}")
        return None

def monitor_trailing_stops():
    """
    Monitorea los trailing stops y los actualiza si el precio sube.
    """
    try:
        conn = sqlite3.connect("trading_bot.db")
        cursor = conn.cursor()
        # Obtener todas las órdenes de compra para las que no se ha configurado un trailing stop
        cursor.execute('''
            SELECT symbol, price, amount FROM transactions
            WHERE action = 'buy' AND id NOT IN (
                SELECT id FROM transactions WHERE action = 'stop_sell'
            )
        ''')
        buys = cursor.fetchall()
        conn.close()

        for buy in buys:
            symbol, purchase_price, amount = buy
            current_price = fetch_price(symbol)
            if current_price is None:
                continue

            # Verificar si el precio ha alcanzado 2x el precio de compra
            if current_price >= purchase_price * 2:
                set_trailing_stop(symbol, amount, purchase_price)
    except Exception as e:
        logging.error(f"Error en monitor_trailing_stops: {e}")

def main():
    initialize_db()
    logging.info("Iniciando bot de trading.")

    previous_symbols = set(fetch_current_symbols())
    logging.info(f"Símbolos iniciales cargados: {len(previous_symbols)}")

    while True:
        try:
            current_symbols = set(fetch_current_symbols())
            new_symbols = get_new_symbols(previous_symbols, current_symbols)
            #if not new_symbols:
            #    logging.info("No hay nuevas monedas para comprar.")
            #    continue
            for symbol in new_symbols:
                logging.info(f"Nueva moneda detectada: {symbol}")
                send_telegram_message(f"🚀 *Nueva moneda detectada*: `{symbol}`")
                send_telegram_message(f"🚀 *Precio actual*: `{fetch_price(symbol)}`")
                # Realizar compra con 5 USDT
               # order = buy_symbol(symbol, budget=5)
               # if order:
               #     purchase_price = order['price']
               #     amount = order['filled']
               #     # Colocar orden de venta a 3x el precio de compra
               #     target_price = purchase_price * 3
               #     sell_order_id = sell_symbol(symbol, amount, target_price, stop_price=None)
               #     # Se configurará el trailing stop cuando el precio alcance 2x
            # Actualizar la lista de símbolos previos
            previous_symbols = current_symbols

            # Monitorear y configurar trailing stops
            #monitor_trailing_stops()

            # Esperar antes de la siguiente verificación (por ejemplo, 60 segundos)
            time.sleep(60)
        except KeyboardInterrupt:
            logging.info("Bot de trading detenido manualmente.")
            break
        except Exception as e:
            logging.error(f"Error en el loop principal: {e}")
            time.sleep(60)  # Esperar antes de reintentar

if __name__ == "__main__":
    main()
