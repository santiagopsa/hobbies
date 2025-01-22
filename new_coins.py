import ccxt
import time
import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timezone
import sqlite3
import requests
import threading

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
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot'  # Cambia a 'future' si operas en futuros
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
    Obtiene la lista actual de símbolos disponibles en Binance que terminan en /USDT.
    """
    try:
        markets = exchange.load_markets()
        symbols = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
        return symbols
    except Exception as e:
        logging.error(f"Error al cargar mercados: {e}")
        return []

def get_new_symbols(previous_symbols, current_symbols):
    """
    Identifica los símbolos que son nuevos en la lista actual y terminan en /USDT.
    """
    new_symbols = [symbol for symbol in current_symbols if symbol not in previous_symbols and symbol.endswith('/USDT')]
    return new_symbols

def fetch_price(symbol, exchange_instance=None):
    """
    Obtiene el precio actual de un par de criptomonedas en USDT.
    """
    if exchange_instance is None:
        exchange_instance = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY_REAL'),
            'secret': os.getenv('BINANCE_SECRET_KEY_REAL'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'  # Cambia a 'future' si operas en futuros
            }
        })
    try:
        ticker = exchange_instance.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error al obtener el precio para {symbol}: {e}")
        return None

def buy_symbol(symbol, budget=5, exchange_instance=None):
    """
    Realiza una orden de compra de mercado para el símbolo especificado con el presupuesto dado.
    
    :param symbol: Par de trading, por ejemplo, 'VTHO/USDT'
    :param budget: Presupuesto en la moneda de cotización, por ejemplo, 5 USDT
    :param exchange_instance: Instancia de CCXT para el intercambio
    :return: Diccionario con detalles de la orden si se ejecuta correctamente, None en caso contrario
    """
    if exchange_instance is None:
        logging.error("La instancia de exchange no se ha proporcionado.")
        return None
    
    try:
        # Obtener el precio actual
        ticker = exchange_instance.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Calcular la cantidad a comprar
        amount = budget / current_price
        
        # Obtener la precisión permitida para la cantidad
        market = exchange_instance.markets[symbol]
        precision = market.get('precision', {}).get('amount', 8)
        amount = exchange_instance.amount_to_precision(symbol, amount)
        
        # Calcular el valor de la orden
        order_notional = float(amount) * float(current_price)
        min_notional = market.get('limits', {}).get('cost', {}).get('min', 0)
        
        logging.info(f"Presupuesto: {budget} USDT")
        logging.info(f"Precio actual: {current_price} USDT")
        logging.info(f"Cantidad calculada: {amount} {symbol.split('/')[0]}")
        logging.info(f"Valor de la orden: {order_notional} USDT")
        
        if order_notional < min_notional:
            logging.error(f"Valor de la orden insuficiente: {order_notional} USDT es menor que el Min Notional de {min_notional} USDT")
            send_telegram_message(f"❌ *Error al comprar* `{symbol}`\nDetalles: Valor de la orden insuficiente ({order_notional} USDT < {min_notional} USDT)")
            return None
        
        # Colocar la orden de compra de mercado
        order = exchange_instance.create_market_buy_order(symbol, amount)
        price = order.get('average', order.get('price', None))
        filled = order.get('filled', 0)
        order_id = order.get('id', 'N/A')
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Registrar la transacción en tu sistema
        insert_transaction(symbol, 'buy', price, filled, timestamp, order_id)
        
        logging.info(f"Compra realizada: {symbol} - Precio: {price} - Cantidad: {filled}")
        send_telegram_message(f"✅ *Compra realizada*\nSímbolo: `{symbol}`\nPrecio: `{price} USDT`\nCantidad: `{filled}`")
        return {
            'price': price,
            'filled': filled,
            'order_id': order_id
        }
    except ccxt.InsufficientFunds as e:
        logging.error(f"Fondos insuficientes para comprar {symbol}: {e}")
        send_telegram_message(f"❌ *Error al comprar* `{symbol}`\nDetalles: Fondos insuficientes.")
    except ccxt.ExchangeError as e:
        logging.error(f"Error del intercambio al comprar {symbol}: {e}")
        send_telegram_message(f"❌ *Error al comprar* `{symbol}`\nDetalles: {e}")
    except Exception as e:
        logging.error(f"Error al comprar {symbol}: {e}")
        send_telegram_message(f"❌ *Error al comprar* `{symbol}`\nDetalles: {e}")
    return None

def sell_symbol(symbol, amount, target_price, stop_price=None, exchange_instance=None):
    """
    Coloca una orden de venta limitada para el símbolo especificado.
    
    :param symbol: Par de trading, por ejemplo, 'VTHO/USDT'
    :param amount: Cantidad de la criptomoneda a vender
    :param target_price: Precio objetivo para la orden de venta
    :param stop_price: Precio de stop para la orden (opcional)
    :param exchange_instance: Instancia de CCXT para el intercambio
    :return: ID de la orden si se ejecuta correctamente, None en caso contrario
    """
    if exchange_instance is None:
        logging.error("La instancia de exchange no se ha proporcionado.")
        return None
    
    try:
        # Obtener la precisión permitida para el precio y la cantidad
        market = exchange_instance.markets.get(symbol, None)
        if not market:
            logging.error(f"El símbolo {symbol} no está disponible en exchange.markets.")
            send_telegram_message(f"❌ *Error al vender* `{symbol}`\nDetalles: Símbolo no disponible en mercados.")
            return None
        
        price_precision = market.get('precision', {}).get('price', 8)
        amount_precision = market.get('precision', {}).get('amount', 8)
        
        # Redondear el precio y la cantidad según la precisión permitida
        target_price = exchange_instance.price_to_precision(symbol, target_price)
        amount = exchange_instance.amount_to_precision(symbol, amount)
        
        # Verificar el balance disponible
        asset = symbol.split('/')[0]
        balance = exchange_instance.fetch_balance()
        available_amount = balance['free'].get(asset, 0)
        
        # Ajustar la cantidad a vender si es necesario
        sell_amount = min(float(amount), available_amount)
        if sell_amount <= 0:
            logging.error(f"No hay suficiente {asset} para vender.")
            send_telegram_message(f"❌ *Error al vender* `{symbol}`\nDetalles: No hay suficiente `{asset}` para vender.")
            return None
        elif sell_amount < float(amount):
            logging.warning(f"Saldo insuficiente para vender la cantidad completa de {symbol}. Vendiendo {sell_amount} en lugar de {amount}.")
            send_telegram_message(f"⚠️ *Aviso de Saldo Insuficiente*\nSímbolo: `{symbol}`\nCantidad vendida: `{sell_amount}` en lugar de `{amount}`.")
        
        # Colocar la orden de venta limitada
        order = exchange_instance.create_limit_sell_order(symbol, sell_amount, target_price)
        order_id = order.get('id', 'N/A')
        
        logging.info(f"Orden de venta limitada colocada para {symbol} a {target_price} USDT - ID: {order_id}")
        send_telegram_message(f"📈 *Orden de venta limitada colocada*\nSímbolo: `{symbol}`\nPrecio objetivo: `{target_price} USDT`\nID Orden: `{order_id}`")
        return order_id
    except ccxt.ExchangeError as e:
        logging.error(f"Error del intercambio al vender {symbol}: {e}")
        send_telegram_message(f"❌ *Error al vender* `{symbol}`\nDetalles: {e}")
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")
        send_telegram_message(f"❌ *Error al vender* `{symbol}`\nDetalles: {e}")
    return None

def set_trailing_stop(symbol, amount, purchase_price, trailing_percent=20, exchange_instance=None):
    """
    Configura un trailing stop para el símbolo especificado en un hilo separado.
    
    :param symbol: Par de trading, por ejemplo, 'VTHO/USDT'
    :param amount: Cantidad de la criptomoneda
    :param purchase_price: Precio de compra de la criptomoneda
    :param trailing_percent: Porcentaje de trailing para el stop
    :param exchange_instance: Instancia de CCXT para el intercambio
    """
    if exchange_instance is None:
        logging.error("La instancia de exchange no se ha proporcionado para configurar el trailing stop.")
        return
    
    def trailing_stop_logic():
        try:
            # Calcular el precio de activación al alcanzar 1.5x el precio de compra
            activation_price = purchase_price * 1.5
            logging.info(f"Trailing stop para {symbol} se activará al alcanzar {activation_price} USDT")
            send_telegram_message(f"🔄 *Trailing Stop configurado* para `{symbol}`\nActivación al alcanzar `{activation_price} USDT`")
            
            while True:
                current_price = fetch_price(symbol, exchange_instance)
                if current_price is None:
                    logging.error(f"No se pudo obtener el precio actual para {symbol}.")
                    time.sleep(60)  # Esperar antes de volver a intentar
                    continue

                # Verificar si el precio ha alcanzado el precio de activación
                if current_price >= activation_price:
                    logging.info(f"{symbol}: Precio de activación alcanzado. Configurando trailing stop.")
                    send_telegram_message(f"📊 *Trailing Stop Activado* para `{symbol}`\nPrecio Actual: `{current_price} USDT`")
                    
                    # Inicializar el precio más alto alcanzado
                    highest_price = current_price
                    while True:
                        updated_price = fetch_price(symbol, exchange_instance)
                        if updated_price is None:
                            logging.error(f"No se pudo obtener el precio actual para {symbol}.")
                            time.sleep(60)
                            continue
                        
                        if updated_price > highest_price:
                            highest_price = updated_price
                            logging.info(f"{symbol}: Nuevo precio más alto alcanzado: {highest_price} USDT")
                            send_telegram_message(f"📈 *Nuevo Precio Más Alto Alcanzado* para `{symbol}`\nNuevo Precio: `{highest_price} USDT`")
                        
                        # Calcular el stop price con el margen del trailing_percent
                        stop_price = highest_price * (1 - trailing_percent / 100)
                        
                        logging.debug(f"{symbol}: Precio actual: {updated_price} USDT, Stop Price: {stop_price} USDT")
                        
                        # Si el precio actual cae por debajo del stop price, colocar la orden de venta
                        if updated_price < stop_price:
                            try:
                                # Binance usa 'STOP_LOSS_LIMIT' para este tipo de órdenes
                                stop_order = exchange_instance.create_order(
                                    symbol,
                                    'STOP_LOSS_LIMIT',
                                    'sell',
                                    amount,
                                    stop_price,
                                    {
                                        'stopPrice': stop_price,
                                        'price': exchange_instance.price_to_precision(symbol, stop_price * 0.99)  # Precio límite ligeramente inferior
                                    }
                                )
                                stop_order_id = stop_order.get('id', 'N/A')
                                timestamp = datetime.now(timezone.utc).isoformat()
                                insert_transaction(symbol, 'stop_sell', stop_price, amount, timestamp, stop_order_id)
                                logging.info(f"Trailing stop activado para {symbol} a {stop_price} USDT")
                                send_telegram_message(f"🔄 *Trailing Stop Activado*\nSímbolo: `{symbol}`\nPrecio de stop: `{stop_price} USDT`")
                                break  # Salir del bucle interno
                            except Exception as e:
                                logging.error(f"Error al colocar orden de trailing stop para {symbol}: {e}")
                                send_telegram_message(f"❌ *Error al configurar trailing stop* `{symbol}`\nDetalles: {e}")
                                break
                        time.sleep(60)  # Esperar antes de la siguiente verificación
                    break  # Salir del bucle externo una vez que se ha configurado el trailing stop
                time.sleep(60)  # Esperar antes de la siguiente verificación

        except Exception as e:
            logging.error(f"Error en trailing_stop_logic para {symbol}: {e}")
            send_telegram_message(f"❌ *Error en trailing_stop_logic* `{symbol}`\nDetalles: {e}")

    # Crear y empezar el hilo
    trailing_thread = threading.Thread(target=trailing_stop_logic, daemon=True)
    trailing_thread.start()

def process_order(order, symbol, exchange_instance):
    """
    Procesa la orden de compra y configura el trailing stop correspondiente.
    
    :param order: Diccionario con detalles de la orden de compra
    :param symbol: Par de trading, por ejemplo, 'VTHO/USDT'
    :param exchange_instance: Instancia de CCXT para el intercambio
    """
    if order:
        logging.info(f"Iniciando procesamiento de la orden para {symbol}.")
        purchase_price = order['price']
        amount = order['filled']
        
        if purchase_price is None or amount <= 0:
            logging.error(f"Datos insuficientes en la orden para {symbol}. Precio: {purchase_price}, Cantidad: {amount}")
            send_telegram_message(f"❌ *Error al procesar la orden de compra* `{symbol}`\nDetalles: Datos insuficientes.")
            return
        
        # Pausa para permitir la actualización del balance
        logging.info(f"Pausando por 5 segundos para permitir la actualización del balance.")
        time.sleep(5)  # Pausa de 5 segundos
        
        # Verificar el balance disponible
        asset = symbol.split('/')[0]
        logging.info(f"Verificando el balance disponible para {asset}.")
        balance = exchange_instance.fetch_balance()
        available_amount = balance['free'].get(asset, 0)
        logging.info(f"Balance disponible para {asset}: {available_amount}")
        
        # Ajustar la cantidad a manejar
        manage_amount = min(amount, available_amount)
        if manage_amount < amount:
            logging.warning(f"Saldo insuficiente para manejar la cantidad completa de {symbol}. Manejar {manage_amount} en lugar de {amount}.")
            send_telegram_message(f"⚠️ *Aviso de Saldo Insuficiente*\nSímbolo: `{symbol}`\nCantidad manejada: `{manage_amount}` en lugar de `{amount}`.")
        
        # Configurar trailing stop con precio objetivo de 3x y promedio de 1.5x
        logging.info(f"Configurando trailing stop para {symbol} con objetivo de 3x y promedio de 1.5x.")
        threading.Thread(target=set_trailing_stop, args=(symbol, manage_amount, purchase_price, 20, exchange_instance), daemon=True).start()

def main():
    initialize_db()
    logging.info("Iniciando bot de trading.")

    previous_symbols = set(fetch_current_symbols())
    logging.info(f"Símbolos iniciales cargados: {len(previous_symbols)}")
    logging.info(f"Se queda en loop hasta que encuentre nuevas monedas")

    while True:
        try:
            current_symbols = set(fetch_current_symbols())
            last_symbol = list(current_symbols)[-1] if current_symbols else None  # Manejar caso vacío
            if last_symbol:
                logging.info(f"Última moneda detectada: {last_symbol}")
            new_symbols = get_new_symbols(previous_symbols, current_symbols)

            for symbol in new_symbols:
                logging.info(f"Nueva moneda detectada: {symbol}")
                send_telegram_message(f"🚀 *Nueva moneda detectada*: `{symbol}`")
                
                current_price = fetch_price(symbol, exchange)
                if current_price is None:
                    logging.error(f"No se pudo obtener el precio actual para {symbol}.")
                    continue
                send_telegram_message(f"🚀 *Precio actual*: `{current_price} USDT`")
                
                market = exchange.markets.get(symbol, None)
                if not market:
                    logging.error(f"El símbolo {symbol} no está disponible en exchange.markets.")
                    continue
                
                min_notional = market.get('limits', {}).get('cost', {}).get('min')
                logging.info(f"Min notional: {min_notional}")
                
                order = buy_symbol(symbol, budget=5, exchange_instance=exchange)
                if order:
                    process_order(order, symbol, exchange)

            # Actualizar la lista de símbolos previos
            previous_symbols = current_symbols

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
