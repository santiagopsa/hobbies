import ccxt
import pandas as pd
from openai import OpenAI
import time
from elegir_cripto import choose_best_cryptos  # Importar la función de selección de criptos
from dotenv import load_dotenv
import os
import csv
from db_manager_real import initialize_db, insert_transaction, fetch_all_transactions, upgrade_db_schema

initialize_db()
#upgrade_db_schema()


if os.getenv("HEROKU") is None:
    load_dotenv()

# Configurar APIs de OpenAI y CCXT
exchange = ccxt.binance({
    "enableRateLimit": True
})

exchange.apiKey = os.getenv("BINANCE_API_KEY_REAL")
exchange.secret = os.getenv("BINANCE_SECRET_KEY_REAL")


# exchange.set_sandbox_mode(True)

# Verificar conexión
try:
    print("Conectando a Binance REAL...")
    balance = exchange.fetch_balance()
    print("Conexión exitosa. Balance:", balance)
except Exception as e:
    print("Error al conectar con Binance Testnet:", e)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada")

client = OpenAI(api_key=OPENAI_API_KEY)

# Variables globales
TRADE_SIZE = 10
TRANSACTION_LOG = []

def get_portfolio_cryptos():
    """
    Obtiene las criptos del portafolio con saldo mayor a 0.
    """
    try:
        balance = exchange.fetch_balance()
        portfolio = balance['free']
        active_cryptos = [
            symbol for symbol, amount in portfolio.items() if amount > 0 and symbol != 'USDT'
        ]
        return active_cryptos
    except Exception as e:
        print(f"❌ Error al obtener el portafolio: {e}")
        return []

def wei_to_bnb(value_in_wei):
    """
    Convierte valores en wei a la unidad principal (BNB o similar).
    """
    return value_in_wei / (10 ** 18)

def log_transaction(order):
    """
    Registra una orden en un archivo CSV con las columnas: símbolo, precio, cantidad ejecutada.
    """
    filename = "ordenes_realizadas.csv"
    fields = ["symbol", "price", "amount"]  # Columnas del archivo

    try:
        # Extraer datos relevantes, adaptando el acceso a los datos según el formato del objeto `order`
        symbol = order.get("symbol", "UNKNOWN") if isinstance(order, dict) else order.symbol
        price = order.get("price", 0) if isinstance(order, dict) else order.price
        amount = order.get("filled", 0) if isinstance(order, dict) else order.filled

        # Escribir en el archivo CSV
        file_exists = os.path.isfile(filename)
        with open(filename, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not file_exists:
                writer.writeheader()  # Escribir encabezado
            writer.writerow({"symbol": symbol, "price": price, "amount": amount})
        print(f"✅ Orden registrada: {symbol}, Precio: {price}, Cantidad: {amount}")
    except AttributeError as e:
        print(f"❌ Error al acceder a los atributos de la orden: {e}")
    except Exception as e:
        print(f"❌ Error al registrar la orden en el archivo: {e}")
    

def fetch_and_prepare_data(symbol):
    """
    Obtiene datos históricos en múltiples marcos temporales y calcula indicadores técnicos.
    """
    try:
        # Definir marcos temporales y cantidad de datos a obtener
        timeframes = ['1h', '4h', '1d']  # Horas, 4 horas, días
        data = {}

        # Obtener datos para cada marco temporal
        for timeframe in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Calcular indicadores técnicos
            df['MA_20'] = df['close'].rolling(window=20).mean()  # Media móvil de 20 periodos
            df['MA_50'] = df['close'].rolling(window=50).mean()  # Media móvil de 50 periodos
            df['RSI'] = calculate_rsi(df['close'])  # Índice de Fuerza Relativa
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['close'])  # Bandas de Bollinger

            data[timeframe] = df  # Almacenar datos por marco temporal

        return data  # Devuelve un diccionario con los datos de cada marco temporal

    except Exception as e:
        print(f"Error al obtener datos para {symbol}: {e}")
        return None

# Función para calcular el RSI
def calculate_rsi(series, period=14):
    """
    Calcula el Índice de Fuerza Relativa (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Función para calcular Bandas de Bollinger
def calculate_bollinger_bands(series, window=20, num_std_dev=2):
    """
    Calcula las Bandas de Bollinger.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, rolling_mean, lower_band

# Decisión de trading con GPT
def gpt_prepare_data(data_by_timeframe):
    """
    Usa GPT-3.5 Turbo para preparar el texto estructurado basado en los datos de mercado.
    """
    combined_data = ""
    for timeframe, df in data_by_timeframe.items():
        if df is not None and not df.empty:
            combined_data += f"\nDatos de {timeframe}:\n"
            combined_data += df.tail(5).to_string(index=False)

    prompt = f"""
    Basándote en los datos de mercado en múltiples marcos temporales, organiza esta información en un texto estructurado.
    El texto debe incluir los indicadores clave (RSI, Bandas de Bollinger, Medias Móviles) y patrones identificados.

    Datos:
    {combined_data}

    Proporciona una salida que pueda ser procesada por GPT-4 Turbo para decidir comprar, vender o mantener.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en análisis de datos financieros."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()


def gpt_decision(prepared_text):
    """
    Usa GPT-4 Turbo para analizar el texto preparado y decidir si comprar, vender o mantener.
    """
    prompt = f"""
    Eres un experto en trading. Basándote en el siguiente texto estructurado, decide si comprar, vender o mantener.

    Texto:
    {prepared_text}

    Inicia tu respuesta con: "comprar", "vender" o "mantener". Incluye un resumen de la decisión y un porcentaje de confianza.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en trading."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    message = response.choices[0].message.content.strip()

    if message.lower().startswith('comprar'):
        action = "comprar"
        confidence = extract_confidence(message)
        explanation = message[7:].strip()
    elif message.lower().startswith('vender'):
        action = "vender"
        confidence = extract_confidence(message)
        explanation = message[6:].strip()
    elif message.lower().startswith('mantener'):
        action = "mantener"
        confidence = extract_confidence(message)
        explanation = message[8:].strip()
    else:
        action = "mantener"
        confidence = 50  # Valor predeterminado
        explanation = "No hay una recomendación clara."

    return action, confidence, explanation


def extract_confidence(message):
    """
    Extrae el porcentaje de confianza de la respuesta.
    """
    import re
    match = re.search(r'(\d+)%', message)
    if match:
        return int(match.group(1))
    return 50  # Valor predeterminado si no se encuentra un porcentaje

def is_valid_notional(symbol, amount):
    """
    Verifica si el valor notional cumple con los requisitos mínimos de Binance.
    """
    try:
        # Cargar los datos del mercado para el símbolo
        markets = exchange.load_markets()
        market = markets.get(symbol)

        # Obtener el precio actual del símbolo
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']  # Último precio del par

        # Calcular el valor notional
        notional = current_price * amount

        # Obtener el valor mínimo de notional desde el mercado
        if market and 'limits' in market and 'cost' in market['limits']:
            min_notional = market['limits']['cost']['min']
        else:
            min_notional = 10  # Valor mínimo genérico si no está definido

        print(f"🔍 Verificación para {symbol}:")
        print(f"    Precio actual: {current_price} USDT")
        print(f"    Cantidad: {amount}")
        print(f"    Valor notional: {notional} USDT")
        print(f"    Mínimo permitido: {min_notional} USDT")

        # Validar si el notional cumple con el requisito
        is_valid = notional >= min_notional
        if not is_valid:
            print(f"⚠️ El valor notional para {symbol} es {notional:.2f} USDT, menor al mínimo permitido de {min_notional:.2f} USDT.")
        return is_valid
    except Exception as e:
        print(f"❌ Error al verificar el valor notional para {symbol}: {e}")
        return False


# Ejecutar orden de compra
def execute_order_buy(symbol, amount, confidence, explanation):
    """
    Ejecuta una orden de compra y registra la transacción en la base de datos.
    """
    try:
        # Validar el notional antes de ejecutar la orden
        if not is_valid_notional(symbol, amount):
            print(f"⚠️ Orden de compra para {symbol} no válida debido al valor notional.")
            return None

        # Ejecutar la orden de compra
        order = exchange.create_market_buy_order(symbol, amount)
        price = order["price"] or 0
        timestamp = pd.Timestamp.now().isoformat()

        # Registrar en la base de datos
        insert_transaction(
            symbol=symbol,
            action="buy",
            price=price,
            amount=amount,
            timestamp=timestamp,
            profit_loss=None,
            confidence_percentage=confidence,
            summary=explanation
        )

        print(f"✅ Orden de compra ejecutada: {symbol}, Precio: {price}, Cantidad: {amount}")
        return order
    except Exception as e:
        print(f"❌ Error al ejecutar la orden de compra para {symbol}: {e}")
        return None

def execute_order_sell(symbol, confidence, explanation):
    """
    Ejecuta una orden de venta y registra la transacción en la base de datos.
    """
    try:
        # Obtener el saldo disponible
        balance = exchange.fetch_balance()
        amount = balance['free'].get(symbol.split('/')[0], 0)

        if amount <= 0:
            print(f"⚠️ No tienes suficiente saldo para vender {symbol}.")
            return None

        # Validar el notional
        if not is_valid_notional(symbol, amount):
            print(f"⚠️ Orden de venta para {symbol} no válida debido al valor notional mínimo.")
            return None

        # Ejecutar la orden de venta
        order = exchange.create_market_sell_order(symbol, amount)

        # Obtener el precio actual si no está en la orden
        ticker = exchange.fetch_ticker(symbol)
        price = order.get("price") or ticker["last"]
        timestamp = pd.Timestamp.now().isoformat()

        # Recuperar todas las compras del símbolo
        transactions = fetch_all_transactions()
        buys = [t for t in transactions if t[1] == symbol and t[2] == "buy"]

        # Calcular el precio promedio de compra
        if buys:
            total_cost = sum(buy[3] * buy[4] for buy in buys)  # Precio * Cantidad
            total_amount = sum(buy[4] for buy in buys)  # Suma de cantidades
            average_price = total_cost / total_amount
        else:
            average_price = 0

        # Calcular ganancia/pérdida
        profit_loss = (price - average_price) * amount

        # Registrar en la base de datos
        insert_transaction(
            symbol=symbol,
            action="sell",
            price=price,
            amount=amount,
            timestamp=timestamp,
            profit_loss=profit_loss,
            confidence_percentage=confidence,
            summary=explanation
        )

        print(f"✅ Orden de venta ejecutada: {symbol}, Precio: {price}, Cantidad: {amount}, Ganancia/Pérdida: {profit_loss}")
        return order
    except Exception as e:
        print(f"❌ Error al ejecutar la orden de venta para {symbol}: {e}")
        return None


def show_transactions():
    transactions = fetch_all_transactions()
    print("Historial de transacciones:")
    for t in transactions:
        print(f"ID: {t[0]}, Symbol: {t[1]}, Action: {t[2]}, Price: {t[3]}, Amount: {t[4]}, Timestamp: {t[5]}, Profit/Loss: {t[6]}, Confidence %: {t[7]}, Summary: {t[8]}")

def calculate_trade_amount(symbol, current_price, confidence, trade_size, min_notional):
    """
    Calcula la cantidad a negociar basada en el precio actual, la confianza, el tamaño máximo permitido y el notional mínimo.

    Args:
        symbol (str): El par de criptomonedas (e.g., BTC/USDT).
        current_price (float): El precio actual de la criptomoneda.
        confidence (float): El porcentaje de confianza de la decisión.
        trade_size (float): El tamaño máximo del trade en USD.
        min_notional (float): El valor notional mínimo permitido por el exchange.

    Returns:
        float: La cantidad de criptomoneda a negociar.
    """
    # Calcular el valor notional basado en la confianza
    desired_notional = trade_size * (confidence / 100)

    # Ajustar el notional si está por debajo del mínimo permitido
    if desired_notional < min_notional and confidence > 80:
        print(f"⚠️ Ajustando el trade a cumplir con el notional mínimo para {symbol}.")
        desired_notional = min_notional

    # Calcular la cantidad en criptomoneda basada en el notional final
    trade_amount = desired_notional / current_price

    # Garantizar que la cantidad esté dentro de los límites
    max_trade_amount = trade_size / current_price
    trade_amount = min(trade_amount, max_trade_amount)

    return trade_amount


# Función principal
def demo_trading():
    print("Iniciando demo de inversión...")

    # Elegir las mejores criptos para compra
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=10)
    print(f"Criptos seleccionadas para compra: {selected_cryptos}")

    # Realizar análisis para compra
    for symbol in selected_cryptos:
        try:
            # Obtener datos históricos de múltiples marcos temporales
            data_by_timeframe = fetch_and_prepare_data(symbol)

            if not data_by_timeframe:
                print(f"⚠️ Datos insuficientes para {symbol}.")
                continue

            # Primera etapa: preparar texto con GPT-3.5 Turbo
            prepared_text = gpt_prepare_data(data_by_timeframe)

            # Segunda etapa: análisis final con GPT-4 Turbo
            action, confidence, explanation = gpt_decision(prepared_text)

            # Dentro de demo_trading(), durante una decisión de compra:
            if action == "comprar":
                usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
                current_price = data_by_timeframe["1h"]['close'].iloc[-1]
                market_info = exchange.load_markets().get(symbol)
                min_notional = market_info['limits']['cost']['min'] if market_info else 10  # Default to 10 if notional not available

                trade_amount = calculate_trade_amount(
                    symbol=symbol,
                    current_price=current_price,
                    confidence=confidence,
                    trade_size=TRADE_SIZE,
                    min_notional=min_notional
                )

                print(trade_amount)

                # Verificar si el saldo es suficiente para ejecutar la compra
                if usdt_balance >= trade_amount * current_price:
                    execute_order_buy(symbol, trade_amount, confidence, explanation)
                else:
                    print(f"⚠️ Saldo insuficiente para comprar {symbol}. Saldo disponible: {usdt_balance} USDT.")

            else:
                print(f"↔️ No se realiza ninguna acción para {symbol} (mantener).")

            time.sleep(1)

        except Exception as e:
            print(f"Error procesando {symbol}: {e}")
            continue

    # Analizar portafolio para venta
    portfolio_cryptos = get_portfolio_cryptos()
    print(f"Criptos en portafolio para analizar venta: {portfolio_cryptos}")

    for symbol in portfolio_cryptos:
        try:
            # Asegurarse de usar el formato correcto de símbolo (por ejemplo, BTC/USDT)
            market_symbol = f"{symbol}/USDT"

            # Obtener datos históricos de múltiples marcos temporales
            data_by_timeframe = fetch_and_prepare_data(market_symbol)

            if not data_by_timeframe:
                print(f"⚠️ Datos insuficientes para {market_symbol}.")
                continue

            # Primera etapa: preparar texto con GPT-3.5 Turbo
            prepared_text = gpt_prepare_data(data_by_timeframe)

            # Segunda etapa: análisis final con GPT-4 Turbo
            action, confidence, explanation = gpt_decision(prepared_text)

            if action == "vender":
                crypto_balance = exchange.fetch_balance()['free'].get(symbol, 0)
                if crypto_balance > 0:
                    execute_order_sell(market_symbol, confidence, explanation)
                else:
                    print(f"⚠️ No tienes suficiente {symbol} para vender.")
            else:
                print(f"↔️ No se realiza ninguna acción para {market_symbol} (mantener).")

            time.sleep(1)

        except Exception as e:
            print(f"Error procesando {symbol}: {e}")
            continue

    print("\n--- Resultados finales ---")
    print(f"Portafolio final: {exchange.fetch_balance()['free']}")

# Ejecutar demo
if __name__ == "__main__":
    demo_trading()
    show_transactions()
