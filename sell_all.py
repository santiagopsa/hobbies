import ccxt
import pandas as pd
from openai import OpenAI
import time
from elegir_cripto import choose_best_cryptos  # Importar la función de selección de criptos
from dotenv import load_dotenv
import os
import csv

if os.getenv("HEROKU") is None:
    load_dotenv()

# Configurar APIs de OpenAI y CCXT
exchange = ccxt.binance({
    "enableRateLimit": True
})
exchange.apiKey = os.getenv("BINANCE_API_KEY_TESTNET")
exchange.secret = os.getenv("BINANCE_SECRET_KEY_TESTNET")


exchange.set_sandbox_mode(True)

# Verificar conexión
try:
    print("Conectando a Binance Testnet...")
    balance = exchange.fetch_balance()
    print("Conexión exitosa. Balance:", balance)
except Exception as e:
    print("Error al conectar con Binance Testnet:", e)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada")

client = OpenAI(api_key=OPENAI_API_KEY)

# Variables globales
TRADE_SIZE = 100
TRANSACTION_LOG = []

# Calcular monto dinámico basado en confianza
def calculate_trade_amount(confidence):
    """
    Calcula el monto en USD basado en la confianza (entre 0 y 1).
    Retorna un valor entre 1 y 10 USD.
    """
    min_amount = 1  # Mínimo en dólares
    max_amount = 10  # Máximo en dólares
    trade_amount = min_amount + (max_amount - min_amount) * confidence
    return round(trade_amount, 2)  # Redondear a 2 decimales

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
    

# Función para obtener y procesar datos
def fetch_and_prepare_data(symbol):
    """
    Obtiene datos históricos en diferentes marcos temporales.
    """
    try:
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=50)
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df_1h
    except Exception as e:
        print(f"Error al obtener datos para {symbol}: {e}")
        return None

# Decisión de trading con GPT
def gpt_decision(data):
    """
    Utiliza GPT para analizar los datos de mercado y decidir si comprar, vender o mantener,
    devolviendo la acción, la confianza y la explicación por separado.
    """
    datos = data.tail(10).to_string(index=False)
    prompt = f"""
    Eres un experto en trading. Basándote en los siguientes datos de mercado, decide si comprar, vender o mantener.
    Proporciona una breve explicación de tu decisión y una puntuación de confianza entre 0 (muy baja) y 1 (muy alta).

    Datos:
    {datos}

    Inicia tu respuesta con: "comprar", "vender" o "mantener", seguido de la puntuación de confianza y la explicación.
    """

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en trading."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    message = response.choices[0].message.content.strip()

    # Analizar respuesta de GPT
    try:
        parts = message.split(",")
        action = parts[0].strip().lower()  # "comprar", "vender" o "mantener"
        confidence = float(parts[1].strip())  # Confianza como número entre 0 y 1
        explanation = ",".join(parts[2:]).strip()  # Explicación restante
    except Exception as e:
        print(f"⚠️ Error procesando respuesta de GPT: {e}")
        action, confidence, explanation = "mantener", 0, "Respuesta no válida."

    return action, confidence, explanation

# Ejecutar orden de compra con monto dinámico
def execute_order_buy(symbol, amount_in_usd):
    """
    Ejecuta una orden de compra utilizando el monto en USD calculado dinámicamente.
    """
    try:
        # Obtener precio actual para calcular la cantidad
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        amount = amount_in_usd / price  # Calcular cantidad de cripto a comprar

        # Ejecutar la orden
        order = exchange.create_market_buy_order(symbol, amount)
        log_transaction(order)  # Registrar la orden en el CSV
        print(f"✅ Orden de compra ejecutada: {symbol}, Monto: {amount_in_usd} USD, Cantidad: {amount:.8f}")
        return order
    except Exception as e:
        print(f"❌ Error al ejecutar la orden de compra para {symbol}: {e}")
        return None

# Ejecutar orden de compra
def execute_order_buy(symbol, amount):
    """
    Ejecuta una orden de compra y registra la transacción en un archivo CSV.
    """
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        print(f"Debug: Orden devuelta por el exchange: {order}")  # Ver contenido del objeto
        log_transaction(order)  # Registrar la orden en el CSV
        print(f"✅ Orden de compra ejecutada: {symbol}")
        return order
    except Exception as e:
        print(f"❌ Error al ejecutar la orden de compra para {symbol}: {e}")
        return None

# Ejecutar orden de venta
def execute_order_sell(symbol, amount):
    """
    Ejecuta una orden de venta y registra la transacción en un archivo CSV.
    """
    try:
        order = exchange.create_market_sell_order(symbol, amount)
        print(f"Debug: Orden devuelta por el exchange: {order}")  # Ver contenido del objeto
        log_transaction(order)  # Registrar la orden en el CSV
        print(f"✅ Orden de venta ejecutada: {symbol}")
        return order
    except Exception as e:
        print(f"❌ Error al ejecutar la orden de venta para {symbol}: {e}")
        return None



# Función principal ajustada
def demo_trading():
    print("Iniciando demo de inversión...")

    # Elegir las mejores criptos
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=10)
    print(f"Criptos seleccionadas para trading: {selected_cryptos}")

    for symbol in selected_cryptos:
        try:
            df = fetch_and_prepare_data(symbol)
            if df is None or df.empty:
                print(f"⚠️ Datos insuficientes para {symbol}.")
                continue

            current_price = df['close'].iloc[-1]
            action, confidence, explanation = gpt_decision(df)

            if action == "comprar":
                # Calcular monto dinámico en USD basado en confianza
                trade_amount_in_usd = calculate_trade_amount(confidence)

                # Verificar saldo disponible
                usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
                if usdt_balance >= trade_amount_in_usd:
                    execute_order_buy(symbol, trade_amount_in_usd)
                else:
                    print(f"⚠️ Saldo insuficiente para comprar {symbol}. Saldo disponible: {usdt_balance} USDT.")

            elif action == "vender":
                crypto_balance = exchange.fetch_balance()['free'].get(symbol.split('/')[0], 0)
                if crypto_balance > 0:
                    execute_order_sell(symbol, crypto_balance)
                else:
                    print(f"⚠️ No tienes suficiente {symbol.split('/')[0]} para vender.")

            else:
                print(f"↔️ No se realiza ninguna acción para {symbol} (mantener).")

            time.sleep(1)

        except Exception as e:
            print(f"Error procesando {symbol}: {e}")
            continue

        # Filtrar y mostrar solo los valores diferentes de cero
    balance = exchange.fetch_balance()['free']
    non_zero_balance = {currency: amount for currency, amount in balance.items() if amount != 0}

    print("\n--- Resultados finales ---")
    print("Portafolio final (solo valores no cero):")
    for currency, amount in non_zero_balance.items():
        print(f"{currency}: {amount}")

# Ejecutar demo
if __name__ == "__main__":
    demo_trading()
