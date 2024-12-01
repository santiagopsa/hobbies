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
    devolviendo la acción y la explicación por separado.
    """
    datos = data.tail(10).to_string(index=False)
    prompt = f"""
    Eres un experto en trading. Basándote en los siguientes datos de mercado, decide si comprar, vender o mantener.
    Proporciona una breve explicación de tu decisión.

    Datos:
    {datos}

    Inicia tu respuesta con: "comprar", "vender" o "mantener" seguido de la explicación.
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

    if message.lower().startswith('comprar'):
        return "comprar", message[7:].strip()
    elif message.lower().startswith('vender'):
        return "vender", message[6:].strip()
    elif message.lower().startswith('mantener'):
        return "mantener", message[8:].strip()
    else:
        return "mantener", "No hay una recomendación clara."

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


# Función principal
def demo_trading():
    print("Iniciando demo de inversión...")

    # Analizar portafolio para venta
    portfolio_cryptos = get_portfolio_cryptos()
    print(f"Criptos en portafolio para analizar venta: {portfolio_cryptos}")

    for symbol in portfolio_cryptos:
        try:
            # Asegurarse de usar el formato correcto de símbolo (por ejemplo, BTC/USDT)
            market_symbol = f"{symbol}/USDT"

            df = fetch_and_prepare_data(market_symbol)
            if df is None or df.empty:
                print(f"⚠️ Datos insuficientes para {market_symbol}.")
                continue

            current_price = df['close'].iloc[-1]
            action, explanation = gpt_decision(df)

            crypto_balance = exchange.fetch_balance()['free'].get(symbol, 0)
            if crypto_balance > 0:
                execute_order_sell(market_symbol, crypto_balance)
            else:
                print(f"⚠️ No tienes suficiente {symbol} para vender.")

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
