import ccxt
import pandas as pd
from openai import OpenAI
import time
from elegir_cripto import choose_best_cryptos  # Importar la función de selección de criptos
from dotenv import load_dotenv
import os

# Configurar APIs de OpenAI y CCXT
exchange = ccxt.binanceus({
    "apiKey": os.getenv("BINANCE_API_KEY_TESTNET"),
    "secret": os.getenv("BINANCE_SECRET_KEY_TESTNET"),
    "enableRateLimit": True
})

exchange.set_sandbox_mode(True)

if os.getenv("HEROKU") is None:
    load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada")

client = OpenAI(api_key=OPENAI_API_KEY)

TRADE_SIZE = 100  # Tamaño de cada operación en USD
TRANSACTION_LOG = []

# Obtener balance inicial del sandbox
def get_sandbox_balance():
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['free']['USDT']  # Balance disponible en USDT
        print(f"Balance inicial del sandbox: {usdt_balance} USDT")
        return usdt_balance
    except Exception as e:
        print(f"Error al obtener el balance del sandbox: {e}")
        return 0  # Fallback a 0 en caso de error

# Obtener portafolio inicial del sandbox
def get_sandbox_portfolio():
    try:
        balance = exchange.fetch_balance()
        portfolio = {asset: details['free'] for asset, details in balance['total'].items() if details > 0}
        print(f"Portafolio actual en el sandbox: {portfolio}")
        return portfolio
    except Exception as e:
        print(f"Error al obtener el portafolio del sandbox: {e}")
        return {}

# Ejecutar orden de compra
def execute_order_buy(symbol, amount):
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        print(f"Orden de compra ejecutada: {order}")
        return order
    except Exception as e:
        print(f"Error al ejecutar la orden de compra para {symbol}: {e}")
        return None

# Ejecutar orden de venta
def execute_order_sell(symbol, amount):
    try:
        order = exchange.create_market_sell_order(symbol, amount)
        print(f"Orden de venta ejecutada: {order}")
        return order
    except Exception as e:
        print(f"Error al ejecutar la orden de venta para {symbol}: {e}")
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

# Obtener y preparar datos
def fetch_and_prepare_data(symbol):
    try:
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=50)
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df_1h
    except Exception as e:
        print(f"Error al obtener datos para {symbol}: {e}")
        return None

# Función principal
def demo_trading():
    print("Iniciando demo de inversión en el sandbox...")

    portfolio = get_sandbox_portfolio()
    print(f"Portafolio inicial: {portfolio}")

    selected_cryptos = choose_best_cryptos()
    print(f"Criptos seleccionadas para trading: {selected_cryptos}")

    for symbol in selected_cryptos:
        try:
            df = fetch_and_prepare_data(symbol)
            if df is None or df.empty:
                print(f"Datos insuficientes para {symbol}.")
                continue

            current_price = df['close'].iloc[-1]
            action, explanation = gpt_decision(df)

            if action == "comprar":
                # Comprar basado en balance disponible
                usdt_balance = get_sandbox_balance()
                trade_amount = TRADE_SIZE / current_price
                if usdt_balance >= TRADE_SIZE:
                    execute_order_buy(symbol, trade_amount)
                    portfolio = get_sandbox_portfolio()  # Actualiza portafolio
                else:
                    print(f"Saldo insuficiente para comprar {symbol}. Saldo disponible: {usdt_balance} USDT.")

            elif action == "vender":
                if symbol in portfolio and portfolio[symbol] > 0:
                    execute_order_sell(symbol, portfolio[symbol])
                    portfolio = get_sandbox_portfolio()  # Actualiza portafolio
                else:
                    print(f"No tienes suficiente {symbol} para vender en el sandbox.")

            else:
                print(f"No se realiza ninguna acción para {symbol} (mantener).")

            time.sleep(1)

        except Exception as e:
            print(f"Error procesando {symbol}: {e}")
            continue

    print("\n--- Resultados finales en el sandbox ---")
    print(f"Portafolio final: {get_sandbox_portfolio()}")
    print(f"Balance final: {get_sandbox_balance()} USDT")

# Ejecutar demo
if __name__ == "__main__":
    demo_trading()
