import ccxt
import pandas as pd
from openai import OpenAI
import time
from elegir_cripto import choose_best_cryptos  # Importar la función de selección de criptos
from dotenv import load_dotenv
import os
from celery_app import Celery


# Configurar APIs de OpenAI y CCXT
exchange = ccxt.binanceus({
    "rateLimit": 1200,
    "enableRateLimit": True
})
# Cargar variables de entorno desde .env solo en desarrollo local
if os.getenv("HEROKU") is None:  # Usamos esta lógica para saber si estamos en Heroku
    load_dotenv()

# Obtener la API Key de OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada")

client = OpenAI(api_key=OPENAI_API_KEY)

# Variables globales
SIMULATED_BALANCE = 1000  # Balance inicial en USD para la simulación
TRADE_SIZE = 100          # Tamaño de cada operación en USD
TRANSACTION_LOG = []      # Registro de transacciones simuladas
PORTFOLIO = {}            # Portafolio de activos actuales

# Función para actualizar el portafolio
def update_portfolio(symbol, action, amount):
    global PORTFOLIO
    if action == "buy":
        if symbol in PORTFOLIO:
            PORTFOLIO[symbol] += amount
        else:
            PORTFOLIO[symbol] = amount
    elif action == "sell":
        if symbol in PORTFOLIO:
            PORTFOLIO[symbol] -= amount
            if PORTFOLIO[symbol] <= 0:
                del PORTFOLIO[symbol]
        else:
            print(f"Error: No tienes {symbol} en el portafolio para vender.")
            return False
    return True

def gpt_decision(data):

        # Dividir la respuesta de GPT entre la acción y la explicación
    action = None
    explanation = None
    """
    Utiliza GPT para analizar los datos de mercado y decidir si comprar, vender o mantener,
    devolviendo la acción y la explicación por separado, tu primera palabra de respuesta debe ser la acción.
    """
    datos = data.tail(10).to_string(index=False)
    prompt = f"""
    Eres un experto en trading. Basándote en los siguientes datos de mercado, decide si comprar, vender o mantener.
    Proporciona una breve explicación de tu decisión, tu primera palabra de respuesta debe ser la acción.

    Datos:
    {datos}

    Inicia tu respuesta con: "comprar", "vender" o "mantener" y después la explicación.
    """
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en trading."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    # Extraemos la respuesta de GPT, separamos la acción y la explicación
    message = response.choices[0].message.content.strip()

    if message.lower().startswith('comprar'):
        action = "comprar"
        explanation = message[7:].strip()  # Eliminar "comprar" y los espacios al inicio
    elif message.lower().startswith('vender'):
        action = "vender"
        explanation = message[6:].strip()  # Eliminar "vender" y los espacios al inicio
    elif message.lower().startswith('mantener'):
        action = "mantener"
        explanation = message[8:].strip()  # Eliminar "mantener" y los espacios al inicio
    else:
        action = "mantener"
        explanation = "No hay una recomendación clara."


    return action, explanation


def execute_simulated_order(symbol, action, price):
    global SIMULATED_BALANCE, TRANSACTION_LOG

    if action == "comprar":
        # Calculate how much can be bought with TRADE_SIZE
        amount = TRADE_SIZE / price
        if SIMULATED_BALANCE >= TRADE_SIZE:
            # Deduct balance and update portfolio
            SIMULATED_BALANCE -= TRADE_SIZE
            update_portfolio(symbol, "buy", amount)
            TRANSACTION_LOG.append({
                "symbol": symbol,
                "action": "buy",
                "price": price,
                "amount": amount,
                "balance": SIMULATED_BALANCE
            })
            print(f"Simulación: Comprado {amount:.6f} {symbol} a {price:.2f} USD.")
        else:
            print(f"Error: No tienes suficiente saldo para comprar {symbol}.")

    elif action == "vender":
        if symbol in PORTFOLIO and PORTFOLIO[symbol] > 0:
            # Determine the amount to sell (minimum of TRADE_SIZE / price or current balance)
            amount = min(TRADE_SIZE / price, PORTFOLIO[symbol])
            if amount > 1e-6:  # Ensure meaningful transaction
                # Increase balance and update portfolio
                SIMULATED_BALANCE += amount * price
                if update_portfolio(symbol, "sell", amount):
                    TRANSACTION_LOG.append({
                        "symbol": symbol,
                        "action": "sell",
                        "price": price,
                        "amount": amount,
                        "balance": SIMULATED_BALANCE
                    })
                    print(f"Simulación: Vendido {amount:.6f} {symbol} a {price:.2f} USD.")
                else:
                    print(f"Error: No se pudo actualizar el portafolio al vender {symbol}.")
            else:
                print(f"Error: La cantidad a vender de {symbol} es demasiado pequeña.")
        else:
            print(f"Error: No tienes suficiente {symbol} para vender.")

    else:
        print(f"Simulación: Mantener posición para {symbol}.")


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

# Función principal
@celery_app.task
def demo_trading():
    global TRANSACTION_LOG
    print("Iniciando demo de inversión...")

    selected_cryptos = choose_best_cryptos()
    print(f"Criptos seleccionadas para trading: {selected_cryptos}")

    # Ciclo de simulación
    for symbol in selected_cryptos:
        try:
            df = fetch_and_prepare_data(symbol)
            if df is None or df.empty:
                print(f"Datos insuficientes para {symbol}.")
                continue

            current_price = df['close'].iloc[-1]

            # Analizar con GPT
            action, explanation = gpt_decision(df)

            # Almacenar la transacción
            TRANSACTION_LOG.append({
                "symbol": symbol,
                "action": action,
                "price": current_price,
                "explanation": explanation
            })

            # Determinar acción a realizar
            if action == "comprar":
                execute_simulated_order(symbol, "comprar", current_price)
            elif action == "vender":
                execute_simulated_order(symbol, "vender", current_price)
            else:
                print(f"No se realiza ninguna acción para {symbol} (mantener).")

            time.sleep(1)  # Pausa entre análisis
        except Exception as e:
            print(f"Error procesando {symbol}: {e}")
            continue

    # Mostrar resultados finales
    print("\n--- Resultados de la simulación ---")
    print(f"Balance final simulado: {SIMULATED_BALANCE:.2f} USD")
    print(f"Transacciones analizadas: {len(TRANSACTION_LOG)}")
    print(f"Portafolio actual: {PORTFOLIO}")
    for log in TRANSACTION_LOG:
        print(log)


# Ejecutar demo
if __name__ == "__main__":
    demo_trading()
