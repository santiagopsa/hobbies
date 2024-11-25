import ccxt
import pandas as pd
from openai import OpenAI
import time
from elegir_cripto import choose_best_cryptos  # Importar la función de selección de criptos
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Configurar APIs de OpenAI y CCXT
exchange = ccxt.binance({
    "rateLimit": 1200,
    "enableRateLimit": True
})
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

# Función para analizar con GPT
def gpt_decision(data):
    """
    Utiliza GPT para analizar los datos de mercado y decidir si comprar, vender o mantener.
    """
    prompt = f"""
    Eres un experto en trading. Basándote en los siguientes datos de mercado, decide si comprar, vender o mantener.
    Proporciona una breve explicación de tu decisión.

    Datos:
    {data.tail(10).to_string(index=False)}

    Responde con: "comprar", "vender" o "mantener" y la explicación.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en trading."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Función para simular una operación
def execute_simulated_order(symbol, action, price):
    global SIMULATED_BALANCE, TRANSACTION_LOG
    if action == "comprar":
        amount = TRADE_SIZE / price
        if SIMULATED_BALANCE >= TRADE_SIZE:
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
            amount = TRADE_SIZE / price
            if PORTFOLIO[symbol] >= amount:
                SIMULATED_BALANCE += TRADE_SIZE
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
            decision = gpt_decision(df)
            print(f"Decisión para {symbol}: {decision}")

            # Determinar acción a realizar
            action = "mantener"
            if "comprar" in decision.lower() and "mantener" not in decision.lower():
                action = "comprar"
            elif "vender" in decision.lower():
                action = "vender"

            # Ejecutar acción
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
    print(f"Transacciones realizadas: {len(TRANSACTION_LOG)}")
    print(f"Portafolio actual: {PORTFOLIO}")
    for log in TRANSACTION_LOG:
        print(log)

    # Generar gráfica de precios
    generate_price_graph(selected_cryptos)

# Función para generar gráfica de precios
def generate_price_graph(cryptos):
    """
    Genera una gráfica tipo mosaico de los precios de cierre de las criptos seleccionadas.
    """
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i, symbol in enumerate(cryptos):
        try:
            df = fetch_and_prepare_data(symbol)
            if df is not None and not df.empty:
                axes[i].plot(pd.to_datetime(df['timestamp'], unit='ms'), df['close'])
                axes[i].set_title(symbol)
                axes[i].set_xlabel("Timestamp")
                axes[i].set_ylabel("Precio de Cierre")
        except Exception as e:
            print(f"No se pudo generar gráfica para {symbol}: {e}")
            continue

    plt.tight_layout()
    plt.show()

# Ejecutar demo
if __name__ == "__main__":
    demo_trading()
