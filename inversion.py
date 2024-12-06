import time
import datetime
import ccxt
import os

# Configurar Binance
exchange = ccxt.binance({
    "enableRateLimit": True,
    "apiKey": os.getenv("BINANCE_API_KEY"),  # Cambia a tu clave real
    "secret": os.getenv("BINANCE_SECRET_KEY"),
})

# Definir variables globales
THRESHOLD_VOLUME_CHANGE = 0.1  # Porcentaje de cambio en el volumen (10%)
INTERVAL_SECONDS = 3600  # Intervalo por defecto: una hora
last_volumes = {}

def fetch_volume(symbol):
    """
    Obtiene el volumen actual de un par de criptomonedas.
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['quoteVolume']  # Volumen en USDT
    except Exception as e:
        print(f"❌ Error al obtener el volumen para {symbol}: {e}")
        return None

def significant_volume_change(symbol, current_volume):
    """
    Verifica si el cambio en el volumen supera el umbral definido.
    """
    global last_volumes

    if symbol not in last_volumes:
        last_volumes[symbol] = current_volume
        return False

    last_volume = last_volumes[symbol]
    if last_volume == 0:
        return False  # Evitar división por cero

    volume_change = abs((current_volume - last_volume) / last_volume)
    last_volumes[symbol] = current_volume

    return volume_change > THRESHOLD_VOLUME_CHANGE

def run_trading():
    """
    Ejecuta la lógica principal de trading.
    """
    print(f"🏁 Ejecutando demo_trading a las {datetime.datetime.now()}")
    try:
        # Aquí llamas a tu función de trading
        from inversion_binance import demo_trading
        demo_trading()
    except Exception as e:
        print(f"❌ Error al ejecutar demo_trading: {e}")

def monitor_and_run():
    """
    Monitorea los volúmenes y ejecuta el trading si el cambio es significativo.
    """
    global INTERVAL_SECONDS
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]  # Ajusta según tus criptos

    while True:
        try:
            execute_now = False

            # Ejecutar monitoreo
            for symbol in symbols:
                current_volume = fetch_volume(symbol)
                if current_volume is None:
                    continue

                if significant_volume_change(symbol, current_volume):
                    print(f"⚠️ Cambio significativo detectado en {symbol}. Ejecutando ahora.")
                    execute_now = True

            if execute_now:
                run_trading()
            else:
                print(f"⏳ Esperando el siguiente intervalo de {INTERVAL_SECONDS} segundos.")

            # Esperar el siguiente intervalo
            time.sleep(INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\n❌ Monitoreo detenido por el usuario.")
            break
        except Exception as e:
            print(f"❌ Error durante el monitoreo: {e}")

if __name__ == "__main__":
    print("Iniciando monitoreo de trading...")
    monitor_and_run()
