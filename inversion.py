import time
import datetime
import ccxt
import os
import requests
import numpy as np

# Configurar Binance
exchange = ccxt.binance({
    "enableRateLimit": True,
    "apiKey": os.getenv("BINANCE_API_KEY"),  # Cambia a tu clave real
    "secret": os.getenv("BINANCE_SECRET_KEY"),
})

# Definir variables globales
THRESHOLD_VOLUME_CHANGE = 0.05  # Porcentaje de cambio en el volumen (5%)
THRESHOLD_PRICE_CHANGE = 0.05  # Cambio del 5%
INTERVAL_SECONDS = 14400  # Intervalo de 4 horas
last_volumes = {}
last_prices = {}

# Función para obtener velas históricas
def fetch_klines(symbol, interval="1h", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"❌ Error al obtener velas para {symbol}: {response.text}")
        return []
    try:
        data = response.json()
        if not isinstance(data, list) or len(data) == 0:
            print(f"⚠️ Respuesta inesperada para {symbol}: {data}")
            return []
        return data
    except Exception as e:
        print(f"❌ Error al procesar datos de {symbol}: {e}")
        return []

# Función para calcular RSI
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        print("⚠️ No hay suficientes datos para calcular RSI.")
        return [np.nan]
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    rsis = [np.nan] * period
    for i in range(period, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsis.append(100 - (100 / (1 + rs)))
    return rsis

# Función para monitorear RSI y decidir trading
def should_trade(symbol):
    data = fetch_klines(symbol, "1h")
    if len(data) == 0:
        print(f"⚠️ No se recibieron datos de velas para {symbol}")
        return False
    closing_prices = [float(candle[4]) for candle in data if len(candle) > 4]
    if len(closing_prices) < 14:
        print(f"⚠️ No hay suficientes datos para calcular RSI de {symbol}")
        return False
    rsi = calculate_rsi(closing_prices, period=14)
    if rsi[-1] > 70:
        print(f"⚠️ RSI de {symbol} está sobrecomprado ({rsi[-1]}). Podría ser un buen momento para vender.")
        return True
    elif rsi[-1] < 30:
        print(f"⚠️ RSI de {symbol} está sobrevendido ({rsi[-1]}). Podría ser un buen momento para comprar.")
        return True
    else:
        print(f"RSI de {symbol}: {rsi[-1]}. No se toman acciones.")
        return False

# Función para detectar cambios significativos de precio
def significant_price_change(symbol, current_price):
    global last_prices
    if symbol not in last_prices:
        last_prices[symbol] = current_price
        return False
    last_price = last_prices[symbol]
    price_change = abs((current_price - last_price) / last_price)
    last_prices[symbol] = current_price
    return price_change > THRESHOLD_PRICE_CHANGE

# Función para detectar cambios significativos de volumen
def significant_volume_change(symbol, current_volume):
    global last_volumes
    if symbol not in last_volumes:
        last_volumes[symbol] = current_volume
        return False
    last_volume = last_volumes[symbol]
    if last_volume == 0:
        return False
    volume_change = abs((current_volume - last_volume) / last_volume)
    last_volumes[symbol] = current_volume
    return volume_change > THRESHOLD_VOLUME_CHANGE

# Función para obtener precios actuales
def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"❌ Error al obtener el precio para {symbol}: {e}")
        return None

# Función para obtener volumen actual
def fetch_volume(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['quoteVolume']
    except Exception as e:
        print(f"❌ Error al obtener el volumen para {symbol}: {e}")
        return None

# Función para ejecutar lógica de trading
def run_trading():
    print(f"🏁 Ejecutando demo_trading a las {datetime.datetime.now()}")
    try:
        from inversion_binance import demo_trading
        demo_trading()
        print("✅ Trading ejecutado correctamente.")
    except Exception as e:
        print(f"❌ Error al ejecutar demo_trading: {e}")

# Monitoreo principal
def monitor_and_run():
    global INTERVAL_SECONDS
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    binance_symbols = [symbol.replace("/", "") for symbol in symbols]
    next_execution_time = time.time() + INTERVAL_SECONDS

    # Ejecución inicial
    print("🚀 Ejecución inicial al iniciar el programa.")
    run_trading()

    while True:
        try:
            execute_now = False
            for symbol, binance_symbol in zip(symbols, binance_symbols):
                print(f"🔍 Monitoreando {symbol}")

                # Verificar precio y volumen
                current_price = fetch_price(binance_symbol)
                current_volume = fetch_volume(binance_symbol)
                if current_price is None or current_volume is None:
                    continue

                if significant_price_change(symbol, current_price):
                    print(f"⚠️ Cambio significativo en el precio de {symbol}. Ejecutando ahora.")
                    execute_now = True

                if significant_volume_change(symbol, current_volume):
                    print(f"⚠️ Cambio significativo detectado en {symbol}. Ejecutando ahora.")
                    execute_now = True

                # RSI y lógica de trading
                if should_trade(binance_symbol):
                    execute_now = True

            # Ejecutar trading si se cumplen condiciones
            if execute_now or time.time() >= next_execution_time:
                run_trading()
                next_execution_time = time.time() + INTERVAL_SECONDS
                print(f"⏳ Próxima ejecución programada a las {datetime.datetime.fromtimestamp(next_execution_time)}")
            else:
                print(f"⏳ No se cumplen las condiciones. Próxima evaluación en 10 segundos.")

            time.sleep(10)

        except KeyboardInterrupt:
            print("\n❌ Monitoreo detenido por el usuario.")
            break
        except Exception as e:
            print(f"❌ Error durante el monitoreo: {e}")

if __name__ == "__main__":
    print("Iniciando monitoreo de trading...")
    monitor_and_run()

