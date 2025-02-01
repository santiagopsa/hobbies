import time
import datetime
import ccxt
import os
import requests
import numpy as np
import sqlite3
from inversion_binance import demo_trading  # demo_trading may start orders and trailing stops as needed
from dotenv import load_dotenv

load_dotenv()

# Configurar Binance
exchange = ccxt.binance({
    "enableRateLimit": True,
    "apiKey": os.getenv("BINANCE_API_KEY_REAL"),  # Cambia a tu clave real
    "secret": os.getenv("BINANCE_SECRET_KEY_REAL"),
})

# Definir variables globales
THRESHOLD_VOLUME_CHANGE = 0.1  # 10%
THRESHOLD_PRICE_CHANGE = 0.1   # 10%
INTERVAL_SECONDS = 14400       # 4 horas
last_volumes = {}
last_prices = {}
last_conditions = {}  # Para registrar la última revisión por símbolo

def calculate_atr(symbol, period=14):
    try:
        klines = fetch_klines(symbol, interval="1h", limit=period+1)
        if not klines:
            return None
        true_ranges = []
        for i in range(1, len(klines)):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_close = float(klines[i-1][4])
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        atr = sum(true_ranges) / len(true_ranges)
        return atr / float(klines[-1][4])  # Como porcentaje del precio actual
    except Exception as e:
        print(f"Error calculating ATR for {symbol}: {e}")
        return None

# Función de inversión principal: se encarga de evaluar condiciones y ejecutar trading.
def run_investment_logic():
    print(f"🏁 Ejecutando lógica de inversión a las {datetime.datetime.now()}")
    try:
        monitor_and_run()
    except Exception as e:
        print(f"⚠️ Error en la lógica de inversión: {e}")

# Bucle principal de la aplicación (sin trailing stop en este módulo)
def main_loop():
    """
    Este bucle ejecuta la lógica de inversión cada 20 minutos.
    Dado que el trailing stop se maneja en inversion_binance.py,
    no se llama ninguna función de trailing stop aquí.
    """
    last_investment_logic_time = time.time()
    
    while True:
        current_time = time.time()
        if current_time - last_investment_logic_time >= 1200:  # Cada 20 minutos
            run_investment_logic()
            last_investment_logic_time = current_time
        time.sleep(1)

def fetch_klines(symbol, interval="1h", limit=100):
    binance_symbol = symbol.replace("/", "")
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
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

def should_trade(symbol, last_conditions):
    data = fetch_klines(symbol, "1h")
    if len(data) == 0:
        print(f"⚠️ No se recibieron datos de velas para {symbol}")
        return False

    closing_prices = [float(candle[4]) for candle in data if len(candle) > 4]
    if len(closing_prices) < 14:
        print(f"⚠️ No hay suficientes datos para calcular RSI de {symbol}")
        return False

    rsi = calculate_rsi(closing_prices, period=14)[-1]
    last_rsi = last_conditions.get(symbol, {}).get("rsi", None)
    if last_rsi and abs(rsi - last_rsi) < 2:
        print(f"⚠️ RSI no cambió significativamente para {symbol} ({rsi}).")
        return False

    last_conditions[symbol] = {"rsi": rsi}
    if rsi > 70:
        print(f"⚠️ RSI de {symbol} está sobrecomprado ({rsi}). Podría ser un buen momento para vender.")
        return True
    elif rsi < 30:
        print(f"⚠️ RSI de {symbol} está sobrevendido ({rsi}). Podría ser un buen momento para comprar.")
        return True
    else:
        print(f"RSI de {symbol}: {rsi}. No se toman acciones.")
        return False

def significant_price_change(symbol, current_price):
    global last_prices
    if symbol not in last_prices:
        last_prices[symbol] = current_price
        return False
    last_price = last_prices[symbol]
    price_change = abs((current_price - last_price) / last_price)
    last_prices[symbol] = current_price
    return price_change > THRESHOLD_PRICE_CHANGE

def significant_volume_change(symbol, current_volume, last_conditions):
    last_volume = last_conditions.get(symbol, {}).get("volume", None)
    if last_volume is not None and abs(current_volume - last_volume) / last_volume < THRESHOLD_VOLUME_CHANGE:
        print(f"⚠️ Cambio de volumen insignificante para {symbol}.")
        return False
    if symbol not in last_conditions:
        last_conditions[symbol] = {}
    last_conditions[symbol]["volume"] = current_volume
    print(f"⚠️ Cambio significativo detectado en el volumen de {symbol}, entrando al proceso de trading......")
    return True

def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"❌ Error al obtener el precio para {symbol}: {e}")
        return None

def fetch_volume(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['quoteVolume']
    except Exception as e:
        print(f"❌ Error al obtener el volumen para {symbol}: {e}")
        return None

def run_trading():
    print(f"🏁 Ejecutando demo_trading a las {datetime.datetime.now()}")
    try:
        demo_trading()  # Esta función (en inversion_binance.py) se encarga de ejecutar trading y gestionar trailing stops
        print("✅ Trading ejecutado correctamente.")
    except Exception as e:
        print(f"❌ Error al ejecutar demo_trading: {e}")

def fetch_portfolio_symbols():
    try:
        balance = exchange.fetch_balance()
        portfolio_symbols = []
        for asset, details in balance['total'].items():
            if asset == 'USDT' or details <= 0:
                continue
            portfolio_symbols.append(f"{asset}/USDT")
        return portfolio_symbols
    except Exception as e:
        print(f"❌ Error al obtener criptos del portafolio: {e}")
        return []

def monitor_and_run():
    """
    Esta función combina símbolos base y del portafolio para ejecutar la lógica de trading.
    Las evaluaciones (por RSI, volumen o cambios de precio) determinan si se debe ejecutar trading.
    """
    global INTERVAL_SECONDS
    base_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    try:
        exchange.check_required_credentials()
    except Exception as e:
        print(f"Error de credenciales: {e}")

    portfolio_symbols = fetch_portfolio_symbols()
    symbols = list(set(base_symbols + portfolio_symbols))
    binance_symbols = [symbol.replace("/", "") for symbol in symbols]
    next_execution_time = time.time() + INTERVAL_SECONDS
    last_conditions = {}

    print("🚀 Ejecución inicial al iniciar el programa.")
    run_trading()

    while True:
        print("Esperando 60 minutos para revisión de niveles para definir si se adelanta la siguiente operación")
        time.sleep(3600)  # Espera 20 minutos
        try:
            execute_now = False
            print(f"🔍 Monitoreando símbolos para definir si entrar en trading")
            for symbol, binance_symbol in zip(symbols, binance_symbols):
                current_price = fetch_price(binance_symbol)
                current_volume = fetch_volume(binance_symbol)
                if current_price is None or current_volume is None:
                    continue
                if should_trade(binance_symbol, last_conditions):
                    execute_now = True
                if significant_volume_change(binance_symbol, current_volume, last_conditions):
                    execute_now = True
                if significant_price_change(binance_symbol, current_price):
                    execute_now = True
            if execute_now or time.time() >= next_execution_time:
                run_trading()
                next_execution_time = time.time() + INTERVAL_SECONDS
                print(f"⏳ Próxima ejecución programada a las {datetime.datetime.fromtimestamp(next_execution_time)}")
            else:
                print(f"⏳ No se cumplen las condiciones. Próxima evaluación en 10 minutos.")
        except KeyboardInterrupt:
            print("\n❌ Monitoreo detenido por el usuario.")
            break
        except Exception as e:
            print(f"❌ Error durante el monitoreo: {e}")

if __name__ == "__main__":
    print("Iniciando el proceso principal...")
    print("Se ejecuta la lógica de inversión cada 20 minutos. Los trailing stops se gestionan en inversion_binance.py")
    main_loop()
