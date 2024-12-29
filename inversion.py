import time
import datetime
import ccxt
import os
import requests
import numpy as np
import sqlite3
from inversion_binance import execute_order_sell
from dotenv import load_dotenv

load_dotenv()

# Configurar Binance
exchange = ccxt.binance({
    "enableRateLimit": True,
    "apiKey": os.getenv("BINANCE_API_KEY_REAL"),  # Cambia a tu clave real
    "secret": os.getenv("BINANCE_SECRET_KEY_REAL"),
})

# Definir variables globales
THRESHOLD_VOLUME_CHANGE = 0.1  # Porcentaje de cambio en el volumen (10%)
THRESHOLD_PRICE_CHANGE = 0.05  # Cambio del 5%
INTERVAL_SECONDS = 14400  # Intervalo de 4 horas
last_volumes = {}
last_prices = {}
last_conditions = {}  # Global para registrar última revisión por símbolo


# Función para la lógica de inversión completa (ejecutar cada 20 minutos)
def run_investment_logic():
    print(f"🏁 Ejecutando lógica de inversión a las {datetime.datetime.now()}")
    try:
        # Llama a tu función de inversión principal
        monitor_and_run()
    except Exception as e:
        print(f"⚠️ Error en la lógica de inversión: {e}")

# Función principal para manejar ambos procesos
def main_loop():
    # Tiempos de última ejecución
    last_trailing_stop_time = time.time()
    last_investment_logic_time = time.time()

    while True:
        current_time = time.time()

        # Verificar si es tiempo de ejecutar el trailing stop (cada 1 minuto)
        if current_time - last_trailing_stop_time >= 60:  # 1 minuto
            monitor_trailing_stops()
            last_trailing_stop_time = current_time

        # Verificar si es tiempo de ejecutar la lógica de inversión (cada 20 minutos)
        if current_time - last_investment_logic_time >= 1200:  # 20 minutos
            run_investment_logic()
            last_investment_logic_time = current_time

        # Pequeña pausa para evitar consumir demasiados recursos
        time.sleep(1)

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

    # Comparar RSI con el último registrado
    last_rsi = last_conditions.get(symbol, {}).get("rsi", None)
    if last_rsi and abs(rsi - last_rsi) < 2:  # Cambios menores de 2 se ignoran
        print(f"⚠️ RSI no cambió significativamente para {symbol} ({rsi}).")
        return False

    # Actualizar condición para el símbolo
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
def significant_volume_change(symbol, current_volume, last_conditions):
    last_volume = last_conditions.get(symbol, {}).get("volume", None)

    if last_volume is not None and abs(current_volume - last_volume) / last_volume < THRESHOLD_VOLUME_CHANGE:
        print(f"⚠️ Cambio de volumen insignificante para {symbol}.")
        return False

    # Actualizar el volumen en las condiciones
    if symbol not in last_conditions:
        last_conditions[symbol] = {}
    last_conditions[symbol]["volume"] = current_volume

    print(f"⚠️ Cambio significativo detectado en el volumen de {symbol}, entrando al proceso de trading......")
    return True

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
    
# Función para implementar el trailing stop con validación de saldo mínimo y valor notional
def trailing_stop(symbol, current_price, atr_multiplier=2):
    """
    Monitorea y ajusta el trailing stop en base al ATR
    """
    try:
        # Obtener ATR
        atr = calculate_atr(symbol)
        if atr is None:
            print(f"⚠️ No se pudo calcular ATR para {symbol}")
            return

        # Consultar el precio de compra
        conn = sqlite3.connect('trading_real.db') 
        cursor = conn.cursor()
        cursor.execute("""
            SELECT price FROM transactions
            WHERE symbol = ? AND action = 'buy'
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol,))
        result = cursor.fetchone()

        if not result:
            print(f"⚠️ No se encontró precio de compra para {symbol} en la base de datos.")
            return

        buy_price = result[0]

        # Obtener saldo disponible del portafolio
        balance = exchange.fetch_balance()
        asset = symbol.split('/')[0]  # Obtener el nombre del activo (e.g., LINK)
        asset_balance = balance['free'].get(asset, 0)

        if asset_balance <= 0:
            print(f"⚠️ {asset} no tiene saldo disponible, se omite del trailing stop.")
            return

        # Calcular el valor notional en USDT
        value_in_usdt = asset_balance * current_price

        # Validar que el valor notional sea mayor al mínimo permitido
        min_notional = 5.0  # Valor mínimo de Binance
        if value_in_usdt < min_notional:
            print(f"⚠️ El valor notional para {symbol} es {value_in_usdt:.2f} USDT, menor al mínimo permitido de {min_notional:.2f} USDT.")
            return

        # Calcular el nivel de trailing stop
        max_price = max(buy_price, current_price)
        stop_price = max_price * (1 - atr_multiplier * atr)

        print(f"🔍 {symbol}: Precio compra: {buy_price:.2f}, Máximo: {max_price:.2f}, Trailing Stop: {stop_price:.2f}")

        # Si el precio cae por debajo del stop, vender
        if current_price <= stop_price:
            print(f"⚠️ Activando trailing stop para {symbol}. Vendiendo...")
            execute_order_sell(symbol, confidence=100, explanation="Trailing stop activado.")

        conn.close()

    except Exception as e:
        print(f"❌ Error en el trailing stop para {symbol}: {e}")

# Ejecutar trailing stop para todo el portafolio
# Función para monitorear trailing stops con validación de saldo mínimo
def monitor_trailing_stops():
    """
    Monitorea el portafolio y aplica el trailing stop a cada activo con saldo mayor a 0.1 USDT.
    """
    try:
        # Obtener el portafolio actual de Binance
        balance = exchange.fetch_balance()

        for asset, details in balance['total'].items():
            # Ignorar USDT o activos con saldo cero
            if asset == 'USDT' or details <= 0:
                continue  

            market_symbol = f"{asset}/USDT"
            current_price = fetch_price(market_symbol)

            if current_price:
                # Calcular el valor en USDT
                value_in_usdt = details * current_price

                # Ignorar monedas con valor menor a 0.1 USDT
                if value_in_usdt < 0.5:
                    #print(f"⚠️ {asset} tiene un valor menor a 0.1 USDT, se omite del monitoreo.")
                    continue  # Corregido: usar continue en lugar de return

                # Aplicar el trailing stop si pasa la validación
                trailing_stop(market_symbol, current_price, trailing_percentage=0.05)  # Ajusta el porcentaje según tu estrategia

    except Exception as e:
        print(f"❌ Error al monitorear trailing stops: {e}")


# Función para ejecutar lógica de trading
def run_trading():
    print(f"🏁 Ejecutando demo_trading a las {datetime.datetime.now()}")
    try:
        from inversion_binance import demo_trading
        demo_trading()
        print("✅ Trading ejecutado correctamente.")
    except Exception as e:
        print(f"❌ Error al ejecutar demo_trading: {e}")

def fetch_portfolio_symbols():
    """
    Obtiene las criptos del portafolio con saldo positivo desde Binance y las devuelve en formato "SYMBOL/USDT".
    """
    try:
        balance = exchange.fetch_balance()
        portfolio_symbols = []
        for asset, details in balance['total'].items():
            if asset == 'USDT' or details <= 0:
                continue  # Ignorar USDT y activos sin saldo
            portfolio_symbols.append(f"{asset}/USDT")
        return portfolio_symbols
    except Exception as e:
        print(f"❌ Error al obtener criptos del portafolio: {e}")
        return []


# Monitoreo principal
def monitor_and_run():
    global INTERVAL_SECONDS

    # Obtener las criptos iniciales que siempre deseas monitorear
    base_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
    try:
        exchange.check_required_credentials()
    except Exception as e:
        print(f"Error de credenciales: {e}")

    
    # Obtener criptos dinámicamente
    portfolio_symbols = fetch_portfolio_symbols()
    symbols = list(set(base_symbols + portfolio_symbols))  # Combina y elimina duplicados
    binance_symbols = [symbol.replace("/", "") for symbol in symbols]

    next_execution_time = time.time() + INTERVAL_SECONDS
    last_conditions = {}  # Almacena RSI, volumen, precios, etc.

    print("🚀 Ejecución inicial al iniciar el programa.")
    run_trading()

    while True:
        print("Esperando 20 minutos para revisión de niveles para definir si se adelanta la siguiente operación")
        time.sleep(1200)  # Espera 20 minutos antes de la próxima evaluación
        try:
            execute_now = False
            print(f"🔍 Monitoreando símbolos para definir si entrar en trading")

            for symbol, binance_symbol in zip(symbols, binance_symbols):
                # Obtener precios y volúmenes actuales
                current_price = fetch_price(binance_symbol)
                current_volume = fetch_volume(binance_symbol)
                if current_price is None or current_volume is None:
                    continue

                # Verificar condiciones para trading
                if should_trade(binance_symbol, last_conditions):
                    execute_now = True

                if significant_volume_change(binance_symbol, current_volume, last_conditions):
                    execute_now = True

                if significant_price_change(binance_symbol, current_price):
                    execute_now = True

            # Ejecutar si hay cambios significativos o se alcanza el tiempo de ejecución
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


# Llama a la función principal
if __name__ == "__main__":
    print("Iniciando el proceso principal...")
    print("Se hace el analisis de volumenes cada 20 minutos y el trailing stop cada 1 minuto")
    main_loop()
