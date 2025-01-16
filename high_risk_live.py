import os
import time
import pandas as pd
import ta
import ccxt
import requests
from dotenv import load_dotenv
import time

# Cargar configuración desde .env
load_dotenv()

# Configurar conexión al exchange
exchange = ccxt.binance({
    "apiKey": os.getenv("BINANCE_API_KEY_REAL"),
    "secret": os.getenv("BINANCE_SECRET_KEY_REAL"),
    "enableRateLimit": True
})

# Configuración global
MAX_PORTFOLIO_SIZE = 7
BUDGET_PER_TRADE = 10  # USDT por compra
MIN_24H_VOLUME = 1_000_000
TRAILING_STOP_PERCENTAGE = 0.4
RSI_THRESHOLD_BUY = 70
RSI_THRESHOLD_SELL = 80
last_buy_time = 0

# Stablecoins y fiat currencies a excluir
STABLECOINS = ["USDT", "USDC", "BUSD", "DAI", "TUSD", "PAX", "GUSD", "UST", "FRAX", "sUSD", "HUSD", "MIM", "USDP"]
FIAT_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "HKD", "SGD"]

# Función para enviar mensajes a Telegram
def send_telegram_message(message):
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("No se configuró Telegram.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

def fetch_portfolio():
    """
    Obtiene el portafolio actual del exchange y convierte los valores a USDT.
    """
    try:
        balance = exchange.fetch_balance()
        portfolio = {}
        
        for asset, amount in balance['free'].items():
            if amount > 0:  # Ignorar activos con balance cero
                if asset == "USDT":
                    # Para USDT, ya no es necesario hacer conversión
                    portfolio["USDT"] = {"amount": amount, "value_usdt": amount}
                else:
                    # Obtener el precio actual de la moneda en USDT
                    symbol = f"{asset}/USDT"
                    try:
                        current_price = fetch_current_price(symbol)
                        if current_price:
                            portfolio[asset] = {
                                "amount": amount,
                                "value_usdt": amount * current_price
                            }
                    except Exception as e:
                        print(f"Error obteniendo precio para {symbol}: {e}")

        # Filtrar solo activos con valor mayor a 0 USDT
        portfolio = {
            asset: data for asset, data in portfolio.items()
            if data["value_usdt"] >= 3  # Filtrar activos con menos de 3 USDT de valor
        }

        if not portfolio:
            print("❌ Portafolio vacío o no se pudo obtener información.")
            send_telegram_message("❌ Portafolio vacío o no se pudo obtener información.")
            exit()

        print("Portafolio sincronizado (valores en USDT):", portfolio)
        return portfolio
    except Exception as e:
        print(f"Error al obtener el portafolio: {e}")
        send_telegram_message(f"❌ Error al obtener el portafolio: {e}")
        exit()


# Evaluar ventas basadas en trailing stop y RSI
# Evaluar ventas basadas en trailing stop y RSI
def evaluate_sell(portfolio, trailing_stop_percentage=0.4, rsi_threshold=80):
    for asset, details in portfolio.items():
        if asset == "USDT":
            continue

        # Acceder a la cantidad del activo
        amount = details["amount"]
        if amount < 3:  # Si la cantidad es menor a 3, no evaluar
            print(f"Cantidad insuficiente para {asset}, no evaluado para venta.")
            continue

        try:
            symbol = f"{asset}/USDT"
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=50)
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

            max_price = df["high"].max()
            last_price = df.iloc[-1]["close"]
            trailing_stop = max_price * (1 - trailing_stop_percentage)

            print(f"Evaluando {symbol}: Precio actual {last_price}, RSI {df.iloc[-1]['RSI']}, Trailing Stop {trailing_stop}")

            # Lógica de venta
            if last_price < trailing_stop or df.iloc[-1]["RSI"] > rsi_threshold:
                execute_order_sell(symbol, amount)

        except Exception as e:
            print(f"Error evaluando venta para {asset}: {e}")

def evaluate_buy(portfolio):
    global last_buy_time
    current_time = time.time()

    # Contar monedas en el portafolio excluyendo USDT
    non_usdt_assets = [asset for asset in portfolio if asset != "USDT"]
    print(non_usdt_assets)
    if len(non_usdt_assets) >= 7:
        print(f"⚠️ Demasiadas monedas en el portafolio ({len(non_usdt_assets)}). Permaneciendo en ciclo de ventas.")
        return

    # Verificar si han pasado 10 minutos desde la última compra
    if current_time - last_buy_time < 600:  # 600 segundos = 10 minutos
        print("Esperando para evaluar nuevas compras...")
        return

    try:
        print("Evaluando compras...")
        markets = exchange.load_markets()
        symbols = [s for s in markets if s.endswith("/USDT")]

        best_candidates = []

        for symbol in symbols:
            asset = symbol.split("/")[0]
            market_info = markets[symbol]

            # Validar que sea un mercado spot
            if not market_info.get("spot", False):
                continue

            # Excluir stablecoins y fiat currencies
            if asset in STABLECOINS or asset in FIAT_CURRENCIES:
                continue

            # Evitar comprar monedas ya presentes en el portafolio
            if asset in portfolio:
                continue

            try:
                ticker = exchange.fetch_ticker(symbol)

                # Validar volumen mínimo de 600,000 USDT
                if not ticker or ticker["quoteVolume"] < 600_000:
                    continue

                # Obtener datos OHLCV
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1h", limit=50)
                if not ohlcv or len(ohlcv) < 50:
                    continue

                # Calcular indicadores técnicos
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["MA_7"] = ta.trend.SMAIndicator(df["close"], window=7).sma_indicator()
                df["MA_25"] = ta.trend.SMAIndicator(df["close"], window=25).sma_indicator()
                df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

                last = df.iloc[-1]
                ma_crossover = last["MA_7"] > last["MA_25"]
                rsi_signal = last["RSI"] < 70

                if ma_crossover and rsi_signal:
                    score = (0.4 * ma_crossover) + (0.3 * (last["volume"] > 1.3 * df["volume"].mean())) + (0.3 * rsi_signal)
                    best_candidates.append({
                        "symbol": symbol,
                        "price": last["close"],
                        "score": score
                    })
            except Exception as e:
                print(f"Error analizando {symbol}: {e}")
                continue

        # Ordenar candidatos por puntaje
        best_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Intentar realizar una compra con el mejor candidato disponible
        for candidate in best_candidates:
            try:
                amount = BUDGET_PER_TRADE / candidate["price"]
                order = make_buy(candidate["symbol"], amount)
                if order:
                    last_buy_time = time.time()
                    print(f"Compra realizada: {candidate}")
                    return
            except Exception as e:
                if "Market is closed" in str(e):
                    print(f"Mercado cerrado para {candidate['symbol']}, intentando siguiente...")
                    continue
                else:
                    print(f"Error al intentar comprar {candidate['symbol']}: {e}")
                    break

        print("No hay criptomonedas elegibles para compra en este ciclo.")
    except Exception as e:
        print(f"Error evaluando compras: {e}")




def execute_order_sell(symbol, amount):
    """
    Ejecuta una orden de venta en Binance para el símbolo dado.
    Valida la cantidad mínima requerida antes de intentar vender.
    """
    try:
        # Obtener los requisitos del mercado
        market = exchange.market(symbol)
        min_amount = market["limits"]["amount"]["min"]

        # Verificar si la cantidad a vender cumple con el mínimo requerido
        if amount < min_amount:
            print(f"Cantidad insuficiente para {symbol}. Mínimo requerido: {min_amount}, Cantidad: {amount}")
            send_telegram_message(f"❌ No se puede vender {symbol}. Mínimo requerido: {min_amount}, Cantidad: {amount}")
            return None

        # Ejecutar la orden de venta
        order = exchange.create_market_sell_order(symbol, amount)
        message = f"✅ Venta ejecutada: {symbol}, Cantidad: {amount:.6f}"
        send_telegram_message(message)
        print(message)
        return order

    except Exception as e:
        message = f"❌ Error al ejecutar venta para {symbol}: {e}"
        send_telegram_message(message)
        print(message)
        return None
    
def fetch_current_price(symbol):
    """
    Obtiene el precio actual de un símbolo en Binance.
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker["last"]
        print(f"Precio actual para {symbol}: {current_price:.6f} USDT")
        return current_price
    except Exception as e:
        print(f"❌ Error al obtener el precio para {symbol}: {e}")
        send_telegram_message(f"❌ Error al obtener el precio para {symbol}: {e}")
        return None


def make_buy(symbol, amount_in_usd):
    """
    Realizar una compra en el exchange Binance.
    """
    try:
        # Obtener el precio actual del símbolo
        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker["last"]

        # Calcular la cantidad a comprar
        amount_to_buy = amount_in_usd / last_price

        # Verificar los límites mínimos del mercado
        market = exchange.markets[symbol]
        min_amount = market["limits"]["amount"]["min"]
        if amount_to_buy < min_amount:
            print(f"Cantidad insuficiente para comprar {symbol}. Mínimo requerido: {min_amount}, Cantidad: {amount_to_buy}")
            send_telegram_message(f"❌ No se pudo comprar {symbol}: cantidad insuficiente. Mínimo: {min_amount}, Intentado: {amount_to_buy}")
            return None
        # Crear orden de mercado de compra
        order = exchange.create_market_buy_order(symbol, amount_to_buy)
        message = f"✅ Compra ejecutada: {symbol}, Cantidad: {amount_to_buy:.6f}, Precio: {last_price:.2f} USDT"
        send_telegram_message(message)
        print(message)
        return order
    except Exception as e:
        message = f"❌ Error al ejecutar compra para {symbol}: {e}"
        send_telegram_message(message)
        print(message)
        return None
    
# Loop principal
def main_loop():
    while True:
        print("Evaluando ventas...")
        portfolio = fetch_portfolio()
        evaluate_sell(portfolio)

        print("Evaluando compras...")
        evaluate_buy(portfolio)

        print("Esperando 1 minuto...")
        time.sleep(60)

if __name__ == "__main__":
    main_loop()
