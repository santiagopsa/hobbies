import ccxt
import pandas as pd
from openai import OpenAI
import time
from elegir_cripto import choose_best_cryptos  # Importar la función de selección de criptos
from dotenv import load_dotenv
import os
import csv
from db_manager_real import initialize_db, insert_transaction, fetch_all_transactions, upgrade_db_schema
import requests

initialize_db()
#upgrade_db_schema()


if os.getenv("HEROKU") is None:
    load_dotenv()

# Configurar APIs de OpenAI y CCXT
exchange = ccxt.binance({
    "enableRateLimit": True
})

exchange.apiKey = os.getenv("BINANCE_API_KEY_REAL")
exchange.secret = os.getenv("BINANCE_SECRET_KEY_REAL")


# exchange.set_sandbox_mode(True)

# Verificar conexión
try:
    print("Conectando a Binance REAL...")
    balance = exchange.fetch_balance()
    #print("Conexión exitosa. Balance:", balance)
except Exception as e:
    print("Error al conectar con Binance:", e)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada")

client = OpenAI(api_key=OPENAI_API_KEY)

# Variables globales
TRADE_SIZE = 10
TRANSACTION_LOG = []

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def fetch_market_cap(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['quoteVolume'] * ticker['last']  # Aproximado como volumen * precio
    except Exception as e:
        print(f"Error al obtener el market cap para {symbol}: {e}")
        return None

def calculate_relative_volume(volume_series):
    return volume_series.iloc[-1] / volume_series.mean() # Último volumen comparado con la media

def calculate_spread(symbol):
    try:
        order_book = exchange.fetch_order_book(symbol)
        spread = order_book['asks'][0][0] - order_book['bids'][0][0]
        return spread
    except Exception as e:
        print(f"Error al calcular el spread para {symbol}: {e}")
        return None

def fetch_fear_greed_index():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url)
        data = response.json()
        return data['data'][0]['value']  # Índice de miedo/codicia
    except Exception as e:
        print(f"Error al obtener el Fear & Greed Index: {e}")
        return None
    
def calculate_price_std_dev(price_series):
    return price_series.std()

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
    

def fetch_and_prepare_data(symbol):
    """
    Obtiene datos históricos en múltiples marcos temporales y calcula indicadores técnicos.
    """
    try:
        # Definir marcos temporales y cantidad de datos a obtener
        timeframes = ['1h', '4h', '1d']  # Horas, 4 horas, días
        data = {}
        volume_series = None
        price_series = None

        # Obtener datos para cada marco temporal
        for timeframe in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Solo guardar los datos de 1h para las series de volumen y precio
            if timeframe == '1h':
                volume_series = df['volume']
                price_series = df['close']

            # Calcular indicadores técnicos
            df['MA_20'] = df['close'].rolling(window=20).mean()  # Media móvil de 20 periodos
            df['MA_50'] = df['close'].rolling(window=50).mean()  # Media móvil de 50 periodos
            df['RSI'] = calculate_rsi(df['close'])  # Índice de Fuerza Relativa
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['close'])  # Bandas de Bollinger

            data[timeframe] = df  # Almacenar datos por marco temporal

        return data, volume_series, price_series  # Devuelve los datos y las series
    except Exception as e:
        print(f"Error al obtener datos para {symbol}: {e}")
        return None, None, None

# Función para calcular el RSI
def calculate_rsi(series, period=14):
    """
    Calcula el Índice de Fuerza Relativa (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Función para calcular Bandas de Bollinger
def calculate_bollinger_bands(series, window=20, num_std_dev=2):
    """
    Calcula las Bandas de Bollinger.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, rolling_mean, lower_band

# Decisión de trading con GPT
def gpt_prepare_data(data_by_timeframe, additional_data):
    """
    Prepara un prompt estructurado para GPT basado en datos técnicos.
    """
    combined_data = ""

    # Incorporar datos de marcos temporales
    for timeframe, df in data_by_timeframe.items():
        if df is not None and not df.empty:
            combined_data += f"\nDatos de {timeframe}:\n"
            combined_data += df.tail(3).to_string(index=False)  # Últimos 3 registros

    # Priorizar indicadores críticos
    prompt = f"""
    Eres un experto en análisis financiero y trading. Basándote en los siguientes datos de mercado e indicadores técnicos,
    analiza y decide si debemos comprar, vender o mantener para optimizar el rendimiento del portafolio.

    Indicadores críticos:
    - RSI (Índice de Fuerza Relativa): {additional_data.get('rsi', 'No disponible')}
    - Soporte: {additional_data.get('support', 'No disponible')}
    - Resistencia: {additional_data.get('resistance', 'No disponible')}
    - ADX (Índice Direccional Promedio): {additional_data.get('adx', 'No disponible')}
    - Precio actual: {additional_data.get('current_price', 'No disponible')}

    Indicadores secundarios:
    - Volumen relativo: {additional_data.get('relative_volume', 'No disponible')}
    - Desviación estándar del precio: {additional_data.get('price_std_dev', 'No disponible')}
    - Market cap: {additional_data.get('market_cap', 'No disponible')}
    - Fear & Greed Index: {additional_data.get('fear_greed', 'No disponible')}

    Contexto histórico de las últimas transacciones:
    {fetch_all_transactions()}

    Basándote en esta información:
    1. Proporciona un resumen estructurado de los indicadores críticos y secundarios.
    2. Decide si debemos "comprar", "vender" o "mantener".
    3. Justifica tu decisión en 1-2 oraciones.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en análisis financiero y trading."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()


def gpt_decision(prepared_text):
    """
    Usa GPT-4 Turbo para analizar el texto preparado y decidir si comprar, vender o mantener.
    """
    prompt = f"""
    Eres un experto en trading. Basándote en el siguiente texto estructurado, decide si comprar, vender o mantener.

    Texto:
    {prepared_text}

    Inicia tu respuesta con: "comprar", "vender" o "mantener". Incluye un resumen extremadamente corto de la decisión y un porcentaje de confianza.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en trading."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    message = response.choices[0].message.content.strip()

    if message.lower().startswith('comprar'):
        action = "comprar"
        confidence = extract_confidence(message)
        explanation = message[7:].strip()
    elif message.lower().startswith('vender'):
        action = "vender"
        confidence = extract_confidence(message)
        explanation = message[6:].strip()
    elif message.lower().startswith('mantener'):
        action = "mantener"
        confidence = extract_confidence(message)
        explanation = message[8:].strip()
    else:
        action = "mantener"
        confidence = 50  # Valor predeterminado
        explanation = "No hay una recomendación clara."

    return action, confidence, explanation


def extract_confidence(message):
    """
    Extrae el porcentaje de confianza de la respuesta.
    """
    import re
    match = re.search(r'(\d+)%', message)
    if match:
        return int(match.group(1))
    return 50  # Valor predeterminado si no se encuentra un porcentaje

def is_valid_notional(symbol, amount):
    """
    Verifica si el valor notional cumple con los requisitos mínimos de Binance.
    """
    try:
        # Cargar los datos del mercado para el símbolo
        markets = exchange.load_markets()
        market = markets.get(symbol)

        # Obtener el precio actual del símbolo
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']  # Último precio del par

        # Calcular el valor notional
        notional = current_price * amount

        # Obtener el valor mínimo de notional desde el mercado
        if market and 'limits' in market and 'cost' in market['limits']:
            min_notional = market['limits']['cost']['min']
        else:
            min_notional = 10  # Valor mínimo genérico si no está definido

        print(f"🔍 Verificación para {symbol}:")
        print(f"    Precio actual: {current_price} USDT")
        print(f"    Cantidad: {amount}")
        print(f"    Valor notional: {notional} USDT")
        print(f"    Mínimo permitido: {min_notional} USDT")

        # Validar si el notional cumple con el requisito
        is_valid = notional >= min_notional
        if not is_valid:
            print(f"⚠️ El valor notional para {symbol} es {notional:.2f} USDT, menor al mínimo permitido de {min_notional:.2f} USDT.")
        return is_valid
    except Exception as e:
        print(f"❌ Error al verificar el valor notional para {symbol}: {e}")
        return False


# Ejecutar orden de compra
def execute_order_buy(symbol, amount, confidence, explanation):
    """
    Ejecuta una orden de compra y registra la transacción en la base de datos.
    """
    try:
        # Validar el notional antes de ejecutar la orden
        if not is_valid_notional(symbol, amount):
            print(f"⚠️ Orden de compra para {symbol} no válida debido al valor notional.")
            return None

        # Ejecutar la orden de compra
        order = exchange.create_market_buy_order(symbol, amount)
        price = order["price"] or 0
        timestamp = pd.Timestamp.now().isoformat()

        # Registrar en la base de datos
        insert_transaction(
            symbol=symbol,
            action="buy",
            price=price,
            amount=amount,
            timestamp=timestamp,
            profit_loss=None,
            confidence_percentage=confidence,
            summary=explanation
        )

        print(f"✅ Orden de compra ejecutada: {symbol}, Precio: {price}, Cantidad: {amount}")
        return order
    except Exception as e:
        print(f"❌ Error al ejecutar la orden de compra para {symbol}: {e}")
        return None

def execute_order_sell(symbol, confidence, explanation):
    """
    Ejecuta una orden de venta y registra la transacción en la base de datos.
    """
    try:
        # Obtener el saldo disponible
        balance = exchange.fetch_balance()
        amount = balance['free'].get(symbol.split('/')[0], 0)

        if amount <= 0:
            print(f"⚠️ No tienes suficiente saldo para vender {symbol}.")
            return None

        # Validar el notional
        if not is_valid_notional(symbol, amount):
            print(f"⚠️ Orden de venta para {symbol} no válida debido al valor notional mínimo.")
            return None

        # Ejecutar la orden de venta
        order = exchange.create_market_sell_order(symbol, amount)

        # Obtener el precio actual si no está en la orden
        ticker = exchange.fetch_ticker(symbol)
        price = order.get("price") or ticker["last"]
        timestamp = pd.Timestamp.now().isoformat()

        # Recuperar todas las compras del símbolo
        transactions = fetch_all_transactions()
        buys = [t for t in transactions if t[1] == symbol and t[2] == "buy"]

        # Calcular el precio promedio de compra
        if buys:
            total_cost = sum(buy[3] * buy[4] for buy in buys)  # Precio * Cantidad
            total_amount = sum(buy[4] for buy in buys)  # Suma de cantidades
            average_price = total_cost / total_amount
        else:
            average_price = 0

        # Calcular ganancia/pérdida
        profit_loss = (price - average_price) * amount

        # Registrar en la base de datos
        insert_transaction(
            symbol=symbol,
            action="sell",
            price=price,
            amount=amount,
            timestamp=timestamp,
            profit_loss=profit_loss,
            confidence_percentage=confidence,
            summary=explanation
        )

        print(f"✅ Orden de venta ejecutada: {symbol}, Precio: {price}, Cantidad: {amount}, Ganancia/Pérdida: {profit_loss}")
        return order
    except Exception as e:
        print(f"❌ Error al ejecutar la orden de venta para {symbol}: {e}")
        return None


def show_transactions():
    transactions = fetch_all_transactions()
    print("Historial de transacciones:")
    for t in transactions:
        print(f"ID: {t[0]}, Symbol: {t[1]}, Action: {t[2]}, Price: {t[3]}, Amount: {t[4]}, Timestamp: {t[5]}, Profit/Loss: {t[6]}, Confidence %: {t[7]}, Summary: {t[8]}")

def calculate_trade_amount(symbol, current_price, confidence, trade_size, min_notional):
    """
    Calcula la cantidad a negociar basada en el precio actual, la confianza, el tamaño máximo permitido y el notional mínimo.

    Args:
        symbol (str): El par de criptomonedas (e.g., BTC/USDT).
        current_price (float): El precio actual de la criptomoneda.
        confidence (float): El porcentaje de confianza de la decisión.
        trade_size (float): El tamaño máximo del trade en USD.
        min_notional (float): El valor notional mínimo permitido por el exchange.

    Returns:
        float: La cantidad de criptomoneda a negociar.
    """
    # Calcular el valor notional basado en la confianza
    desired_notional = trade_size * (confidence / 100)

    # Ajustar el notional si está por debajo del mínimo permitido
    if desired_notional < min_notional and confidence > 80:
        print(f"⚠️ Ajustando el trade a cumplir con el notional mínimo para {symbol}.")
        desired_notional = min_notional

    # Calcular la cantidad en criptomoneda basada en el notional final
    trade_amount = desired_notional / current_price

    # Garantizar que la cantidad esté dentro de los límites
    max_trade_amount = trade_size / current_price
    trade_amount = min(trade_amount, max_trade_amount)

    return trade_amount

def fetch_avg_volume_24h(volume_series):
    """
    Calcula el volumen promedio de las últimas 24 horas basado en la serie de volumen.
    """
    if volume_series is None or len(volume_series) < 24:
        print("⚠️ Datos insuficientes para calcular el volumen promedio de 24h.")
        return None
    return volume_series.tail(24).mean()

def identify_candlestick_patterns(df):
    """
    Identifica patrones básicos de velas japonesas.
    """
    last_candle = df.iloc[-1]
    body = abs(last_candle['close'] - last_candle['open'])
    upper_shadow = last_candle['high'] - max(last_candle['close'], last_candle['open'])
    lower_shadow = min(last_candle['close'], last_candle['open']) - last_candle['low']

    if lower_shadow > body * 2 and upper_shadow < body:
        return "hammer"  # Martillo
    elif upper_shadow > body * 2 and lower_shadow < body:
        return "shooting_star"  # Estrella fugaz
    else:
        return "none"

def calculate_market_depth(symbol, depth=10):
    """
    Calcula la profundidad del mercado basado en las 10 mejores órdenes de compra y venta.
    """
    try:
        order_book = exchange.fetch_order_book(symbol)
        total_bids = sum([bid[1] for bid in order_book['bids'][:depth]])  # Volumen total en bids
        total_asks = sum([ask[1] for ask in order_book['asks'][:depth]])  # Volumen total en asks
        return {"total_bids": total_bids, "total_asks": total_asks}
    except Exception as e:
        print(f"Error al calcular la profundidad del mercado para {symbol}: {e}")
        return {"total_bids": None, "total_asks": None}

def calculate_support_resistance(price_series, period=14):
    """
    Calcula niveles de soporte y resistencia basado en máximos y mínimos locales.
    """
    rolling_max = price_series.rolling(window=period).max()
    rolling_min = price_series.rolling(window=period).min()

    support = rolling_min.iloc[-1]
    resistance = rolling_max.iloc[-1]
    return support, resistance

def calculate_correlation_with_btc(symbol_price_series, btc_price_series):
    """
    Calcula la correlación entre el precio de una cripto y BTC.
    """
    if len(symbol_price_series) != len(btc_price_series):
        print("⚠️ Las series de precios tienen tamaños diferentes.")
        return None

    correlation = symbol_price_series.corr(btc_price_series)
    return correlation

def calculate_adx(df, period=14):
    """
    Calcula el Índice Direccional Promedio (ADX) usando los datos OHLC.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().clip(upper=0).abs()

    atr = (plus_dm + minus_dm).rolling(window=period).mean()

    plus_di = (plus_dm / atr) * 100
    minus_di = (minus_dm / atr) * 100
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100

    adx = dx.rolling(window=period).mean()
    return adx.iloc[-1]  # Último valor del ADX


# Función principal
def demo_trading():
    print("Iniciando proceso de inversión...")

    # Obtener saldo en USDT
    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
    print(f"Saldo disponible en USDT: {usdt_balance}")

    # Si no hay saldo suficiente, omitir compras
    if usdt_balance <= 5:
        print("⚠️ Sin saldo suficiente en USDT. Se omiten compras y se pasa directamente a analizar ventas.")
    else:
        # Elegir las mejores criptos para compra
        selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=10)
        print(f"Criptos seleccionadas para compra: {selected_cryptos}")

        # Realizar análisis para compra
        for symbol in selected_cryptos:
            try:
                # Obtener datos históricos de múltiples marcos temporales y las series
                data_by_timeframe, volume_series, price_series = fetch_and_prepare_data(symbol)

                # Verificar si los datos son válidos
                if not data_by_timeframe or volume_series is None or price_series is None:
                    print(f"⚠️ Datos insuficientes para {symbol}.")
                    continue

                # Calcular métricas adicionales
                support, resistance = calculate_support_resistance(price_series)
                market_depth = calculate_market_depth(symbol)
                candlestick_pattern = identify_candlestick_patterns(data_by_timeframe["1h"])
                adx = calculate_adx(data_by_timeframe["1h"])
                btc_series = fetch_and_prepare_data("BTC/USDT")[2]
                correlation_with_btc = calculate_correlation_with_btc(price_series, btc_series)

                additional_data = {
                    "relative_volume": calculate_relative_volume(volume_series),
                    "avg_volume_24h": fetch_avg_volume_24h(volume_series),
                    "market_cap": fetch_market_cap(symbol),
                    "spread": calculate_spread(symbol),
                    "fear_greed": fetch_fear_greed_index(),
                    "price_std_dev": calculate_price_std_dev(price_series),
                    "adx": adx,
                    "correlation_with_btc": correlation_with_btc,
                    "support": support,
                    "resistance": resistance,
                    "market_depth_bids": market_depth["total_bids"],
                    "market_depth_asks": market_depth["total_asks"],
                    "candlestick_pattern": candlestick_pattern,
                }

                # Preparar texto con GPT-3.5 Turbo
                prepared_text = gpt_prepare_data(data_by_timeframe, additional_data)

                # Analizar la decisión con GPT-4 Turbo
                action, confidence, explanation = gpt_decision(prepared_text)

                # Decidir acción de compra
                if action == "comprar":
                    current_price = data_by_timeframe["1h"]['close'].iloc[-1]
                    market_info = exchange.load_markets().get(symbol)
                    min_notional = market_info['limits']['cost']['min'] if market_info else 10  # Default if no data

                    trade_amount = calculate_trade_amount(
                        symbol=symbol,
                        current_price=current_price,
                        confidence=confidence,
                        trade_size=TRADE_SIZE,
                        min_notional=min_notional,
                    )

                    print(f"💰 Trade Amount Calculado para {symbol}: {trade_amount}")

                    # Verificar saldo antes de ejecutar la compra
                    if usdt_balance >= trade_amount * current_price:
                        execute_order_buy(symbol, trade_amount, confidence, explanation)
                    else:
                        print(f"⚠️ Saldo insuficiente para comprar {symbol}. Saldo disponible: {usdt_balance} USDT.")
                else:
                    print(f"↔️ No se realiza ninguna acción para {symbol} (mantener).")

                time.sleep(1)

            except Exception as e:
                print(f"❌ Error procesando {symbol}: {e}")
                continue

    # Analizar portafolio para ventas
    portfolio_cryptos = get_portfolio_cryptos()
    print(f"📊 Criptos en portafolio para analizar venta: {portfolio_cryptos}")

    for symbol in portfolio_cryptos:
        try:
            # Usar formato de símbolo correcto (e.g., BTC/USDT)
            market_symbol = f"{symbol}/USDT"

            # Obtener datos históricos y series
            data_by_timeframe, volume_series, price_series = fetch_and_prepare_data(market_symbol)

            # Verificar si los datos son válidos
            if not data_by_timeframe or volume_series is None or price_series is None:
                print(f"⚠️ Datos insuficientes para {market_symbol}.")
                continue

            # Calcular métricas adicionales
            support, resistance = calculate_support_resistance(price_series)
            market_depth = calculate_market_depth(market_symbol)
            candlestick_pattern = identify_candlestick_patterns(data_by_timeframe["1h"])
            adx = calculate_adx(data_by_timeframe["1h"])
            additional_data = {
                "relative_volume": calculate_relative_volume(volume_series),
                "avg_volume_24h": fetch_avg_volume_24h(volume_series),
                "market_cap": fetch_market_cap(market_symbol),
                "spread": calculate_spread(market_symbol),
                "fear_greed": fetch_fear_greed_index(),
                "price_std_dev": calculate_price_std_dev(price_series),
                "adx": adx,
                "support": support,
                "resistance": resistance,
                "market_depth_bids": market_depth["total_bids"],
                "market_depth_asks": market_depth["total_asks"],
                "candlestick_pattern": candlestick_pattern,
            }

            # Preparar texto para venta
            prepared_text = gpt_prepare_data(data_by_timeframe, additional_data)

            # Analizar la decisión
            action, confidence, explanation = gpt_decision(prepared_text)

            # Decidir acción de venta
            if action == "vender":
                crypto_balance = exchange.fetch_balance()['free'].get(symbol, 0)
                if crypto_balance > 0:
                    execute_order_sell(market_symbol, confidence, explanation)
                else:
                    print(f"⚠️ No tienes suficiente {symbol} para vender.")
            else:
                print(f"↔️ No se realiza ninguna acción para {market_symbol} (mantener).")

            time.sleep(1)

        except Exception as e:
            print(f"❌ Error procesando {symbol}: {e}")
            continue


    # Mostrar resultados finales
    #print("\n--- Resultados finales ---")
    #print(f"Portafolio final: {exchange.fetch_balance()['free']}")

# Ejecutar demo
if __name__ == "__main__":
    demo_trading()
    show_transactions()
