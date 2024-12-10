import ccxt
import pandas as pd
from openai import OpenAI
import time
from elegir_cripto import choose_best_cryptos  # Importar la función de selección de criptos
from dotenv import load_dotenv
import os
import csv
from db_manager_real import initialize_db, insert_transaction, fetch_all_transactions, upgrade_db_schema, insert_market_condition, fetch_last_resistance_levels
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

def gpt_prepare_data(data_by_timeframe, additional_data):
    combined_data = ""
    for timeframe, df in data_by_timeframe.items():
        if df is not None and not df.empty:
            combined_data += f"\nDatos de {timeframe} (últimos 3 registros):\n"
            combined_data += df.tail(3).to_string(index=False) + "\n"

    prompt = f"""
    Eres un experto en análisis financiero y trading. Basándote en los siguientes datos de mercado e indicadores técnicos,
    analiza y decide si debemos comprar, vender o mantener para optimizar el rendimiento del portafolio.

    {combined_data}
    Indicadores críticos:
    - RSI: {additional_data.get('rsi', 'No disponible')}
    - Soporte: {additional_data.get('support', 'No disponible')}
    - Resistencia: {additional_data.get('resistance', 'No disponible')}
    - ADX: {additional_data.get('adx', 'No disponible')}
    - Precio actual: {additional_data.get('current_price', 'No disponible')}

    Indicadores secundarios:
    - Volumen relativo: {additional_data.get('relative_volume', 'No disponible')}
    - Desviación estándar del precio: {additional_data.get('price_std_dev', 'No disponible')}
    - Market cap: {additional_data.get('market_cap', 'No disponible')}
    - Fear & Greed Index: {additional_data.get('fear_greed', 'No disponible')}
    - Promedio 24h: {additional_data.get('avg_volume_24h', 'No disponible')}
    - Niveles historicos de resistencia recientes: {additional_data.get('historical_resistances', 'No disponible')}
    - Spread: {additional_data.get('spread', 'No disponible')}
    - Market Depth Bids: {additional_data.get('market_depth_bids', 'No disponible')}
    - Market Depth Asks: {additional_data.get('market_depth_asks', 'No disponible')}

    - Se necesita liquidez?: {additional_data.get('liquidity_need', 'No disponible')}
    
    Contexto histórico de las últimas transacciones:
    {fetch_all_transactions()}

    Basándote en esta información:
    1. Proporciona un resumen estructurado de los indicadores críticos y secundarios.
    2. Decide si debemos "comprar", "vender" o "mantener".
    3. Justifica tu decisión en 1-2 oraciones.
    4. Instrucciones adicionales {additional_data.get('instruction', 'No disponible')}
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
    message = response.choices[0].message.content.strip().lower()

    import re
    action = "mantener"
    if message.startswith("comprar"):
        action = "comprar"
    elif message.startswith("vender"):
        action = "vender"

    match = re.search(r'(\d+)%', message)
    confidence = int(match.group(1)) if match else 50
    explanation = message.split("\n",1)[0]

    return action, confidence, explanation


def chunk_list(lst, chunk_size):
    """Divide una lista en sublistas de tamaño chunk_size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def gpt_group_selection(data_by_symbol):
    """
    data_by_symbol: dict con {symbol: (data_by_timeframe, additional_data)}
    Selecciona la mejor cripto en dos fases:
    1. Divide en grupos pequeños y elige la mejor de cada grupo.
    2. De las ganadoras de cada grupo, elige la final con GPT-4.
    """
    symbols = list(data_by_symbol.keys())
    selected_per_group = {}

    # Primera fase: Selección por grupos (usando GPT-3.5)
    print("=== Iniciando selección por grupos ===")
    for group in chunk_list(symbols, 3):
        print(f"\nAnalizando grupo: {group}")
        prompt_for_group = "Eres un experto en análisis financiero. Aquí tienes datos de varias criptomonedas. Necesito que elijas la mejor para comprar de este grupo.\n"

        for symbol in group:
            data_by_timeframe, additional_data = data_by_symbol[symbol]
            sub_text = gpt_prepare_data(data_by_timeframe, additional_data)
            # Imprimimos una parte del sub_text para no saturar
            print(f"\n--- Datos para {symbol} ---\n{sub_text[:500]}...\n")  # Muestra primeros 500 caracteres
            prompt_for_group += f"\n### {symbol}\n{sub_text}\n"

        prompt_for_group += "\nBasándote en la información anterior, ¿cuál de estas criptos es la mejor opción para comprar? Devuelve solo el símbolo."
        
        # Imprimimos el prompt completo que se envía a GPT-3.5
        print(f"\n[Prompt a GPT-3.5 para el grupo {group}]:\n{prompt_for_group}\n")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis financiero y trading."},
                {"role": "user", "content": prompt_for_group}
            ],
            temperature=0.5
        )
        answer = response.choices[0].message.content.strip()
        print(f"[Respuesta GPT-3.5 para el grupo {group}]: {answer}")

        chosen_symbol = None
        for s in group:
            # Extraer la parte base del símbolo (antes de /USDT)
            base_name = s.split('/')[0].upper()  # Por ejemplo, PEPE/USDT -> PEPE
            # Comprobar si el nombre base está en la respuesta de GPT (ignorar mayúsculas/minúsculas)
            if base_name in answer.upper():
                chosen_symbol = s
                break

        if chosen_symbol:
            print(f"✔ Grupo {group}: GPT eligió {chosen_symbol} como ganador.")
            selected_per_group[chosen_symbol] = "winner"
        else:
            print(f"⚠ Grupo {group}: No se reconoció un símbolo claro en la respuesta. Se elige {group[0]} por defecto.")
            selected_per_group[group[0]] = "default_winner"

    # Segunda fase: Selección final entre ganadoras (usando GPT-4)
    finalists = list(selected_per_group.keys())
    print(f"\n=== Finalistas tras la primera fase: {finalists} ===")
    if len(finalists) == 1:
        final_winner = finalists[0]
        print(f"Solo hay un finalista: {final_winner}. No se requiere segunda fase.")
    else:
        prompt_final = "Tengo estas criptos finalistas, elige la mejor de entre ellas para comprar:\n"
        for sym in finalists:
            data_by_timeframe, additional_data = data_by_symbol[sym]
            sub_prepared = gpt_prepare_data(data_by_timeframe, additional_data)
            print(f"\n--- Datos finalistas para {sym} ---\n{sub_prepared[:500]}...\n")
            prompt_final += f"\n### {sym}\n{sub_prepared}\n"

        prompt_final += "\n¿Cuál es la mejor cripto para comprar? Devuelve solo su símbolo."

        # Imprimimos el prompt completo que se envía a GPT-4
        print(f"\n[Prompt a GPT-4 para finalistas]:\n{prompt_final}\n")

        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis financiero y trading."},
                {"role": "user", "content": prompt_final}
            ],
            temperature=0.5
        )
        final_answer = final_response.choices[0].message.content.strip()
        print(f"[Respuesta GPT-4 para finalistas]: {final_answer}")

        final_winner = None
        for s in finalists:
            if s in final_answer:
                final_winner = s
                break

        if not final_winner:
            print("⚠ GPT-4 no reconoció un finalista claro. Se toma el primero por defecto.")
            final_winner = finalists[0]
        else:
            print(f"✔ GPT-4 eligió {final_winner} como el ganador final.")

    return final_winner


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
    Parámetros:
        df: DataFrame con columnas ['high', 'low', 'close'].
        period: Periodo para calcular el ADX.
    Retorno:
        Último valor del ADX o None si no se puede calcular.
    """
    try:
        # Validar que el DataFrame tiene suficientes datos
        if len(df) < period:
            print(f"⚠️ No hay suficientes datos para calcular el ADX. Se requieren al menos {period} filas.")
            return None

        high = df['high']
        low = df['low']
        close = df['close']

        # Cálculo del DM+ y DM-
        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()

        # Cálculo del ATR
        true_range = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        # Cálculo del DI+ y DI-
        plus_di = (plus_dm / atr).rolling(window=period).mean() * 100
        minus_di = (minus_dm / atr).rolling(window=period).mean() * 100

        # Cálculo del DX
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100

        # Cálculo del ADX
        adx = dx.rolling(window=period).mean()

        # Retornar el último valor del ADX
        if adx.iloc[-1] is not None and not pd.isna(adx.iloc[-1]):
            return adx.iloc[-1]
        else:
            print("⚠️ ADX no calculable debido a datos insuficientes o NaN intermedios.")
            return None
    except Exception as e:
        print(f"❌ Error al calcular el ADX: {e}")
        return None


def fetch_price(symbol):
    """
    Obtiene el precio actual de un par de criptomonedas en USDT.
    """
    try:
        ticker = exchange.fetch_ticker(symbol)  # Fetch ticker para obtener datos actuales del mercado
        return ticker['last']  # Precio de la última transacción
    except Exception as e:
        print(f"❌ Error al obtener el precio para {symbol}: {e}")
        return None
    


# Función principal
# Función principal
def demo_trading():
    print("Iniciando proceso de inversión...")
    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
    print(f"Saldo disponible en USDT: {usdt_balance}")

    if usdt_balance <= 5:
        print("⚠️ Sin saldo suficiente en USDT. Se omite el proceso de compra.")
    else:

        selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=10)
        print(f"Criptos seleccionadas para análisis: {selected_cryptos}")

        data_by_symbol = {}
        for symbol in selected_cryptos:
            data_by_timeframe, volume_series, price_series = fetch_and_prepare_data(symbol)
            final_price = fetch_price(symbol)
            if data_by_timeframe and volume_series is not None and price_series is not None:
                support, resistance = calculate_support_resistance(price_series)
                adx = calculate_adx(data_by_timeframe["1h"])
                rsi = data_by_timeframe["1h"]["RSI"].iloc[-1] if "RSI" in data_by_timeframe["1h"].columns else "No disponible"
                market_depth = calculate_market_depth(symbol)
                candlestick_pattern = identify_candlestick_patterns(data_by_timeframe["1h"])

                additional_data = {
                "current_price": final_price,
                "relative_volume": calculate_relative_volume(volume_series),
                "rsi": rsi,
                "avg_volume_24h": fetch_avg_volume_24h(volume_series),
                "market_cap": fetch_market_cap(symbol),
                "spread": calculate_spread(symbol),
                "fear_greed": fetch_fear_greed_index(),
                "price_std_dev": calculate_price_std_dev(price_series),
                "adx": adx,
                "support": support,
                "resistance": resistance,
                "market_depth_bids": market_depth["total_bids"],
                "market_depth_asks": market_depth["total_asks"],
                "candlestick_pattern": candlestick_pattern,
            }
                data_by_symbol[symbol] = (data_by_timeframe, additional_data)
            else:
                print(f"⚠️ Datos insuficientes para {symbol}, se omite.")

        if not data_by_symbol:
            print("⚠️ No hay criptos con datos válidos para analizar.")
            return

        # Seleccionar la mejor cripto entre todas (fase grupal y luego final)
        final_winner = gpt_group_selection(data_by_symbol)
        if not final_winner:
            print("⚠️ No se pudo determinar una cripto ganadora.")
            return

        # Ahora que tenemos la cripto ganadora, consultamos a GPT si realmente debemos comprarla
        winner_data_by_timeframe, winner_additional_data = data_by_symbol[final_winner]
        prepared_text = gpt_prepare_data(winner_data_by_timeframe, winner_additional_data)
        action, confidence, explanation = gpt_decision(prepared_text)

        if action == "comprar":
            final_price = fetch_price(final_winner)
            if final_price and (usdt_balance > final_price * 0.001):
                amount_to_buy = TRADE_SIZE / final_price
                order = execute_order_buy(final_winner, amount_to_buy, confidence, explanation)
                if order:
                    print(f"✅ Orden de compra ejecutada para {final_winner}")
                    insert_transaction(
                        symbol=final_winner,
                        action="buy",
                        price=final_price,
                        amount=amount_to_buy,
                        timestamp=pd.Timestamp.now().isoformat(),
                        profit_loss=None,
                        confidence_percentage=confidence,
                        summary=explanation
                    )
                else:
                    print("❌ No se pudo ejecutar la orden de compra.")
            else:
                print("⚠️ No hay saldo suficiente o no se pudo obtener el precio para la cripto finalista.")
        else:
            # GPT decidió que no es momento de comprar (mantener o vender)
            print(f"↔️ GPT recomienda {action}. No se realiza compra de {final_winner}.")

    # Lógica de ventas
    portfolio_cryptos = get_portfolio_cryptos()
    filtered_portfolio = []
    for symbol in portfolio_cryptos:
        try:
            crypto_balance = exchange.fetch_balance()['free'].get(symbol, 0)
            market_symbol = f"{symbol}/USDT"
            current_price = fetch_price(market_symbol)
            if current_price is None:
                print(f"⚠️ No se pudo obtener el precio actual para {symbol}.")
                continue

            value_in_usdt = crypto_balance * current_price

            # Si el valor es menor a 0.1 USDT no vale la pena analizar venta
            # Ajusta este valor según el mínimo notional del exchange o tu criterio
            if value_in_usdt < 0.1:
                print(f"⚠️ {symbol} tiene un valor en USDT muy bajo ({value_in_usdt:.5f}), se omite análisis de venta.")
            else:
                filtered_portfolio.append(symbol)
                print(f"{symbol} vale {value_in_usdt:.2f} USDT, se incluye para análisis de venta.")

        except Exception as e:
            print(f"❌ Error al procesar {symbol} para filtrado: {e}")
            continue

    print(f"📊 Criptos en portafolio para analizar venta: {filtered_portfolio}")

    for symbol in filtered_portfolio:
        try:
            market_symbol = f"{symbol}/USDT"

            # Obtener datos históricos y preparar series de precios y volúmenes
            data_by_timeframe, volume_series, price_series = fetch_and_prepare_data(market_symbol)
            if not data_by_timeframe or volume_series is None or price_series is None:
                print(f"⚠️ Datos insuficientes para {market_symbol}.")
                continue

            # Calcular soporte y resistencia actuales
            support, resistance = calculate_support_resistance(price_series)

            # Calcular otros indicadores técnicos
            adx = calculate_adx(data_by_timeframe["1h"])
            current_price = fetch_price(market_symbol)
            relative_volume = calculate_relative_volume(volume_series)
            avg_volume_24h = fetch_avg_volume_24h(volume_series)
            market_cap = fetch_market_cap(market_symbol)
            spread = calculate_spread(market_symbol)
            price_std_dev = calculate_price_std_dev(price_series)
            candlestick_pattern = identify_candlestick_patterns(data_by_timeframe["1h"])
            market_depth = calculate_market_depth(market_symbol)

            # Insertar los datos actuales en la tabla `market_conditions`
            insert_market_condition(
                symbol=market_symbol,
                timestamp=pd.Timestamp.now().isoformat(),
                resistance=resistance,
                support=support,
                adx=adx,
                rsi=data_by_timeframe["1h"]["RSI"].iloc[-1] if "RSI" in data_by_timeframe["1h"].columns else None,
                relative_volume=relative_volume,
                avg_volume_24h=avg_volume_24h,
                market_cap=market_cap,
                spread=spread,
                price_std_dev=price_std_dev,
                candlestick_pattern=candlestick_pattern,
                current_price=current_price,
                market_depth_bids=market_depth["total_bids"],
                market_depth_asks=market_depth["total_asks"]
            )

            print(f"✅ Datos de mercado insertados para {symbol} en la base de datos.")

            # Recuperar resistencias históricas
            historical_resistances = fetch_last_resistance_levels(market_symbol)
            if not historical_resistances:
                print(f"⚠️ No se encontraron resistencias históricas para {symbol}.")
                continue

            # Preparar datos para GPT, incluyendo resistencias históricas
            resistances_to_consider = [resistance] + [r[0] for r in historical_resistances]
            additional_data = {
                "current_price": current_price,
                "support": support,
                "resistance": resistance,
                "historical_supports": resistances_to_consider,
                "adx": adx,
                "relative_volume": relative_volume,
                "avg_volume_24h": avg_volume_24h,
                "market_cap": market_cap,
                "spread": spread,
                "price_std_dev": price_std_dev,
                "candlestick_pattern": candlestick_pattern,
                "liquidity_need": True,
                "Instruction": "If liquidity is True, try to sell if close to the levels of resistance based on historical supports"
            }

            # Preparar texto para GPT con los datos
            prepared_text = gpt_prepare_data(data_by_timeframe, additional_data)

            # Decisión de GPT
            action, confidence, explanation = gpt_decision(prepared_text)

            # Ejecutar venta si GPT lo decide
            if action == "vender":
                crypto_balance = exchange.fetch_balance()['free'].get(symbol, 0)
                if crypto_balance > 0:
                    order = execute_order_sell(market_symbol, confidence, explanation)
                    if order:
                        insert_transaction(
                            symbol=market_symbol,
                            action="sell",
                            price=current_price,
                            amount=crypto_balance,
                            timestamp=pd.Timestamp.now().isoformat(),
                            profit_loss=None,
                            confidence_percentage=confidence,
                            summary=explanation
                        )
                        print(f"✅ Venta realizada para {symbol} a {current_price} USDT.")
                else:
                    print(f"⚠️ No tienes suficiente {symbol} para vender.")
            else:
                print(f"↔️ Decisión de GPT: mantener {symbol}.")

            # Pausa entre solicitudes para evitar limitaciones del exchange
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
