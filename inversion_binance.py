import ccxt
import pandas as pd
import numpy as np
import sqlite3
import os
import csv
import time
import requests
import json
import logging
import re
import math
import ta
import pytz
import threading
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from openai import OpenAI
from elegir_cripto import choose_best_cryptos  # Function to select cryptos
from db_manager_real import (
    initialize_db,
    insert_transaction,
    fetch_all_transactions,
    upgrade_db_schema,
    insert_market_condition,
    fetch_last_resistance_levels,
)

# =============================================================================
# CONFIGURATION & INITIALIZATION
# =============================================================================

load_dotenv()

GPT_MODEL = "gpt-4o-mini"

DB_NAME = "trading_real.db"
# Initialize the database (uncomment upgrade_db_schema() if needed)
initialize_db()
# upgrade_db_schema()

# (Optional) List of symbols you never want to sell. (Not used in this code.)
NO_SELL = ["BTC", "TRUMP"]

# Configure the Binance exchange via CCXT
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {
        'adjustForTimeDifference': True,
    },
})
exchange.apiKey = os.getenv("BINANCE_API_KEY_REAL")
exchange.secret = os.getenv("BINANCE_SECRET_KEY_REAL")
# exchange.set_sandbox_mode(True)  # Enable if needed

# Verify connection to Binance
try:
    print("Conectando a Binance REAL...")
    balance = exchange.fetch_balance()
    print("Conexión exitosa.")
except Exception as e:
    print("Error al conectar con Binance:", e)

# Configure OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está configurada")
client = OpenAI(api_key=OPENAI_API_KEY)

# Set up logging for GPT responses and errors
logging.basicConfig(
    level=logging.INFO,
    filename="gpt_responses.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =============================================================================
# UTILITY & INDICATOR FUNCTIONS
# =============================================================================

def get_colombia_timestamp():
    """Returns the current timestamp in the Colombia timezone."""
    colombia_timezone = pytz.timezone("America/Bogota")
    colombia_time = datetime.now(colombia_timezone)
    return colombia_time.strftime("%Y-%m-%d %H:%M:%S")

    """
    Calculates the net balance (buys minus sells) for high risk transactions
    in the last 24 hours.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_24h_timestamp = last_24h.strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            SELECT symbol, action, SUM(price * amount) AS total, risk_type
            FROM transactions
            WHERE timestamp >= ?
            GROUP BY symbol, action, risk_type
            """,
            (last_24h_timestamp,),
        )
        rows = cursor.fetchall()
        print("🔍 Transacciones encontradas (últimas 24h):", rows)

        transactions_by_symbol = {}
        for symbol, action, total, risk_type in rows:
            if symbol not in transactions_by_symbol:
                transactions_by_symbol[symbol] = {"buy": 0, "sell": 0, "risk_type": None}
            transactions_by_symbol[symbol][action] += total
            if risk_type:
                transactions_by_symbol[symbol]["risk_type"] = risk_type

        for symbol, data in transactions_by_symbol.items():
            if data["risk_type"] is None:
                cursor.execute(
                    """
                    SELECT risk_type
                    FROM transactions
                    WHERE symbol = ? AND action = 'buy'
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (symbol,),
                )
                last_risk_type = cursor.fetchone()
                data["risk_type"] = last_risk_type[0] if last_risk_type else "unknown"

        conn.close()

        high_risk_balance = 0
        for symbol, data in transactions_by_symbol.items():
            if data.get("risk_type") == "high_risk":
                high_risk_balance += data["buy"] - data["sell"]

        print(f"📊 Saldo neto de alto riesgo: {high_risk_balance:.2f} USDT")
        return high_risk_balance

    except sqlite3.Error as e:
        print(f"❌ Error al calcular el saldo de alto riesgo: {e}")
        return None

def send_telegram_message(message):
    """
    Sends a plain text message to your Telegram bot using environment variables.
    """
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("No se configuró el token o chat ID de Telegram.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Mensaje enviado a Telegram con éxito.")
        else:
            print(f"❌ Error al enviar mensaje a Telegram: {response.text}")
    except Exception as e:
        print(f"❌ Error al conectar con Telegram: {str(e)}")

def detect_momentum_divergences(price_series, rsi_values):
    """
    Detects divergences between price and RSI.
    Returns a list of tuples (divergence_type, index).
    """
    try:
        price_series = np.array(price_series)
        rsi_values = np.array(rsi_values)
        divergences = []
        window = 5  # Window size for local extrema detection

        for i in range(window, len(price_series) - window):
            if (all(price_series[i] > price_series[i-window:i]) and
                all(price_series[i] > price_series[i+1:i+window+1])):
                if price_series[i] > price_series[i-window] and rsi_values[i] < rsi_values[i-window]:
                    divergences.append(("bearish", i))
            if (all(price_series[i] < price_series[i-window:i]) and
                all(price_series[i] < price_series[i+1:i+window+1])):
                if price_series[i] < price_series[i-window] and rsi_values[i] > rsi_values[i-window]:
                    divergences.append(("bullish", i))
        return divergences
    except Exception as e:
        print(f"Error en detect_momentum_divergences: {e}")
        return []

def calculate_rsi(prices, period=14):
    """Calculates the Relative Strength Index (RSI) for a price series."""
    if len(prices) < period + 1:
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

def calculate_bollinger_bands(series, window=20, num_std_dev=2):
    """Calculates Bollinger Bands for a given price series."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, rolling_mean, lower_band

def calculate_support_resistance(price_series, period=14):
    """
    Calculates support and resistance levels based on rolling minima and maxima.
    Returns a tuple (support, resistance).
    """
    rolling_max = price_series.rolling(window=period).max()
    rolling_min = price_series.rolling(window=period).min()
    support = rolling_min.iloc[-1]
    resistance = rolling_max.iloc[-1]
    return support, resistance

def calculate_adx(df, period=14):
    """
    Calculates the Average Directional Index (ADX) using OHLC data.
    Returns the latest ADX value or None if not calculable.
    """
    try:
        if len(df) < period:
            print(f"⚠️ No hay suficientes datos para calcular el ADX.")
            return None

        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff().clip(lower=0)
        minus_dm = low.diff().clip(upper=0).abs()

        true_range = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        plus_di = (plus_dm / atr).rolling(window=period).mean() * 100
        minus_di = (minus_dm / atr).rolling(window=period).mean() * 100

        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()

        if adx.iloc[-1] is not None and not pd.isna(adx.iloc[-1]):
            return adx.iloc[-1]
        else:
            print("⚠️ ADX no calculable debido a datos insuficientes o NaN.")
            return None
    except Exception as e:
        print(f"❌ Error al calcular el ADX: {e}")
        return None

def fetch_price(symbol):
    """Obtains the current price for a given symbol."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"❌ Error al obtener el precio para {symbol}: {e}")
        return None

def fetch_and_prepare_data(symbol):
    """
    Fetches historical OHLCV data in multiple timeframes and calculates technical indicators.
    Returns a tuple: (data_by_timeframe, volume_series, price_series)
    """
    try:
        timeframes = ['1h', '4h', '1d']
        data = {}
        volume_series = None
        price_series = None

        for timeframe in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            if timeframe == '1h':
                volume_series = df['volume']
                price_series = df['close']

            df['MA_20'] = df['close'].rolling(window=20).mean()
            df['MA_50'] = df['close'].rolling(window=50).mean()
            df['RSI'] = calculate_rsi(df['close'])
            df['BB_upper'], df['BB_middle'], df['BB_lower'] = calculate_bollinger_bands(df['close'])

            data[timeframe] = df

        return data, volume_series, price_series
    except Exception as e:
        print(f"Error al obtener datos para {symbol}: {e}")
        return None, None, None

# =============================================================================
# GPT-BASED ANALYSIS FUNCTIONS
# =============================================================================

def gpt_prepare_data(data_by_timeframe, additional_data):
    """
    Combines market data from different timeframes and additional analysis,
    then prepares a prompt for GPT to analyze.
    """
    combined_data = ""
    for timeframe, df in data_by_timeframe.items():
        if df is not None and not df.empty:
            combined_data += f"\nDatos de {timeframe} (últimos 3 registros):\n"
            combined_data += df.tail(3).to_string(index=False) + "\n"

    prompt = f"""
    Eres un experto en análisis financiero y trading. Basándote en los siguientes datos de mercado e indicadores técnicos,
    analiza y decide si debemos comprar, vender o mantener para optimizar el rendimiento del portafolio en el corto plazo.

    {combined_data}
    
    Análisis Adicional:
    1. Divergencias de Momentum: {additional_data.get('momentum_divergences', 'No disponible')}
    2. Sentimiento del Mercado:
       - Tendencia de Volumen: {additional_data.get('market_sentiment', {}).get('volume_trend', 'No disponible')}
       - Sentimiento General: {additional_data.get('market_sentiment', {}).get('overall_sentiment', 'No disponible')}
    
    Basándote en esta información completa:
    1. Proporciona un resumen estructurado de los indicadores críticos y secundarios.
    2. Decide si debemos "comprar", "vender" o "mantener".
    3. Justifica tu decisión en 1 oración.
    4. Instrucciones adicionales: {additional_data.get('instruction', 'No disponible')}
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "developer", "content": "Eres un experto en crypto trading y análisis financiero."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def gpt_decision_buy(prepared_text):
    """
    Uses GPT to decide whether to buy or hold, based on the provided market data.
    Returns a tuple (accion, confianza, explicacion).
    """
    prompt = f"""
        Eres un experto en trading enfocado en criptomonedas de alta especulación.
        Tu objetivo es conseguir al menos un 3% de crecimiento en el corto plazo de la criptomoneda que queremos comprar, los indicadores te deben sugerir que es una buena oportunidad.

        Basándote en el siguiente texto estructurado, decide si COMPRAR esta criptomoneda por la tendencia que sugieren los indicadores, o si debes MANTENER.

        Condiciones clave:
        - No compres si hay sobrecompra
        - Buscamos alta probabilidad de crecimiento y aumento de volumen de manera inmediata.
        - Si los indicadores sugieren potencial de crecimiento, preferimos COMPRAR para mantener el capital en movimiento.
        - MANTENER solo en caso de señales negativas.

        Información disponible:
        {prepared_text}

        **Instrucciones:**
        - Responde únicamente en formato JSON con los campos "accion", "confianza" y "explicacion".
        - "accion" debe ser "comprar" o "mantener".
        - "confianza" debe ser un número entero entre 0 y 100.
        - "explicacion" debe contener una breve justificación (1 o 2 líneas).
        - No incluyas texto adicional.
    """
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "developer", "content": "Eres un experto en trading enfocado en criptomonedas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        message = response.choices[0].message.content.strip()
        logging.info(f"Respuesta completa de GPT: {message}")
        message = message.replace('```json', '').replace('```', '').strip()
        try:
            decision = json.loads(message)
            accion = decision.get("accion", "mantener").lower()
            confianza = int(decision.get("confianza", 50))
            explicacion = decision.get("explicacion", "")
            if accion not in ["comprar", "mantener"]:
                accion = "mantener"
            if not (0 <= confianza <= 100):
                confianza = 50
        except json.JSONDecodeError as e:
            logging.error(f"Error al parsear JSON: {e}")
            accion = "mantener"
            confianza = 50
            explicacion = "Respuesta no estructurada o inválida, se mantiene por defecto."
    except Exception as e:
        logging.error(f"Error en la llamada a la API de GPT: {e}")
        accion = "mantener"
        confianza = 50
        explicacion = "Error en la llamada a la API, se mantiene por defecto."
    return accion, confianza, explicacion

# =============================================================================
# GPT-BASED GROUP SELECTION FUNCTIONS
# =============================================================================

def chunk_list(lst, chunk_size):
    """Divides a list into chunks of size chunk_size."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def gpt_group_selection(data_by_symbol):
    """
    Uses GPT in two phases to select the best crypto candidate from the data.
    
    Phase 1: Divide symbols into small groups (e.g. groups of 6) and ask GPT
    to pick the best candidate from each group.
    
    Phase 2: From the winners of each group, ask GPT to choose the final candidate.
    
    Returns the final winning symbol.
    """
    symbols = list(data_by_symbol.keys())
    selected_per_group = {}

    print("=== Iniciando selección por grupos ===")
    for group in chunk_list(symbols, 6):
        print(f"\nAnalizando grupo: {group}")
        prompt_for_group = "Eres un experto en análisis financiero. Aquí tienes datos de varias criptomonedas. Necesito que elijas SOLO la mejor (SOLO UNA) para comprar de este grupo y me digas cuál es.\n"
        for symbol in group:
            data_by_timeframe, additional_data = data_by_symbol[symbol]
            sub_text = gpt_prepare_data(data_by_timeframe, additional_data)
            prompt_for_group += f"\n### {symbol}\n{sub_text}\n"
        prompt_for_group += "\nBasándote en la información anterior, ¿cuál de estas criptos es la mejor opción para comprar teniendo en cuenta que quiero vender en el corto plazo? Devuelve solo el símbolo de la mejor con el formato 'BTC/USDT'."

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "developer", "content": "Eres un experto en análisis financiero y trading."},
                {"role": "user", "content": prompt_for_group}
            ],
            temperature=0
        )
        answer = response.choices[0].message.content.strip()
        print(answer)
        chosen_symbol = None
        for s in group:
            base_name = s.split('/')[0].upper()
            if base_name in answer.upper():
                chosen_symbol = s
                break

        if chosen_symbol:
            print(f"✔ Grupo {group}: GPT eligió {chosen_symbol} como ganador.")
            selected_per_group[chosen_symbol] = "winner"
        else:
            print(f"⚠ Grupo {group}: No se reconoció un símbolo claro en la respuesta. Se elige {group[0]} por defecto.")
            selected_per_group[group[0]] = "default_winner"

    # Phase 2: From the winners, select the final candidate
    finalists = list(selected_per_group.keys())
    print(f"\n=== Finalistas tras la primera fase: {finalists} ===")
    if len(finalists) == 1:
        final_winner = finalists[0]
        print(f"Solo hay un finalista: {final_winner}. No se requiere segunda fase.")
    else:
        prompt_final = ("Eres un experto en análisis de criptomonedas. Necesito que analices los siguientes finalistas "
                        "y elijas el MEJOR para comprar en el corto plazo.\n\n")
        for sym in finalists:
            data_by_timeframe, additional_data = data_by_symbol[sym]
            sub_prepared = gpt_prepare_data(data_by_timeframe, additional_data)
            prompt_final += f"\n### {sym}\n{sub_prepared}\n"
        final_response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "developer", "content": "Eres un experto en análisis financiero de criptomonedas. DEBES responder ÚNICAMENTE con el símbolo exacto de la mejor cripto."},
                {"role": "user", "content": prompt_final}
            ],
            temperature=0
        )
        final_answer = final_response.choices[0].message.content.strip()
        print(f"[Respuesta GPT-3.5-turbo para finalistas]: {final_answer}")

        final_winner = None
        for sym in finalists:
            if sym in final_answer:
                final_winner = sym
                break
            base_symbol = sym.split('/')[0]
            if base_symbol in final_answer:
                final_winner = sym
                break

        if not final_winner:
            print("❌ Error: GPT no proporcionó un símbolo válido. Se selecciona el primer finalista.")
            final_winner = finalists[0]
    print(f"✅ GPT eligió {final_winner} como el ganador final.")
    return final_winner

# =============================================================================
# TRAILING STOP FUNCTIONS (SELLING LOGIC)
# =============================================================================

def set_trailing_stop(symbol, amount, purchase_price, trailing_percent=3, exchange_instance=None):
    """
    Configura un trailing stop para el símbolo dado.
    
    Cuando el precio actual alcanza el nivel de activación (en este caso, igual al precio de compra),
    la lógica del trailing stop comienza a seguir el precio máximo alcanzado.
    Una vez que el precio cae por debajo del máximo menos el trailing percentage, se ejecuta una venta inmediata.
    """
    if exchange_instance is None:
        exchange_instance = exchange

    def trailing_stop_logic():
        try:
            # En este ejemplo, la activación se realiza tan pronto se alcanza el precio de compra.
            activation_price = purchase_price
            logging.info(f"Trailing stop para {symbol} se activará al alcanzar {activation_price}")
            send_telegram_message(f"🔄 *Trailing Stop configurado* para `{symbol}`\nActivación al alcanzar `{activation_price}`")
            
            while True:
                current_price = fetch_price(symbol)
                if current_price is None:
                    logging.error(f"No se pudo obtener el precio actual para {symbol}.")
                    time.sleep(60)
                    continue

                if current_price >= activation_price:
                    logging.info(f"{symbol}: Precio de activación alcanzado. Configurando trailing stop.")
                    send_telegram_message(f"📊 *Trailing Stop Activado* para `{symbol}`\nPrecio Actual: `{current_price}`")
                    
                    highest_price = current_price
                    while True:
                        updated_price = fetch_price(symbol)
                        if updated_price is None:
                            logging.error(f"No se pudo obtener el precio actual para {symbol}.")
                            send_telegram_message(f"❌ *Error al obtener el precio actual* para `{symbol}`\nDetalles: No se pudo obtener el precio actual.")
                            time.sleep(60)
                            continue
                        
                        if updated_price > highest_price:
                            highest_price = updated_price
                            logging.info(f"{symbol}: Nuevo precio más alto alcanzado: {highest_price}")
                            send_telegram_message(f"📈 *Nuevo Precio Más Alto Alcanzado* para `{symbol}`\nNuevo Precio: `{highest_price} comprado a {purchase_price}`")
                        
                        stop_price = highest_price * (1 - trailing_percent / 100)
                        logging.debug(f"{symbol}: Precio actual: {updated_price} , Precio de stop: {stop_price}")
                        send_telegram_message(f"📉 *Precio de Stop Calculado* para `{symbol}`\nPrecio de Stop: `{stop_price}, precio comprado: {purchase_price}`")
                        
                        if updated_price < stop_price:
                            try:
                                # Ejecutar venta inmediata (orden de mercado)
                                sell_order = exchange_instance.create_market_sell_order(symbol, amount)
                                sell_order_id = sell_order.get('id', 'N/A')
                                timestamp = datetime.now(timezone.utc).isoformat()
                                insert_transaction(symbol, 'market_sell', updated_price, amount, timestamp, sell_order_id)
                                logging.info(f"Venta ejecutada para {symbol} a mercado, al precio: {updated_price}")
                                send_telegram_message(f"🔄 *Venta Ejecutada a Mercado* para `{symbol}`\nPrecio: `{updated_price}`\nCantidad: `{amount}`, precio comprado: `{purchase_price}` ganancia/perdida: `{(updated_price - purchase_price)*100/purchase_price} %`")
                                break
                            except Exception as e:
                                logging.error(f"Error al ejecutar la venta de mercado para {symbol}: {e}")
                                send_telegram_message(f"❌ *Error al ejecutar la venta de mercado* `{symbol}`\nDetalles: {e}")
                                break
                        time.sleep(60)
                    break
                time.sleep(60)
        except Exception as e:
            logging.error(f"Error en trailing_stop_logic para {symbol}: {e}")
            send_telegram_message(f"❌ *Error en trailing_stop_logic* `{symbol}`\nDetalles: {e}")

    trailing_thread = threading.Thread(target=trailing_stop_logic)
    trailing_thread.start()

def process_order(order, symbol, exchange_instance=exchange):
    """
    Processes the purchase order and starts the trailing stop process.
    
    Expects the order to be a dictionary containing at least:
      - 'price': the purchase price
      - 'filled': the purchased amount
    """
    if order:
        logging.info(f"Iniciando procesamiento de la orden para {symbol}.")
        purchase_price = order.get('price')
        amount = order.get('filled', 0)
        
        if purchase_price is None or amount <= 0:
            logging.error(f"Datos insuficientes en la orden para {symbol}. Precio: {purchase_price}, Cantidad: {amount}")
            send_telegram_message(f"❌ *Error al procesar la orden de compra* `{symbol}`\nDetalles: Datos insuficientes.")
            return
        
        logging.info("Pausando 5 segundos para actualización de balance.")
        time.sleep(5)
        
        asset = symbol.split('/')[0]
        balance = exchange.fetch_balance()
        available_amount = balance['free'].get(asset, 0)
        logging.info(f"Balance disponible para {asset}: {available_amount}")
        
        manage_amount = min(amount, available_amount)
        if manage_amount < amount:
            logging.warning(f"Saldo insuficiente para manejar la cantidad completa de {symbol}. Manejar {manage_amount} en lugar de {amount}.")
            send_telegram_message(f"⚠️ *Aviso de Saldo Insuficiente*\nSímbolo: `{symbol}`\nCantidad manejada: `{manage_amount}` en lugar de `{amount}`.")
        
        logging.info(f"Configurando trailing stop para {symbol} (purchase price: {purchase_price} USDT, cantidad: {manage_amount}).")
        set_trailing_stop(symbol, manage_amount, purchase_price, trailing_percent=3, exchange_instance=exchange_instance)
    else:
        logging.error(f"No se recibió orden para {symbol}.")

# =============================================================================
# ORDER EXECUTION (BUYING LOGIC)
# =============================================================================

def is_valid_notional(symbol, amount):
    """
    Verifies that the notional value (price * amount) meets the exchange's minimum.
    """
    try:
        markets = exchange.load_markets()
        market = markets.get(symbol)
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        notional = current_price * amount
        if market and 'limits' in market and 'cost' in market['limits']:
            min_notional = market['limits']['cost']['min']
        else:
            min_notional = 10  # Generic fallback minimum
        print(f"🔍 Verificación para {symbol}: Precio actual: {current_price} USDT, Cantidad: {amount}, Valor notional: {notional} USDT, Mínimo permitido: {min_notional} USDT")
        is_valid = notional >= min_notional
        if not is_valid:
            print(f"⚠️ El valor notional para {symbol} es {notional:.2f} USDT, menor al mínimo permitido de {min_notional:.2f} USDT.")
        return is_valid
    except Exception as e:
        print(f"❌ Error al verificar el valor notional para {symbol}: {e}")
        return False

def execute_order_buy(symbol, amount, confidence, explanation, risk_t):
    """
    Executes a market buy order and records the transaction in the database.
    """
    try:
        if not is_valid_notional(symbol, amount):
            print(f"⚠️ Orden de compra para {symbol} no válida debido al valor notional.")
            return None

        order = exchange.create_market_buy_order(symbol, amount)
        price = order.get("price") or 0
        timestamp = pd.Timestamp.now().isoformat()

        insert_transaction(
            symbol=symbol,
            action="buy",
            price=price,
            amount=amount,
            timestamp=timestamp,
            profit_loss=None,
            confidence_percentage=confidence,
            risk_type=risk_t,
            summary=explanation
        )
        print(f"✅ Orden de compra ejecutada: {symbol}, Precio: {price}, Cantidad: {amount}")
        return order
    except Exception as e:
        print(f"❌ Error al ejecutar la orden de compra para {symbol}: {e}")
        return None

def make_buy(symbol, budget, risk_type, confidence, explanation=None):
    """
    Adjusts the purchase budget based on risk type and confidence, then executes a buy order.
    After a successful buy, the trailing stop process is initiated.
    """
    if risk_type == "alto riesgo":
        adjusted_budget = budget * 0.8 if confidence > 80 else budget * 0.5
    elif risk_type == "bajo riesgo":
        if confidence <= 70:
            adjusted_budget = budget * 0.05
        elif 70 < confidence <= 75:
            adjusted_budget = budget * 0.15
        elif 75 < confidence <= 80:
            adjusted_budget = budget * 0.4
        elif confidence > 80:
            adjusted_budget = budget * 0.7
        else:
            print(f"⚠️ Confianza no válida: {confidence}")
            return
    else:
        print(f"⚠️ Tipo de riesgo no válido: {risk_type}")
        return

    print(f"🔍 Presupuesto ajustado: {adjusted_budget:.2f} USDT (Confianza: {confidence}%, Riesgo: {risk_type})")
    final_price = fetch_price(symbol)
    if not final_price:
        print(f"⚠️ No se pudo obtener el precio para {symbol}.")
        return

    amount_to_buy = adjusted_budget / final_price
    print(f"Cantidad a comprar: {amount_to_buy:.6f}")
    if amount_to_buy * final_price >= 2:
        order = execute_order_buy(symbol, amount_to_buy, confidence, explanation, risk_type)
        if order:
            timestamp = get_colombia_timestamp()
            print(f"✅ Compra ejecutada para {symbol} ({risk_type})")
            url_binance = f"https://www.binance.com/en/trade/{symbol}_USDT?_from=markets&type=spot"
            try:
                send_telegram_message(
                    f"Comprando {symbol} ({risk_type}) EXITOSAMENTE a un valor de {amount_to_buy} USDT, confianza: {confidence}%, explicación: {explanation}"
                )
                send_telegram_message(f"URL de Binance: {url_binance} - Timestamp: {timestamp}")
            except Exception as e:
                print(f"❌ Error enviando mensaje a Telegram: {e}")
            # Initiate the trailing stop process immediately after a successful purchase:
            process_order(order, symbol, exchange_instance=exchange)
        else:
            print(f"❌ No se pudo ejecutar la compra para {symbol} ({risk_type}).")
    else:
        print(f"⚠️ La cantidad calculada para {symbol} no cumple con el mínimo notional.")

# =============================================================================
# TESTING INDICATORS (Optional)
# =============================================================================

def debug_new_indicators(symbol):
    """
    Diagnostic function to test new technical indicators for a given symbol.
    """
    print(f"\n🔍 Diagnóstico de indicadores para {symbol}")
    print("=" * 50)
    try:
        data_by_timeframe, volume_series, price_series = fetch_and_prepare_data(symbol)
        if not all([data_by_timeframe, volume_series is not None, price_series is not None]):
            print("❌ Error: No se pudieron obtener los datos base")
            return
        print("\n1️⃣ Análisis de Divergencias:")
        prices = data_by_timeframe["1h"]["close"].values
        rsi_values = calculate_rsi(prices)
        divergences = detect_momentum_divergences(prices, rsi_values)
        print(f"- Últimos 5 precios: {prices[-5:]}")
        print(f"- Últimos 5 RSI: {rsi_values[-5:]}")
        print(f"- Divergencias encontradas: {divergences}")
        print("\n2️⃣ Análisis de Sentimiento:")
        # Placeholder for market sentiment analysis (if implemented)
        sentiment = {}
        print(f"- Sentimiento: {sentiment}")
        validation = {
            "Divergencias válidas": all(isinstance(d, tuple) and len(d) == 2 for d in divergences),
            "Sentimiento completo": True,
        }
        for check, result in validation.items():
            print(f"- {check}: {'✅' if result else '❌'}")
        return validation
    except Exception as e:
        print(f"❌ Error durante el diagnóstico: {e}")
        return None

def test_new_indicators():
    """Runs tests on a set of symbols to check technical indicators."""
    print("\n🧪 Iniciando pruebas de nuevos indicadores")
    test_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    results = {}
    for symbol in test_symbols:
        print(f"\nProbando {symbol}")
        print("-" * 30)
        results[symbol] = debug_new_indicators(symbol)
    return results

# =============================================================================
# MAIN TRADING FUNCTION (BUYING LOGIC WITH GPT-BASED SELECTION)
# =============================================================================

def demo_trading():
    """
    Main trading function that:
      - Checks USDT balance.
      - Selects candidate cryptos.
      - Prepares market data.
      - Uses GPT to decide whether to buy.
      - Uses GPT group selection to pick the best candidate.
      - Executes buy orders if recommended.
      - Immediately launches a trailing stop after a successful purchase.
    """
    timestamp = get_colombia_timestamp()
    print(f"Iniciando proceso de inversión con el timestamp {timestamp}")
    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
    print(f"Saldo disponible en USDT: {usdt_balance}")

    if usdt_balance <= 5:
        print("⚠️ Sin saldo suficiente en USDT. Se omite el proceso de compra.")
        return

    low_risk_budget = 100 if usdt_balance > 100 else usdt_balance
    print(f"Presupuesto de inversión bajo riesgo: {low_risk_budget} USDT")
    print("Analizando criptos de alto volumen (bajo riesgo)...")
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=24)
    data_by_symbol = {}

    for symbol in selected_cryptos:
        data_by_timeframe, volume_series, price_series = fetch_and_prepare_data(symbol)
        final_price = fetch_price(symbol)
        if data_by_timeframe and volume_series is not None and price_series is not None:
            support, resistance = calculate_support_resistance(price_series)
            adx = calculate_adx(data_by_timeframe["1h"])
            if "close" in data_by_timeframe["1h"].columns:
                prices = data_by_timeframe["1h"]["close"].values
                rsi_values = calculate_rsi(prices, period=14)
                rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else "No disponible"
                divergences = detect_momentum_divergences(prices, rsi_values)
                # Placeholder: insert your market sentiment analysis if available.
                market_sentiment = {}
            else:
                rsi = "No disponible"
                divergences = []
                market_sentiment = None

            # Placeholders for additional analyses (market depth, candlestick patterns, etc.)
            candlestick_pattern = []

            additional_data = {
                "current_price": final_price,
                "relative_volume": volume_series.iloc[-1] / volume_series.mean() if volume_series.mean() != 0 else None,
                "rsi": rsi,
                "avg_volume_24h": volume_series.tail(24).mean() if len(volume_series) >= 24 else None,
                "market_cap": None,   # Implement fetch_market_cap() if needed.
                "spread": None,       # Implement calculate_spread() if needed.
                "fear_greed": None,   # Implement fetch_fear_greed_index() if needed.
                "price_std_dev": np.std(price_series),
                "adx": adx,
                "support": support,
                "resistance": resistance,
                "market_depth_bids": None,
                "market_depth_asks": None,
                "candlestick_pattern": candlestick_pattern,
                "momentum_divergences": divergences,
                "market_sentiment": market_sentiment,
                "instruction": "Comprar si los indicadores muestran potencial de crecimiento a corto plazo."
            }
            data_by_symbol[symbol] = (data_by_timeframe, additional_data)
        else:
            print(f"⚠️ Datos insuficientes para {symbol}, se omite.")

    if data_by_symbol:
        # Use GPT-based group selection to choose the best candidate instead of simply taking the first.
        final_winner = gpt_group_selection(data_by_symbol)
        print(f"El candidato seleccionado es: {final_winner}")
            # Check if the final_winner was bought in the last 8 hours
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        eight_hours_ago = datetime.now() - timedelta(hours=8)
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM transactions
            WHERE symbol = ? AND action = 'buy' AND timestamp >= ?
            """,
            (final_winner, eight_hours_ago.strftime("%Y-%m-%d %H:%M:%S")),
        )
        recent_buys = cursor.fetchone()[0]
        conn.close()

        if recent_buys > 0:
            print(f"⚠️ {final_winner} fue comprado en las últimas 8 horas. Se omite la compra.")
            send_telegram_message(f"⚠️ {final_winner} fue comprado en las últimas 8 horas. Se omite la compra.")
            return
        winner_data_by_timeframe, winner_additional_data = data_by_symbol[final_winner]
        prepared_text = gpt_prepare_data(winner_data_by_timeframe, winner_additional_data)
        action, confidence, explanation = gpt_decision_buy(prepared_text)
        print("******************************************")
        print(f"Se recomienda {action} para {final_winner}")
        print("******************************************")
        print(f"La explicación es: {explanation}")

        if action == "comprar":
            make_buy(final_winner, low_risk_budget, "bajo riesgo", confidence, explanation)
    else:
        print("⚠️ No se encontraron criptos válidas de alto volumen.")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    demo_trading()
    test_new_indicators()  # Optional: runs indicator tests

if __name__ == "__main__":
    main()
