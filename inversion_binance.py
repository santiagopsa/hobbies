import ccxt
import pandas as pd
import numpy as np
import sqlite3
import os
import time
import requests
import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from ta.trend import MACD
import pytz
from elegir_cripto import choose_best_cryptos
# Configuración e Inicialización
load_dotenv()
GPT_MODEL = "gpt-4o-mini"
DB_NAME = "trading_real.db"

# Inicializar la base de datos (crear tabla si no existe)
def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            timestamp TEXT NOT NULL,
            trade_id TEXT NOT NULL,
            rsi REAL,
            adx REAL,
            atr REAL,
            relative_volume REAL,
            divergence TEXT,
            bb_position TEXT,
            confidence REAL
        )
    ''')
    conn.commit()
    conn.close()

initialize_db()

exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY_REAL"),
    'secret': os.getenv("BINANCE_SECRET_KEY_REAL"),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(
    level=logging.INFO,
    filename="trading_real.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constantes
MAX_DAILY_BUYS = 5
MIN_NOTIONAL = 10
RSI_THRESHOLD = 65
ADX_THRESHOLD = 25
VOLUME_GROWTH_THRESHOLD = 1.0

# Cache de decisiones
decision_cache = {}
CACHE_EXPIRATION = 1800  # 30 minutos

# Funciones Utilitarias
def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram no configurado.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        logging.error(f"Error al enviar a Telegram: {e}")

def get_colombia_timestamp():
    colombia_tz = pytz.timezone("America/Bogota")
    return datetime.now(colombia_tz).strftime("%Y-%m-%d %H:%M:%S")

def fetch_price(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error al obtener precio de {symbol}: {e}")
        return None

def fetch_volume(symbol):
    try:
        ticker = exchange.fetch_ticker(symbol)
        return ticker['quoteVolume']
    except Exception as e:
        logging.error(f"Error al obtener volumen de {symbol}: {e}")
        return None

def fetch_and_prepare_data(symbol):
    try:
        timeframes = ['1h', '4h', '1d']
        data = {}
        for tf in timeframes:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=50)
            if not ohlcv:
                logging.warning(f"No se obtuvieron datos OHLCV para {symbol} en {tf}")
                return None, None, None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['RSI'] = pd.Series(np.nan_to_num(pd.Series([talib.RSI(df['close'].values, timeperiod=14)]).iloc[0]))
            df['ATR'] = pd.Series(np.nan_to_num(pd.Series([talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)]).iloc[0]))
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            df['BB_upper'] = bb_upper
            df['BB_lower'] = bb_lower
            macd = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            data[tf] = df
        volume_series = data['1h']['volume']
        price_series = data['1h']['close']
        if volume_series.empty or price_series.empty:
            logging.warning(f"Datos vacíos para {symbol}: volumen o precios no disponibles")
            return None, None, None
        return data, volume_series, price_series
    except Exception as e:
        logging.error(f"Error al obtener datos de {symbol}: {e}")
        return None, None, None

def calculate_adx(df):
    try:
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else None
    except Exception as e:
        logging.error(f"Error al calcular ADX: {e}")
        return None

def detect_momentum_divergences(price_series, rsi_series):
    try:
        price = np.array(price_series)
        rsi = np.array(rsi_series)
        divergences = []
        window = 5
        for i in range(window, len(price) - window):
            if (price[i] > max(price[i-window:i]) and price[i] > max(price[i+1:i+window+1]) and
                rsi[i] < rsi[i-window]):
                divergences.append(("bearish", i))
            elif (price[i] < min(price[i-window:i]) and price[i] < min(price[i+1:i+window+1]) and
                  rsi[i] > rsi[i-window]):
                divergences.append(("bullish", i))
        return "bullish" if any(d[0] == "bullish" for d in divergences) else "bearish" if any(d[0] == "bearish" for d in divergences) else "none"
    except Exception as e:
        logging.error(f"Error en divergencias: {e}")
        return "none"

def get_bb_position(price, bb_upper, bb_lower):
    if price is None or bb_upper is None or bb_lower is None:
        return "unknown"
    if price > bb_upper:
        return "above_upper"
    elif price < bb_lower:
        return "below_lower"
    else:
        return "between"

def has_recent_macd_crossover(macd_series, signal_series, lookback=5):
    """
    Verifica si ha habido un cruce alcista de MACD en las últimas 'lookback' velas.
    """
    for i in range(-1, -lookback-1, -1):
        if i < -len(macd_series):
            break
        if macd_series.iloc[i-1] <= signal_series.iloc[i-1] and macd_series.iloc[i] > signal_series.iloc[i]:
            return True, abs(i)
    return False, None

# Funciones de GPT
def gpt_prepare_data(data_by_timeframe, additional_data):
    combined_data = ""
    for tf, df in data_by_timeframe.items():
        if not df.empty:
            combined_data += f"\nDatos de {tf} (últimos 3):\n{df.tail(3).to_string(index=False)}\n"
    prompt = f"""
    Analiza los siguientes datos y decide si comprar esta criptomoneda:
    {combined_data}
    Indicadores adicionales:
    - RSI: {additional_data.get('rsi', 'No disponible')}
    - ADX: {additional_data.get('adx', 'No disponible')}
    - Divergencias: {additional_data.get('momentum_divergences', 'No disponible')}
    - Volumen relativo: {additional_data.get('relative_volume', 'No disponible')}
    - Precio actual: {additional_data.get('current_price', 'No disponible')}
    - Cruce alcista reciente de MACD: {additional_data.get('macd_crossover', 'No disponible')}
    - Velas desde el cruce: {additional_data.get('candles_since_crossover', 'No disponible')}
    """
    return prompt

def gpt_decision_buy(prepared_text):
    prompt = f"""
    Eres un experto en trading de criptomonedas. Basándote en los datos:
    {prepared_text}
    Decide si "comprar" o "mantener". Responde SOLO con un JSON válido como este:
    {{"accion": "comprar", "confianza": 85, "explicacion": "Cruce alcista de MACD reciente y RSI bajo"}}
    Criterios:
    - Prioriza los cruces alcistas recientes de MACD como una señal fuerte de compra, especialmente si están acompañados de RSI < 65, ADX > 25, y volumen relativo > 1.0.
    - Comprar si hay señales de crecimiento combinadas con el cruce de MACD.
    - Mantener si no hay cruce alcista reciente o si hay señales de sobrecompra (RSI > 70).
    """
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw_response = response.choices[0].message.content.strip()
            decision = json.loads(raw_response)
            
            accion = decision.get("accion", "mantener").lower()
            confianza = decision.get("confianza", 50)
            explicacion = decision.get("explicacion", "Respuesta incompleta")

            if accion not in ["comprar", "mantener"]:
                accion = "mantener"
            if not isinstance(confianza, (int, float)) or not 0 <= confianza <= 100:
                confianza = 50
                explicacion = "Confianza inválida, ajustada a 50"

            return accion, confianza, explicacion
        except json.JSONDecodeError as e:
            logging.error(f"Intento {attempt + 1} fallido: Respuesta de GPT no es JSON válido - {raw_response}")
            if attempt == max_retries:
                return "mantener", 50, f"Error en formato JSON tras {max_retries + 1} intentos"
        except Exception as e:
            logging.error(f"Error en GPT (intento {attempt + 1}): {e}")
            if attempt == max_retries:
                return "mantener", 50, "Error al procesar respuesta de GPT"
        time.sleep(1)

# Lógica de Trading
def get_daily_buys():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) FROM transactions_new WHERE action='buy' AND timestamp LIKE ?", (f"{today}%",))
    count = cursor.fetchone()[0]
    conn.close()
    return count

def execute_order_buy(symbol, amount, indicators, confidence):
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        price = order.get("price", fetch_price(symbol))
        if price is None:
            logging.error(f"No se pudo obtener precio para {symbol} después de la orden")
            return None
        timestamp = datetime.now(timezone.utc).isoformat()
        trade_id = f"{symbol}_{timestamp.replace(':', '-')}"
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions_new (symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr, relative_volume, divergence, bb_position, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, "buy", price, amount, timestamp, trade_id,
            indicators.get('rsi'), indicators.get('adx'), indicators.get('atr'),
            indicators.get('relative_volume'), indicators.get('divergence'), indicators.get('bb_position'), confidence
        ))
        conn.commit()
        conn.close()
        
        logging.info(f"Compra ejecutada: {symbol} a {price} por {amount} (ID: {trade_id})")
        return {"price": price, "filled": amount, "trade_id": trade_id, "indicators": indicators}
    except Exception as e:
        logging.error(f"Error al comprar {symbol}: {e}")
        return None

def sell_symbol(symbol, amount, trade_id):
    try:
        data, volume_series, price_series = fetch_and_prepare_data(symbol)
        if data is None:
            logging.error(f"No se pudieron obtener datos para vender {symbol}")
            price = fetch_price(symbol)
            timestamp = datetime.now(timezone.utc).isoformat()
            order = exchange.create_market_sell_order(symbol, amount)
            sell_price = order.get("price", price)
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transactions_new (symbol, action, price, amount, timestamp, trade_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, "sell", sell_price, amount, timestamp, trade_id))
            conn.commit()
            conn.close()
            return
        
        price = fetch_price(symbol)
        timestamp = datetime.now(timezone.utc).isoformat()
        rsi = data['1h']['RSI'].iloc[-1] if not pd.isna(data['1h']['RSI'].iloc[-1]) else None
        adx = calculate_adx(data['1h'])
        atr = data['1h']['ATR'].iloc[-1] if not pd.isna(data['1h']['ATR'].iloc[-1]) else None
        
        order = exchange.create_market_sell_order(symbol, amount)
        sell_price = order.get("price", price)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transactions_new (symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, "sell", sell_price, amount, timestamp, trade_id, rsi, adx, atr))
        conn.commit()
        conn.close()
        
        logging.info(f"Venta ejecutada: {symbol} a {sell_price} (ID: {trade_id})")
        send_telegram_message(f"✅ *Venta Ejecutada* para `{symbol}`\nPrecio: `{sell_price}`\nCantidad: `{amount}`")
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")

def dynamic_trailing_stop(symbol, amount, purchase_price, trade_id, entry_indicators):
    def trailing_logic():
        try:
            data, _, price_series = fetch_and_prepare_data(symbol)
            if data is None:
                logging.error(f"No se pudieron obtener datos iniciales para trailing stop de {symbol}")
                sell_symbol(symbol, amount, trade_id)
                return
            
            atr = data['1h']['ATR'].iloc[-1] if not pd.isna(data['1h']['ATR'].iloc[-1]) else 0
            volatility = atr / purchase_price * 100 if purchase_price else 0
            trailing_percent = max(2, min(5, volatility * 1.5))
            stop_loss_percent = trailing_percent * 0.5
            highest_price = purchase_price
            activated = False

            entry_msg = (
                f"🔄 *Trailing Stop Dinámico Iniciado* para `{symbol}`\n"
                f"Precio de Compra: `{purchase_price}`\n"
                f"Cantidad: `{amount}`\n"
                f"Trailing Percent: `{trailing_percent:.2f}%`\n"
                f"Stop Loss Inicial: `{stop_loss_percent:.2f}%`\n"
                f"RSI Inicial: `{entry_indicators.get('rsi', 'N/A')}`\n"
                f"ADX Inicial: `{entry_indicators.get('adx', 'N/A')}`\n"
                f"ATR Inicial: `{entry_indicators.get('atr', 'N/A')}`\n"
                f"Volumen Relativo: `{entry_indicators.get('relative_volume', 'N/A')}`\n"
                f"Divergencia: `{entry_indicators.get('divergence', 'N/A')}`\n"
                f"Posición BB: `{entry_indicators.get('bb_position', 'N/A')}`\n"
                f"MACD: `{entry_indicators.get('macd', 'N/A')}`\n"
                f"MACD Signal: `{entry_indicators.get('macd_signal', 'N/A')}`"
            )
            logging.info(entry_msg)
            send_telegram_message(entry_msg)

            while True:
                current_price = fetch_price(symbol)
                current_data, _, _ = fetch_and_prepare_data(symbol)
                if not current_price or not current_data:
                    logging.warning(f"Datos no disponibles para {symbol}, reintentando en 5s")
                    time.sleep(5)
                    continue

                if not activated and current_price < purchase_price * (1 - stop_loss_percent / 100):
                    sell_symbol(symbol, amount, trade_id)
                    logging.info(f"Stop loss ejecutado para {symbol} a {current_price} (ID: {trade_id})")
                    break

                if current_price >= purchase_price * 1.01:
                    activated = True

                if activated:
                    if current_price > highest_price:
                        highest_price = current_price
                        logging.info(f"Nuevo máximo para {symbol}: {highest_price} (ID: {trade_id})")
                    stop_price = highest_price * (1 - trailing_percent / 100)
                    if current_price < stop_price:
                        sell_symbol(symbol, amount, trade_id)
                        logging.info(f"Trailing stop ejecutado para {symbol} a {current_price} (ID: {trade_id})")
                        break
                time.sleep(5)
        except Exception as e:
            logging.error(f"Error en trailing stop de {symbol}: {e}")
            sell_symbol(symbol, amount, trade_id)

    threading.Thread(target=trailing_logic, daemon=True).start()

def generate_profit_loss_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr, 
               relative_volume, divergence, bb_position, confidence 
        FROM transactions_new 
        WHERE trade_id IS NOT NULL 
        ORDER BY trade_id, timestamp
    """)
    transactions = cursor.fetchall()
    conn.close()

    trades = {}
    for row in transactions:
        symbol, action, price, amount, timestamp, trade_id, rsi, adx, atr, rel_vol, div, bb_pos, conf = row
        if trade_id not in trades:
            trades[trade_id] = {"buy": None, "sell": None}
        if action == "buy":
            trades[trade_id]["buy"] = {
                "price": price, "amount": amount, "timestamp": timestamp, "rsi": rsi, "adx": adx, 
                "atr": atr, "relative_volume": rel_vol, "divergence": div, "bb_position": bb_pos, 
                "confidence": conf
            }
        elif action == "sell":
            trades[trade_id]["sell"] = {
                "price": price, "amount": amount, "timestamp": timestamp, "rsi": rsi, "adx": adx, "atr": atr
            }

    completed_trades = []
    for trade_id, data in trades.items():
        if data["buy"] and data["sell"]:
            buy_data = data["buy"]
            sell_data = data["sell"]
            symbol = trade_id.split('_')[0]
            buy_price = buy_data["price"]
            sell_price = sell_data["price"]
            amount = min(buy_data["amount"], sell_data["amount"])
            profit_loss = (sell_price - buy_price) * amount
            profit_percent = (sell_price - buy_price) / buy_price * 100 if buy_price else 0
            buy_time = pd.to_datetime(buy_data["timestamp"])
            sell_time = pd.to_datetime(sell_data["timestamp"])
            trade_duration = (sell_time - buy_time).total_seconds() / 60
            trailing_percent = max(2, min(5, (buy_data["atr"] / buy_price * 100) * 1.5)) if buy_data["atr"] and buy_price else 3

            completed_trades.append({
                "trade_id": trade_id,
                "symbol": symbol,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "amount": amount,
                "profit_loss": profit_loss,
                "profit_percent": profit_percent,
                "buy_time": buy_data["timestamp"],
                "sell_time": sell_data["timestamp"],
                "rsi_buy": buy_data["rsi"],
                "adx_buy": buy_data["adx"],
                "atr_buy": buy_data["atr"],
                "relative_volume_buy": buy_data["relative_volume"],
                "divergence_buy": buy_data["divergence"],
                "bb_position_buy": buy_data["bb_position"],
                "confidence": buy_data["confidence"],
                "rsi_sell": sell_data["rsi"],
                "adx_sell": sell_data["adx"],
                "atr_sell": sell_data["atr"],
                "trailing_percent": trailing_percent,
                "trade_duration": trade_duration
            })

    if not completed_trades:
        logging.info("No hay operaciones completadas para mostrar.")
        return

    df = pd.DataFrame(completed_trades)
    df = df[[
        "trade_id", "symbol", "buy_price", "sell_price", "amount", "profit_loss", "profit_percent",
        "buy_time", "sell_time", "rsi_buy", "adx_buy", "atr_buy", "relative_volume_buy", 
        "divergence_buy", "bb_position_buy", "confidence", "rsi_sell", "adx_sell", "atr_sell",
        "trailing_percent", "trade_duration"
    ]]
    
    print("\n=== Tabla de Ganancias/Pérdidas ===")
    print(df.to_string(index=False))
    logging.info("Tabla de ganancias/pérdidas generada:")
    logging.info(df.to_string(index=False))
    
    summary = f"📊 *Resumen de Operaciones*\nGanadoras: {len(df[df['profit_loss'] > 0])}\nPerdedoras: {len(df[df['profit_loss'] < 0])}\nTotal P/L: {df['profit_loss'].sum():.2f} USDT"
    send_telegram_message(summary)
    
    df.to_csv("trade_results.csv", index=False)
    logging.info("Resultados exportados a trade_results.csv")

def get_cached_decision(symbol, current_indicators):
    if symbol in decision_cache:
        cached_time, cached_decision, cached_indicators = decision_cache[symbol]
        if time.time() - cached_time < CACHE_EXPIRATION:
            if all(abs(current_indicators.get(key, 0) - cached_indicators.get(key, 0)) / cached_indicators.get(key, 1) < 0.05 for key in current_indicators):
                return cached_decision
    return None

def demo_trading(high_volume_symbols=None):
    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
    if usdt_balance < MIN_NOTIONAL:
        logging.warning("Saldo insuficiente en USDT.")
        return

    reserve = 0.10 * usdt_balance
    available_for_trading = usdt_balance - reserve

    daily_buys = get_daily_buys()
    if daily_buys >= MAX_DAILY_BUYS:
        logging.info("Límite diario de compras alcanzado.")
        return
    if high_volume_symbols is None:
        high_volume_symbols = choose_best_cryptos(exchange, 100)
        
    budget_per_trade = available_for_trading / (MAX_DAILY_BUYS - daily_buys)
    selected_cryptos = high_volume_symbols
    data_by_symbol = {}

    balance = exchange.fetch_balance()['free']

    for symbol in selected_cryptos:
        base_asset = symbol.split('/')[0]
        if base_asset in balance and balance[base_asset] > 0:
            logging.info(f"Se omite {symbol} porque ya tienes una posición abierta.")
            continue

        daily_volume = fetch_volume(symbol)
        if daily_volume is None or daily_volume < 1000000:
            logging.info(f"Se omite {symbol} por volumen insuficiente: {daily_volume}")
            continue

        data, volume_series, price_series = fetch_and_prepare_data(symbol)
        if not data or volume_series is None or price_series is None:
            logging.warning(f"Se omite {symbol} por datos insuficientes")
            continue

        current_price = fetch_price(symbol)
        if current_price is None:
            logging.warning(f"No se pudo obtener precio para {symbol}")
            continue

        rsi = data['1h']['RSI'].iloc[-1] if not pd.isna(data['1h']['RSI'].iloc[-1]) else None
        adx = calculate_adx(data['1h'])
        atr = data['1h']['ATR'].iloc[-1] if not pd.isna(data['1h']['ATR'].iloc[-1]) else None
        relative_volume = volume_series.iloc[-1] / volume_series.mean() if volume_series.mean() != 0 else None
        divergence = detect_momentum_divergences(price_series, data['1h']['RSI'])
        bb_position = get_bb_position(current_price, data['1h']['BB_upper'].iloc[-1], data['1h']['BB_lower'].iloc[-1])
        macd = data['1h']['MACD'].iloc[-1]
        macd_signal = data['1h']['MACD_signal'].iloc[-1]

        macd_series = data['1h']['MACD']
        signal_series = data['1h']['MACD_signal']
        has_crossover, candles_since = has_recent_macd_crossover(macd_series, signal_series, lookback=5)

        additional_data = {
            "current_price": current_price,
            "rsi": rsi if rsi is not None else "No disponible",
            "adx": adx if adx is not None else "No disponible",
            "atr": atr if atr is not None else "No disponible",
            "relative_volume": relative_volume if relative_volume is not None else "No disponible",
            "momentum_divergences": divergence,
            "macd_crossover": "Sí" if has_crossover else "No",
            "candles_since_crossover": candles_since if has_crossover else "N/A"
        }
        indicators = {
            "rsi": rsi,
            "adx": adx,
            "atr": atr,
            "relative_volume": relative_volume,
            "divergence": divergence,
            "bb_position": bb_position,
            "macd": macd,
            "macd_signal": macd_signal,
            "has_macd_crossover": has_crossover,
            "candles_since_crossover": candles_since
        }

        if (rsi is None or rsi >= RSI_THRESHOLD) or \
           (adx is None or adx < ADX_THRESHOLD) or \
           (relative_volume is None or relative_volume < VOLUME_GROWTH_THRESHOLD) or \
           (not has_crossover and macd <= macd_signal and not (macd > macd_signal and macd_signal > 0)):
            logging.info(f"Se omite {symbol} por no cumplir filtros cuantitativos: RSI={rsi}, ADX={adx}, Volumen Relativo={relative_volume}, MACD={macd} vs Signal={macd_signal}, Cruce MACD={'Sí' if has_crossover else 'No'}")
            continue

        data_by_symbol[symbol] = (data, additional_data, indicators)

    candidates = sorted(data_by_symbol.items(), key=lambda x: x[1][2]['rsi'])[:5]
    buys_to_execute = min(MAX_DAILY_BUYS - daily_buys, len(candidates))

    for symbol, (data, additional_data, indicators) in candidates:
        current_price = fetch_price(symbol)
        if current_price is None:
            logging.warning(f"No se pudo obtener precio actual para {symbol}, omitiendo compra")
            continue

        cached_decision = get_cached_decision(symbol, indicators)
        if cached_decision:
            action, confidence, explanation = cached_decision
        else:
            if indicators['rsi'] is not None and indicators['rsi'] < 30 and indicators['adx'] is not None and indicators['adx'] > 35:
                action = "comprar"
                confidence = 90
                explanation = "RSI muy bajo y ADX alto"
            else:
                prepared_text = gpt_prepare_data(data, additional_data)
                action, confidence, explanation = gpt_decision_buy(prepared_text)
                decision_cache[symbol] = (time.time(), (action, confidence, explanation), indicators.copy())

        if action == "comprar" and confidence >= 70:
            amount = budget_per_trade / current_price
            if amount * current_price >= MIN_NOTIONAL:
                order = execute_order_buy(symbol, amount, indicators, confidence)
                if order:
                    logging.info(f"Compra ejecutada para {symbol}: {explanation}")
                    send_telegram_message(f"✅ *Compra* `{symbol}`\nConfianza: `{confidence}%`\nExplicación: `{explanation}`")
                    dynamic_trailing_stop(symbol, order['filled'], order['price'], order['trade_id'], order['indicators'])

    generate_profit_loss_table()
    logging.info("Trading ejecutado correctamente")

if __name__ == "__main__":
    high_volume_symbols = choose_best_cryptos(base_currency="USDT", top_n=100)
    demo_trading(high_volume_symbols)