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
from elegir_cripto import choose_best_cryptos
import ta
import pytz
from db_manager_real import initialize_db, insert_transaction, fetch_all_transactions

# Configuración e Inicialización
load_dotenv()
GPT_MODEL = "gpt-4o-mini"
DB_NAME = "trading_real.db"
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
RSI_THRESHOLD = 70
ADX_THRESHOLD = 25
VOLUME_GROWTH_THRESHOLD = 1.0

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
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
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
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
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
    """
    return prompt

def gpt_decision_buy(prepared_text):
    prompt = f"""
    Eres un experto en trading de criptomonedas. Basándote en los datos:
    {prepared_text}
    Decide si "comprar" o "mantener". Responde SOLO con un JSON válido como este:
    {{"accion": "comprar", "confianza": 85, "explicacion": "RSI bajo y volumen creciendo"}}
    No incluyas texto adicional fuera del JSON, ni etiquetas como ```json```.
    Criterios:
    - Comprar si hay señales de crecimiento (RSI < 70, ADX > 25, volumen creciendo, divergencias alcistas).
    - Mantener si hay sobrecompra (RSI > 70) o señales débiles.
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
        
        insert_transaction(
            symbol=symbol, 
            action="buy", 
            price=price, 
            amount=amount, 
            timestamp=timestamp, 
            trade_id=trade_id,
            rsi=indicators.get('rsi'),
            adx=indicators.get('adx'),
            atr=indicators.get('atr'),
            relative_volume=indicators.get('relative_volume'),
            divergence=indicators.get('divergence'),
            bb_position=indicators.get('bb_position'),
            confidence=confidence
        )
        logging.info(f"Compra ejecutada: {symbol} a {price} por {amount} (ID: {trade_id})")
        return {"price": price, "filled": amount, "trade_id": trade_id}
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
            insert_transaction(symbol=symbol, action="sell", price=sell_price, amount=amount, timestamp=timestamp, trade_id=trade_id)
            return
        
        price = fetch_price(symbol)
        timestamp = datetime.now(timezone.utc).isoformat()
        rsi = data['1h']['RSI'].iloc[-1] if not pd.isna(data['1h']['RSI'].iloc[-1]) else None
        adx = calculate_adx(data['1h'])
        atr = data['1h']['ATR'].iloc[-1] if not pd.isna(data['1h']['ATR'].iloc[-1]) else None
        
        order = exchange.create_market_sell_order(symbol, amount)
        sell_price = order.get("price", price)
        insert_transaction(
            symbol=symbol, 
            action="sell", 
            price=sell_price, 
            amount=amount, 
            timestamp=timestamp, 
            trade_id=trade_id,
            rsi=rsi,
            adx=adx,
            atr=atr
        )
        logging.info(f"Venta ejecutada: {symbol} a {sell_price} (ID: {trade_id})")
        send_telegram_message(f"✅ *Venta Ejecutada* para `{symbol}`\nPrecio: `{sell_price}`\nCantidad: `{amount}`")
    except Exception as e:
        logging.error(f"Error al vender {symbol}: {e}")

def dynamic_trailing_stop(symbol, amount, purchase_price, trade_id):
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

            logging.info(f"Trailing stop para {symbol}: {trailing_percent}% (ID: {trade_id})")
            send_telegram_message(f"🔄 *Trailing Stop Dinámico* para `{symbol}`\nTrailing: `{trailing_percent}%`\nStop Loss: `{stop_loss_percent}%`")

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

def demo_trading():
    usdt_balance = exchange.fetch_balance()['free'].get('USDT', 0)
    if usdt_balance < MIN_NOTIONAL:
        logging.warning("Saldo insuficiente en USDT.")
        return

    daily_buys = get_daily_buys()
    if daily_buys >= MAX_DAILY_BUYS:
        logging.info("Límite diario de compras alcanzado.")
        return

    budget_per_trade = usdt_balance / (MAX_DAILY_BUYS - daily_buys)
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=24)
    data_by_symbol = {}

    for symbol in selected_cryptos:
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
        
        additional_data = {
            "current_price": current_price,
            "rsi": rsi if rsi is not None else "No disponible",
            "adx": adx if adx is not None else "No disponible",
            "atr": atr if atr is not None else "No disponible",
            "relative_volume": relative_volume if relative_volume is not None else "No disponible",
            "momentum_divergences": divergence
        }
        indicators = {
            "rsi": rsi,
            "adx": adx,
            "atr": atr,
            "relative_volume": relative_volume,
            "divergence": divergence,
            "bb_position": bb_position
        }
        
        # Filtro cuantitativo previo al LLM
        if (rsi is not None and rsi >= RSI_THRESHOLD) or \
           (adx is not None and adx < ADX_THRESHOLD) or \
           (relative_volume is not None and relative_volume < VOLUME_GROWTH_THRESHOLD):
            logging.info(f"Se omite {symbol} por no cumplir filtros cuantitativos: RSI={rsi}, ADX={adx}, Volumen Relativo={relative_volume}")
            continue
        
        data_by_symbol[symbol] = (data, additional_data, indicators)

    candidates = []
    for symbol, (data, additional_data, indicators) in data_by_symbol.items():
        prepared_text = gpt_prepare_data(data, additional_data)
        action, confidence, explanation = gpt_decision_buy(prepared_text)
        if action == "comprar" and confidence >= 70:
            candidates.append((symbol, confidence, explanation, indicators))

    candidates.sort(key=lambda x: x[1], reverse=True)
    buys_to_execute = min(MAX_DAILY_BUYS - daily_buys, len(candidates))

    for i in range(buys_to_execute):
        symbol, confidence, explanation, indicators = candidates[i]
        current_price = fetch_price(symbol)
        if current_price is None:
            logging.warning(f"No se pudo obtener precio actual para {symbol}, omitiendo compra")
            continue
        amount = budget_per_trade / current_price
        if amount * current_price >= MIN_NOTIONAL:
            order = execute_order_buy(symbol, amount, indicators, confidence)
            if order:
                logging.info(f"Compra ejecutada para {symbol}: {explanation}")
                send_telegram_message(f"✅ *Compra* `{symbol}`\nConfianza: `{confidence}%`\nExplicación: `{explanation}`")
                dynamic_trailing_stop(symbol, order['filled'], order['price'], order['trade_id'])

    generate_profit_loss_table()

if __name__ == "__main__":
    demo_trading()