import ccxt
import os
import pandas as pd
import numpy as np
import time
import logging
import logging.handlers
import requests
import sqlite3
from ta.momentum import RSIIndicator
from ta.trend import MACD
from datetime import datetime

# --- Configuration ---
MIN_NOTIONAL = 10          # USDT per trade
TP_PCT = 0.03              # Take profit: +3%
SL_PCT = 0.02              # Stop loss: -2%
MAX_HOLD_MINUTES = 30      # Timeout exit
MAX_TRADES = 2             # Max trades per run

# --- Exchange (Binance Futures) ---
exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY_REAL"),
    'secret': os.getenv("BINANCE_SECRET_KEY_REAL"),
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True}
})

# --- Logging Setup ---
log_path = os.path.expanduser("~/hobbies/breakout_trades.log")
logger = logging.getLogger("breakout_logger")
logger.setLevel(logging.DEBUG)
handler = logging.handlers.TimedRotatingFileHandler(
    log_path, when='midnight', interval=1, backupCount=30, utc=True
)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# --- Telegram Setup ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, json=payload)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# --- SQLite DB Setup ---
db_path = os.path.expanduser("~/hobbies/breakout_trades.db")

def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            side TEXT,
            qty REAL,
            entry_price REAL,
            exit_price REAL,
            pnl REAL,
            result TEXT,
            entry_time TEXT,
            exit_time TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_trade_to_db(symbol, side, qty, entry_price, exit_price, pnl, result, entry_time, exit_time):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades (
            symbol, side, qty, entry_price, exit_price, pnl, result, entry_time, exit_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, side, qty, entry_price, exit_price, pnl, result, entry_time.isoformat(), exit_time.isoformat()))
    conn.commit()
    conn.close()

# --- Symbol & Market Helpers ---
def fetch_symbols():
    markets = exchange.load_markets()
    return [
        symbol for symbol, meta in markets.items()
        if meta.get('contract') is True
        and meta.get('future') is True
        and '/USDT' in symbol
        and symbol.endswith(':USDT')
        and '1000' not in symbol
        and 'DOWN' not in symbol
        and 'UP' not in symbol
    ]


def fetch_ohlcv(symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=150)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.warning(f"[FETCH ERROR] {symbol}: {e}")
        return None

# --- Signal Logic ---
def check_breakout_conditions(df):
    if len(df) < 50:
        return False

    recent_vol = df['volume'].iloc[-1]
    avg_vol = df['volume'].iloc[-20:-1].mean()
    vol_spike = recent_vol > 2.5 * avg_vol

    rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]
    momentum = rsi > 65

    macd_line = MACD(df['close']).macd_diff().iloc[-1]
    bullish = macd_line > 0

    return vol_spike and momentum and bullish

# --- Monitor and Close Trade ---
def monitor_trade(symbol, entry_price, quantity):
    logger.info(f"⏱️ Monitoring {symbol}...")
    entry_time = datetime.utcnow()

    while True:
        time.sleep(60)
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            change_pct = (current_price - entry_price) / entry_price

            if change_pct >= TP_PCT:
                result = "TP"
            elif change_pct <= -SL_PCT:
                result = "SL"
            elif (datetime.utcnow() - entry_time).seconds > MAX_HOLD_MINUTES * 60:
                result = "Timeout"
            else:
                logger.info(f"[{symbol}] Δ: {change_pct:.4f}, price: {current_price}")
                continue

            # Close position
            close_order = exchange.create_market_sell_order(symbol, quantity)
            pnl = quantity * (current_price - entry_price)
            exit_time = datetime.utcnow()

            report = (
                f"📉 TRADE CLOSED ({result})\n"
                f"{symbol}\nQty: {quantity}\n"
                f"Entry: {entry_price:.4f}\nExit: {current_price:.4f}\n"
                f"PnL: ${pnl:.2f}\nTime: {exit_time}"
            )
            logger.info(report)
            send_telegram(report)

            # Log to DB
            log_trade_to_db(
                symbol=symbol,
                side="LONG",
                qty=quantity,
                entry_price=entry_price,
                exit_price=current_price,
                pnl=pnl,
                result=result,
                entry_time=entry_time,
                exit_time=exit_time
            )
            break

        except Exception as e:
            logger.error(f"[MONITOR ERROR] {symbol}: {e}")
            continue

# --- Open Trade ---
def open_trade(symbol, leverage=10):
    try:
        exchange.set_leverage(leverage, symbol)
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        quantity = round(MIN_NOTIONAL / price, 3)

        if quantity <= 0:
            logger.warning(f"[SKIP] {symbol} quantity too low at price {price}")
            return False

        order = exchange.create_market_buy_order(symbol, quantity)

        msg = (
            f"✅ TRADE OPENED\n"
            f"{symbol} LONG\nQty: {quantity}\n"
            f"Price: {price:.4f}\nNotional: ${quantity * price:.2f}\n"
            f"Time: {datetime.utcnow()}"
        )
        logger.info(msg)
        send_telegram(msg)

        monitor_trade(symbol, price, quantity)
        return True

    except Exception as e:
        logger.error(f"[TRADE ERROR] {symbol}: {e}")
        send_telegram(f"❌ Failed to open trade for {symbol}\nError: {e}")
        return False

# --- Main Runner ---
def main():
    init_db()
    logger.info("🟢 Bot started.")
    send_telegram("🤖 Breakout bot scanning for trades...")

    symbols = fetch_symbols()
    print("Sample symbols:", symbols[:5])
    logger.info(f"{len(symbols)} symbols loaded.")

    trades_executed = 0
    for symbol in symbols[:50]:
        if trades_executed >= MAX_TRADES:
            break

        df = fetch_ohlcv(symbol)
        if df is None: continue

        if check_breakout_conditions(df):
            logger.info(f"[SIGNAL] {symbol} breakout detected!")
            opened = open_trade(symbol)
            if opened:
                trades_executed += 1
                logger.info(f"{symbol} opened. Total trades: {trades_executed}")

if __name__ == "__main__":
    main()
