# Full upgraded futures trading bot with SL, TP, trailing stop, multi-timeframe filter, Telegram, SQLite logging

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
from pathlib import Path
from dotenv import load_dotenv

# === Configuration ===
MIN_NOTIONAL = 10
TP_PCT = 0.03
SL_PCT = 0.02
TRAILING_SL_PCT = 0.015
MAX_HOLD_MINUTES = 30
MAX_TRADES = 2
TRADE_COOLDOWN_MINUTES = 30
DRY_RUN = False  # Set True for simulation

load_dotenv()
# === Exchange Setup ===
exchange = ccxt.binanceusdm({
    'apiKey': os.getenv("BINANCE_API_KEY_REAL"),
    'secret': os.getenv("BINANCE_SECRET_KEY_REAL"),
    'enableRateLimit': True,
    'options': {'adjustForTimeDifference': True}
})

# === Logging Setup ===
log_path = os.path.expanduser("~/hobbies/breakout_trades.log")
logger = logging.getLogger("breakout_logger")
logger.setLevel(logging.DEBUG)
handler = logging.handlers.TimedRotatingFileHandler(log_path, when='midnight', interval=1, backupCount=30, utc=True)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# === Telegram Setup ===
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram no configurado.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=3)
    except Exception as e:
        logger.error(f"Error al enviar a Telegram: {e}")
# === SQLite Setup ===
db_path = os.path.expanduser("~/hobbies/breakout_trades.db")
cooldowns = {}

def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, side TEXT, qty REAL,
            entry_price REAL, exit_price REAL,
            pnl REAL, result TEXT,
            entry_time TEXT, exit_time TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_trade_to_db(symbol, side, qty, entry_price, exit_price, pnl, result, entry_time, exit_time):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        INSERT INTO trades (
            symbol, side, qty, entry_price, exit_price,
            pnl, result, entry_time, exit_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, side, qty, entry_price, exit_price, pnl, result,
          entry_time.isoformat(), exit_time.isoformat()))
    conn.commit()
    conn.close()

# === Market Helpers ===
def fetch_symbols():
    markets = exchange.load_markets(True)
    symbols = []

    for symbol, meta in markets.items():
        if (
            meta.get('contract') and
            meta.get('linear') and
            meta.get('expiry') is None
        ):
            vol = float(meta.get('info', {}).get('quoteVolume', 0))
            if vol > 0:
                symbols.append((symbol, vol))

    symbols.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s[0] for s in symbols[:30]]
    print(f"🔁 Fetched {len(top_symbols)} symbols:", top_symbols[:5])  # TEMP DEBUG
    return top_symbols


def fetch_ohlcv(symbol, timeframe='5m'):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=150)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.warning(f"[FETCH ERROR] {symbol}: {e}")
        return None

# === Signal Logic ===
def check_breakout_conditions(df):
    if len(df) < 50:
        return False
    vol_spike = df['volume'].iloc[-1] > 2.5 * df['volume'].iloc[-20:-1].mean()
    rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]
    momentum = rsi > 65
    macd_line = MACD(df['close']).macd_diff().iloc[-1]
    bullish = macd_line > 0
    return vol_spike and momentum and bullish

# === Trade Monitoring ===
def monitor_trade(symbol, entry_price, quantity):
    logger.info(f"⏱️ Monitoring {symbol}...")
    entry_time = datetime.utcnow()
    peak_price = entry_price

    while True:
        time.sleep(60)
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            if current_price > peak_price:
                peak_price = current_price

            change_pct = (current_price - entry_price) / entry_price
            drop_from_peak = (current_price - peak_price) / peak_price

            result = None
            if change_pct >= TP_PCT:
                result = "TP"
            elif change_pct <= -SL_PCT:
                result = "SL"
            elif drop_from_peak <= -TRAILING_SL_PCT:
                result = "Trailing SL"
            elif (datetime.utcnow() - entry_time).seconds > MAX_HOLD_MINUTES * 60:
                result = "Timeout"

            if result:
                if not DRY_RUN:
                    exchange.create_market_sell_order(symbol, quantity)
                pnl = quantity * (current_price - entry_price)
                exit_time = datetime.utcnow()
                log_trade_to_db(symbol, "LONG", quantity, entry_price, current_price, pnl, result, entry_time, exit_time)
                send_telegram(f"📉 TRADE CLOSED ({result}) {symbol} PnL: ${pnl:.2f}")
                break

            logger.info(f"[{symbol}] Δ: {change_pct:.4f}, Peak drop: {drop_from_peak:.4f}")
        except Exception as e:
            logger.error(f"[MONITOR ERROR] {symbol}: {e}")

# === Trade Execution ===
def open_trade(symbol, leverage=10):
    try:
        if symbol in cooldowns and (datetime.utcnow() - cooldowns[symbol]).seconds < TRADE_COOLDOWN_MINUTES * 60:
            return False

        exchange.set_leverage(leverage, symbol)
        balance = exchange.fetch_balance()
        usdt_amount = balance['total']['USDT'] * 0.01
        price = exchange.fetch_ticker(symbol)['last']
        quantity = round(max(usdt_amount, MIN_NOTIONAL) / price, 3)
        if quantity <= 0:
            return False

        if not DRY_RUN:
            exchange.create_market_buy_order(symbol, quantity)

        cooldowns[symbol] = datetime.utcnow()
        send_telegram(f"✅ TRADE OPENED {symbol} | Qty: {quantity} | Price: {price:.4f}")
        monitor_trade(symbol, price, quantity)
        return True
    except Exception as e:
        logger.error(f"[TRADE ERROR] {symbol}: {e}")
        send_telegram(f"❌ Trade error {symbol}: {e}")
        return False

# === PnL Report ===
def report_pnl():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()
    if df.empty:
        send_telegram("No trades yet.")
    else:
        send_telegram(f"📊 Total PnL: ${df['pnl'].sum():.2f} | Trades: {len(df)}")

# === Main Loop ===
def main():
    init_db()
    send_telegram("🤖 Upgraded Futures Bot activated.")
    while True:
        if Path("stop_bot.txt").exists():
            send_telegram("🛑 Manual stop triggered.")
            break
        try:
            logger.info("🚀 Starting new scan...")
            symbols = fetch_symbols()
            trades_executed = 0
            total_scanned = 0
            signals_found = 0
            skipped_due_to_rsi = 0

            for symbol in symbols:
                print(f"🔍 Checking {symbol}")
                if trades_executed >= MAX_TRADES:
                    break
                df = fetch_ohlcv(symbol)
                df_1h = fetch_ohlcv(symbol, timeframe='1h')
                if df is None or df_1h is None:
                    logger.warning(f"⚠️ Skipping {symbol} due to missing OHLCV")
                    continue

                rsi_1h = RSIIndicator(df_1h['close'], window=14).rsi().iloc[-1]
                total_scanned += 1
                if rsi_1h < 50:
                    skipped_due_to_rsi += 1
                    logger.info(f"❌ {symbol} skipped (1h RSI too low: {rsi_1h:.2f})")
                    continue

                if check_breakout_conditions(df):
                    signals_found += 1
                    print(f"✅ Breakout detected on {symbol}")
                    send_telegram(f"🚨 Breakout signal: {symbol}")
                    if open_trade(symbol):
                        trades_executed += 1
                        logger.info(f"✅ Trade executed on {symbol}")
                else:
                    logger.info(f"⬛ No breakout signal for {symbol}")

            summary = (
                f"📊 Cycle Summary\n"
                f"Scanned: {total_scanned} symbols\n"
                f"Signals found: {signals_found}\n"
                f"Trades executed: {trades_executed}\n"
                f"Skipped (low RSI): {skipped_due_to_rsi}\n"
                f"🕒 Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
            )
            print(summary)
            send_telegram(summary)
            logger.info("🛌 Sleeping for 5 min...")
            time.sleep(5 * 60)

        except Exception as e:
            logger.exception("💥 Error in main loop")
            send_telegram(f"❌ Error in main loop:\n{e}")
            time.sleep(30)


if __name__ == "__main__":
    main()
