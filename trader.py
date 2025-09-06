#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Crypto Spot Trader (Binance via ccxt)

Decision (1h timeframe):
- Hard guards: 24h quoteVolume >= 5M USDT, percent spread <= 0.5%
- Uptrend lane: EMA20 > EMA50 and close > EMA20
- Flexible score (volume-first):
    * RVOL10 tiers (>=1.5 / >=2.0 / >=3.0)
    * Price slope(10) > 0
    * RSI(14) in [45, 72] (+ extra if >= 60)
    * ADX(14) >= 20 (+ extra if >= 28)
    * Volume slope(10) > 0 (minor nudge)

Execution:
- Position size: fixed fraction of USDT, nudged by confidence
- Exit: ATR-aware trailing stop: trail% = max(1.5*ATR%, 2%) capped at 8%, with optional 5% TP

DISCLAIMER: Not financial advice. Test thoroughly before using with real funds.
"""

import os
import time
import json
import threading
import sqlite3
import logging
import logging.handlers
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from scipy.stats import linregress
import requests

# =========================
# Config & initialization
# =========================
load_dotenv()

DB_NAME = "trading_real.db"
LOG_PATH = os.path.expanduser("~/hobbies/trading.log")

# Pairs to scan (USDT)
TOP_COINS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'TON', 'ADA', 'TRX', 'AVAX']
SELECTED_CRYPTOS = [f"{c}/USDT" for c in TOP_COINS]

# --- Risk / execution ---
MIN_NOTIONAL = 10.0
MAX_OPEN_TRADES = 10
RESERVE_USDT = 150.0

# --- Decision params (hybrid) ---
DECISION_TIMEFRAME = "1h"
SPREAD_MAX_PCT = 0.005          # <= 0.5%
MIN_QUOTE_VOL_24H = 5_000_000   # safer minimum 24h quote volume (USDT)
ADX_MIN = 20
RSI_MIN, RSI_MAX = 45, 72
RVOL_BASE = 1.5                 # base participation threshold

# --- Sizing ---
RISK_FRACTION = 0.05            # allocate up to 5% of USDT per signal (nudged by confidence)

# Optional Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(text: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=4)
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

# --- Logger ---
logger = logging.getLogger("hybrid_trader")
logger.setLevel(logging.INFO)
handler = logging.handlers.TimedRotatingFileHandler(
    LOG_PATH, when="midnight", interval=1, backupCount=14, utc=True
)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

def check_log_rotation(max_size_mb=1):
    try:
        if os.path.getsize(handler.baseFilename) > max_size_mb * 1024 * 1024:
            handler.doRollover()
    except Exception:
        pass

# --- DB (lean) ---
def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            ts TEXT NOT NULL,
            trade_id TEXT NOT NULL,
            adx REAL,
            rsi REAL,
            rvol REAL,
            atr REAL,
            score REAL,
            confidence INTEGER,
            status TEXT DEFAULT 'open'
        )
    """)
    conn.commit()
    conn.close()

initialize_db()

# --- Exchange ---
exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY_REAL"),
    'secret': os.getenv("BINANCE_SECRET_KEY_REAL"),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})

# =========================
# Utilities
# =========================
def fetch_price(symbol: str):
    try:
        t = exchange.fetch_ticker(symbol)
        price = t.get('last', None)
        return float(price) if price is not None else None
    except Exception as e:
        logger.error(f"fetch_price error {symbol}: {e}")
        return None

def fetch_ohlcv_with_retry(symbol: str, timeframe: str, limit: int = 200, max_retries: int = 3):
    for k in range(max_retries):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if data and len(data) > 0:
                return data
        except Exception as e:
            logger.warning(f"OHLCV retry {k+1}/{max_retries} {symbol} {timeframe}: {e}")
        time.sleep(2 ** k)
    return None

def percent_spread(order_book: dict) -> float:
    try:
        bid = order_book['bids'][0][0]
        ask = order_book['asks'][0][0]
        if ask:
            return (ask - bid) / ask
    except Exception:
        pass
    return float('inf')

def get_open_trades_count() -> int:
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transactions WHERE side='buy' AND status='open'")
    cnt = cur.fetchone()[0]
    conn.close()
    return int(cnt)

# =========================
# Indicators (hybrid prep)
# =========================
def fetch_and_prepare_data_hybrid(symbol: str, limit: int = 250, timeframe: str = DECISION_TIMEFRAME):
    ohlcv = fetch_ohlcv_with_retry(symbol, timeframe, limit=limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)

    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce').ffill().bfill()

    # Core indicators
    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['RSI']   = ta.rsi(df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14'] if adx is not None and not adx.empty else np.nan
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Relative volume vs rolling 10
    rv_mean = df['volume'].rolling(10).mean()
    df['RVOL10'] = df['volume'] / rv_mean

    # Price/Volume slopes over last 10 candles
    if len(df) >= 10:
        x = np.arange(10)
        df.loc[df.index[-1], 'PRICE_SLOPE10'] = linregress(x, df['close'].iloc[-10:]).slope
        df.loc[df.index[-1], 'VOL_SLOPE10']   = linregress(x, df['volume'].iloc[-10:]).slope
    else:
        df['PRICE_SLOPE10'] = np.nan
        df['VOL_SLOPE10']   = np.nan

    return df

# =========================
# Decision rule (hybrid)
# =========================
def hybrid_decision(symbol: str):
    # ---------- Hard guards ----------
    try:
        t = exchange.fetch_ticker(symbol)
        qvol = float(t.get('quoteVolume', 0.0) or 0.0)
        if qvol < MIN_QUOTE_VOL_24H:
            return "hold", 50, 0.0, f"low 24h quote vol: {qvol:.0f}"
    except Exception as e:
        return "hold", 50, 0.0, f"ticker error: {e}"

    try:
        ob = exchange.fetch_order_book(symbol, limit=50)
        if percent_spread(ob) > SPREAD_MAX_PCT:
            return "hold", 50, 0.0, "spread too wide"
    except Exception as e:
        return "hold", 50, 0.0, f"orderbook error: {e}"

    df = fetch_and_prepare_data_hybrid(symbol)
    if df is None or len(df) < 60:
        return "hold", 50, 0.0, "not enough candles"

    row = df.iloc[-1]

    # Uptrend lane
    uptrend_lane = bool(row['EMA20'] > row['EMA50'] and row['close'] > row['EMA20'])
    if not uptrend_lane:
        return "hold", 55, 1.0, "not in uptrend lane (EMA20<=EMA50 or close<=EMA20)"

    # ---------- Flexible score ----------
    score = 0.0
    notes = []

    # RVOL tiers
    rvol = float(row['RVOL10']) if pd.notna(row['RVOL10']) else 0.0
    if rvol >= RVOL_BASE: score += 2.0; notes.append(f"RVOL≥{RVOL_BASE} ({rvol:.2f})")
    if rvol >= 2.0:       score += 1.0
    if rvol >= 3.0:       score += 1.0

    # Price slope
    price_slope = float(row.get('PRICE_SLOPE10', 0.0) or 0.0)
    if price_slope > 0:
        score += 1.0; notes.append("price slope>0")

    # RSI band (wider, pump-friendly)
    rsi = float(row['RSI']) if pd.notna(row['RSI']) else 50.0
    if RSI_MIN <= rsi <= RSI_MAX: score += 1.0; notes.append(f"RSI in band ({rsi:.1f})")
    if rsi >= 60:                 score += 0.5

    # ADX (trend strength)
    adx = float(row['ADX']) if pd.notna(row['ADX']) else 0.0
    if adx >= ADX_MIN:     score += 1.0; notes.append(f"ADX≥{ADX_MIN} ({adx:.1f})")
    if adx >= 28:          score += 0.5

    # Volume slope nudge
    vol_slope = float(row.get('VOL_SLOPE10', 0.0) or 0.0)
    if vol_slope > 0: score += 0.5

    # ---------- Decision thresholds ----------
    if score >= 4.0:
        conf = int(min(92, 70 + (score - 4.0)*5))  # 70..92
        msg = f"score={score:.1f} | " + ", ".join(notes) + f" | RSI={rsi:.1f} ADX={adx:.1f}"
        return "buy", conf, score, msg

    if score >= 2.5:
        return "hold", 60, score, f"partial alignment score={score:.1f}"

    return "hold", 50, score, f"weak score={score:.1f}"

# =========================
# Execution & trailing
# =========================
def execute_order_buy(symbol: str, amount: float, signals: dict):
    """
    signals = {'adx':..., 'rsi':..., 'rvol':..., 'atr':..., 'score':..., 'confidence':...}
    """
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        price = order.get("price", fetch_price(symbol))
        filled = order.get("filled", amount)
        if not price:
            return None
        trade_id = f"{symbol}-{datetime.now(timezone.utc).isoformat().replace(':','-')}"
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transactions (symbol, side, price, amount, ts, trade_id, adx, rsi, rvol, atr, score, confidence, status)
            VALUES (?, 'buy', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (
            symbol, float(price), float(filled),
            datetime.now(timezone.utc).isoformat(), trade_id,
            signals.get('adx'), signals.get('rsi'),
            signals.get('rvol'), signals.get('atr'),
            signals.get('score'), signals.get('confidence')
        ))
        conn.commit()
        conn.close()
        send_telegram_message(f"✅ BUY {symbol}\nPrice: {price}\nAmount: {filled}\nConf: {signals.get('confidence', 0)}%\nScore: {signals.get('score', 0):.1f}")
        return {"price": price, "filled": filled, "trade_id": trade_id}
    except Exception as e:
        logger.error(f"execute_order_buy error {symbol}: {e}")
        send_telegram_message(f"❌ BUY failed {symbol}: {e}")
        return None

def sell_symbol(symbol: str, amount: float, trade_id: str):
    try:
        price_now = fetch_price(symbol)
        order = exchange.create_market_sell_order(symbol, amount)
        sell_price = order.get("price", price_now) or price_now or 0.0

        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transactions (symbol, side, price, amount, ts, trade_id, status)
            VALUES (?, 'sell', ?, ?, ?, ?, 'closed')
        """, (symbol, float(sell_price), float(amount), datetime.now(timezone.utc).isoformat(), trade_id))
        cur.execute("UPDATE transactions SET status='closed' WHERE trade_id=? AND side='buy'", (trade_id,))
        conn.commit()
        conn.close()

        send_telegram_message(f"✅ SELL {symbol}\nPrice: {sell_price}\nAmount: {amount}")
        logger.info(f"Sold {symbol} @ {sell_price} (trade {trade_id})")
    except Exception as e:
        logger.error(f"sell_symbol error {symbol}: {e}")

def dynamic_trailing_stop(symbol: str, amount: float, purchase_price: float, trade_id: str, atr_abs: float):
    """
    ATR-aware trailing:
      trail_pct = max(1.5 * (ATR/purchase)% , 2%) capped at 8%
    Optional take-profit at +5% from entry.
    """
    def loop():
        try:
            highest = purchase_price
            atr_pct = (atr_abs / purchase_price) * 100 if purchase_price > 0 and atr_abs else 2.0
            trail_pct = max(1.5 * atr_pct, 2.0)
            trail_pct = min(trail_pct, 8.0)
            take_profit = purchase_price * 1.05  # optional

            while True:
                price = fetch_price(symbol)
                if not price:
                    time.sleep(30)
                    continue

                if price > highest:
                    highest = price

                stop_price = highest * (1 - trail_pct/100.0)

                logger.info(f"[trail] {symbol} price={price:.6f} high={highest:.6f} stop={stop_price:.6f} trail={trail_pct:.2f}%")

                if price <= stop_price or price >= take_profit:
                    sell_symbol(symbol, amount, trade_id)
                    break
                time.sleep(30)
        except Exception as e:
            logger.error(f"Trailing error {symbol}: {e}")
            try:
                sell_symbol(symbol, amount, trade_id)
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True).start()

# =========================
# Sizing
# =========================
def size_position(price: float, usdt_balance: float, confidence: int) -> float:
    """
    Fixed fraction sizing modulated by confidence (50..92% -> 0.6..1.0 multiplier on base risk).
    Ensures at least the minimum notional in USDT.
    """
    if not price or price <= 0:
        return 0.0
    conf_mult = (confidence - 50) / 50.0  # 0..0.84 roughly (capped by decision)
    conf_mult = max(0.0, min(conf_mult, 0.84))
    budget = usdt_balance * (RISK_FRACTION * (0.6 + 0.4 * conf_mult))
    amount = max(MIN_NOTIONAL / price, budget / price)
    return amount

# =========================
# Main trading step
# =========================
buy_lock = threading.Lock()

def trade_once(symbol: str):
    try:
        # Slots
        open_trades = get_open_trades_count()
        if open_trades >= MAX_OPEN_TRADES:
            logger.info("Max open trades reached.")
            return False

        balances = exchange.fetch_balance()
        usdt = float(balances.get('free', {}).get('USDT', 0.0))
        if usdt - RESERVE_USDT < MIN_NOTIONAL:
            logger.info(f"Insufficient USDT after reserve. USDT={usdt:.2f}")
            return False

        # Decision
        action, conf, score, note = hybrid_decision(symbol)
        logger.info(f"{symbol}: decision={action} conf={conf} score={score:.1f} note={note}")

        if action != "buy":
            return False

        with buy_lock:
            # re-check within lock
            open_trades = get_open_trades_count()
            if open_trades >= MAX_OPEN_TRADES:
                return False

            balances = exchange.fetch_balance()
            usdt = float(balances.get('free', {}).get('USDT', 0.0))
            if usdt - RESERVE_USDT < MIN_NOTIONAL:
                return False

            price = fetch_price(symbol)
            if not price or price <= 0:
                return False

            # indicators snapshot for logging / trailing
            df = fetch_and_prepare_data_hybrid(symbol, limit=200, timeframe=DECISION_TIMEFRAME)
            row = df.iloc[-1]
            atr_abs = float(row['ATR']) if pd.notna(row['ATR']) else price * 0.02

            amount = size_position(price, usdt, conf)
            trade_val = amount * price
            if trade_val < MIN_NOTIONAL and trade_val < (usdt - RESERVE_USDT):
                logger.info(f"{symbol}: trade value {trade_val:.2f} < MIN_NOTIONAL, skip.")
                return False

            order = execute_order_buy(symbol, amount, {
                'adx': float(row['ADX']) if pd.notna(row['ADX']) else None,
                'rsi': float(row['RSI']) if pd.notna(row['RSI']) else None,
                'rvol': float(row['RVOL10']) if pd.notna(row['RVOL10']) else None,
                'atr': atr_abs,
                'score': float(score),
                'confidence': int(conf)
            })

            if order:
                logger.info(f"{symbol}: BUY @ {order['price']} (conf {conf}%) — {note}")
                send_telegram_message(f"{symbol}: BUY @ {order['price']} (conf {conf}%) — {note}")
                dynamic_trailing_stop(symbol, order['filled'], order['price'], order['trade_id'], atr_abs)
                return True
            return False
    except Exception as e:
        logger.error(f"trade_once error {symbol}: {e}", exc_info=True)
        return False

# =========================
# Main loop
# =========================
if __name__ == "__main__":
    logger.info("Starting hybrid trader...")
    try:
        while True:
            check_log_rotation()
            for sym in SELECTED_CRYPTOS:
                trade_once(sym)
                time.sleep(1)   # small pacing between symbols
            time.sleep(30)      # main cycle
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down gracefully.")
        logging.shutdown()
        for h in logger.handlers[:]:
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)
