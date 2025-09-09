#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Crypto Spot Trader (Binance via ccxt) — Buy-often adaptive version

Key ideas:
- Buys enough: adaptive score-gate to hit a target buy rate; small exploration; momentum override.
- Accepts some losses: soft learning nudges, not hard filters.
- Observability: Telegram + feature JSON + status.json + decision_log DB.
- Robust restarts: startup_cleanup sells non-USDT and syncs DB; reconcile orphan BUYs.

Test in paper or with smallest notional first!
"""

import os, time, json, threading, sqlite3, logging, logging.handlers, random
from datetime import datetime, timezone
import ccxt, numpy as np, pandas as pd, pandas_ta as ta, requests
from dotenv import load_dotenv
from scipy.stats import linregress

# =========================
# Config & initialization
# =========================
load_dotenv()

DB_NAME = "trading_real.db"
LOG_PATH = os.path.expanduser("~/hobbies/trading.log")

TOP_COINS = ['BTC','ETH','BNB','SOL','XRP','DOGE','TON','ADA','TRX','AVAX']
SELECTED_CRYPTOS = [f"{c}/USDT" for c in TOP_COINS]

# Execution / risk
MIN_NOTIONAL = 8.0
MAX_OPEN_TRADES = 10
RESERVE_USDT = 100.0
RISK_FRACTION = 0.05             # base fraction of USDT per signal (modulated by confidence & controller)

# Decision params (default)
DECISION_TIMEFRAME = "1h"
SPREAD_MAX_PCT_DEFAULT = 0.005   # <= 0.5%
MIN_QUOTE_VOL_24H_DEFAULT = 3_000_000  # lower than before to allow more buys

ADX_MIN_DEFAULT = 20
RSI_MIN_DEFAULT, RSI_MAX_DEFAULT = 45, 72
RVOL_BASE_DEFAULT = 1.5
SCORE_GATE_START = 4.0
SCORE_GATE_MIN, SCORE_GATE_MAX = 2.5, 6.0

# Learning ranges & rates
ADX_MIN_RANGE = (15, 35)
RSI_MAX_RANGE = (60, 80)
RVOL_BASE_RANGE = (1.2, 3.0)
LEARN_RATE = 0.15                 # soft decay for parameter nudges

# Buy-flow controller
TARGET_BUY_RATIO = 0.25           # target % of decisions resulting in executed BUY
BUY_RATIO_DELTA = 0.05            # deadband
GATE_STEP = 0.1                   # gate nudge step
EPSILON_EXPLORE = 0.03            # chance to jitter score near boundary
DECISION_WINDOW = 120             # last N decisions for controller window

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Logger
logger = logging.getLogger("hybrid_trader")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_PATH, when="midnight", interval=1, backupCount=14, utc=True
)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(fmt); logger.addHandler(file_handler)

console = logging.StreamHandler()
console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console.setFormatter(fmt); logger.addHandler(console)

def check_log_rotation(max_size_mb=1):
    try:
        if os.path.getsize(file_handler.baseFilename) > max_size_mb * 1024 * 1024:
            file_handler.doRollover()
    except Exception:
        pass

# =========================
# DB schema
# =========================
def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,                 -- 'buy' / 'sell'
            price REAL NOT NULL,
            amount REAL NOT NULL,
            ts TEXT NOT NULL,
            trade_id TEXT NOT NULL,
            adx REAL, rsi REAL, rvol REAL, atr REAL,
            score REAL, confidence INTEGER,
            status TEXT DEFAULT 'open'          -- 'open' | 'closed' | 'closed_orphan'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,                 -- 'entry' / 'exit'
            features_json TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_analysis (
            trade_id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            pnl_usdt REAL, pnl_pct REAL, duration_sec INTEGER,
            entry_snapshot_json TEXT, exit_snapshot_json TEXT,
            adjustments_json TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS learn_params (
            symbol TEXT PRIMARY KEY,
            rsi_min REAL, rsi_max REAL, adx_min REAL, rvol_base REAL,
            updated_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS controller (
            id INTEGER PRIMARY KEY CHECK (id=1),
            score_gate REAL,
            updated_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS decision_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            score REAL,
            action TEXT,
            executed INTEGER
        )
    """)

    # init controller row
    cur.execute("SELECT score_gate FROM controller WHERE id=1")
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO controller (id, score_gate, updated_at) VALUES (1, ?, ?)",
                    (SCORE_GATE_START, datetime.now(timezone.utc).isoformat()))
    conn.commit(); conn.close()

initialize_db()

# =========================
# Exchange
# =========================
exchange = ccxt.binance({
    'apiKey': os.getenv("BINANCE_API_KEY_REAL"),
    'secret': os.getenv("BINANCE_SECRET_KEY_REAL"),
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
})

# =========================
# Telegram helpers
# =========================
def send_telegram_message(text: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=8)
    except Exception as e:
        logger.error(f"Telegram sendMessage error: {e}")

def send_telegram_document(file_path: str, caption: str = ""):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        with open(file_path, "rb") as f:
            files = {"document": (os.path.basename(file_path), f, "application/json")}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]}
            requests.post(url, files=files, data=data, timeout=60)
    except Exception as e:
        logger.error(f"Telegram sendDocument error: {e}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
            snippet = raw[:3500] + (" ... (truncated)" if len(raw) > 3500 else "")
            send_telegram_message(f"*Document upload failed; inline snippet:*\n```\n{snippet}\n```")
        except Exception as e2:
            logger.error(f"Telegram fallback snippet error: {e2}")

# =========================
# Utils
# =========================
def fetch_price(symbol: str):
    try:
        t = exchange.fetch_ticker(symbol)
        price = t.get('last', None)
        return float(price) if price is not None else None
    except Exception as e:
        logger.error(f"fetch_price error {symbol}: {e}")
        return None

def fetch_ticker_safe(symbol: str):
    try:
        return exchange.fetch_ticker(symbol) or {}
    except Exception as e:
        logger.warning(f"fetch_ticker_safe {symbol}: {e}")
        return {}

def fetch_order_book_safe(symbol: str, limit: int = 100):
    try:
        return exchange.fetch_order_book(symbol, limit=limit) or {}
    except Exception as e:
        logger.warning(f"fetch_order_book_safe {symbol}: {e}")
        return {}

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
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transactions WHERE side='buy' AND status='open'")
    cnt = cur.fetchone()[0]; conn.close()
    return int(cnt)

def has_open_position(symbol: str) -> bool:
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transactions WHERE symbol=? AND side='buy' AND status='open'", (symbol,))
    cnt = cur.fetchone()[0]; conn.close()
    return cnt > 0

def abbr(num):
    try: num = float(num)
    except Exception: return None
    for unit in ["","K","M","B","T"]:
        if abs(num) < 1000.0:
            return f"{num:.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}P"

def pct(x):
    try: return f"{float(x):.2f}%"
    except Exception: return "—"

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# =========================
# Feature builders
# =========================
def compute_timeframe_features(df: pd.DataFrame, label: str):
    features = {}
    try:
        ema20 = ta.ema(df['close'], length=20)
        ema50 = ta.ema(df['close'], length=50)
        ema200 = ta.ema(df['close'], length=200) if len(df) >= 200 else pd.Series(index=df.index, dtype=float)

        rsi = ta.rsi(df['close'], length=14)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        adx = adx_df['ADX_14'] if adx_df is not None and not adx_df.empty else pd.Series(index=df.index, dtype=float)
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)

        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        macd = macd_df['MACD_12_26_9'] if macd_df is not None and not macd_df.empty else pd.Series(index=df.index, dtype=float)
        macds = macd_df['MACDs_12_26_9'] if macd_df is not None and not macd_df.empty else pd.Series(index=df.index, dtype=float)
        macdh = macd_df['MACDh_12_26_9'] if macd_df is not None and not macd_df.empty else pd.Series(index=df.index, dtype=float)

        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        stoch_k = stoch['STOCHk_14_3_3'] if stoch is not None and not stoch.empty else pd.Series(index=df.index, dtype=float)
        stoch_d = stoch['STOCHd_14_3_3'] if stoch is not None and not stoch.empty else pd.Series(index=df.index, dtype=float)

        bb = ta.bbands(df['close'], length=20, std=2)
        bbl = bb['BBL_20_2.0'] if bb is not None and not bb.empty else pd.Series(index=df.index, dtype=float)
        bbm = bb['BBM_20_2.0'] if bb is not None and not bb.empty else pd.Series(index=df.index, dtype=float)
        bbu = bb['BBU_20_2.0'] if bb is not None and not bb.empty else pd.Series(index=df.index, dtype=float)

        obv = ta.obv(df['close'], df['volume'])
        roc7 = ta.roc(df['close'], length=7)
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

        rv_mean = df['volume'].rolling(10).mean()
        rvol10 = df['volume'] / rv_mean

        price_slope10 = vol_slope10 = np.nan
        if len(df) >= 10:
            x = np.arange(10)
            price_slope10 = linregress(x, df['close'].iloc[-10:]).slope
            vol_slope10   = linregress(x, df['volume'].iloc[-10:]).slope

        last = df.iloc[-1]
        def sf(x):
            try:
                if pd.isna(x): return None
                return float(x)
            except Exception:
                return None

        bb_pos = None
        if (not pd.isna(bbu.iloc[-1]) and not pd.isna(bbl.iloc[-1]) and (bbu.iloc[-1]-bbl.iloc[-1]) != 0):
            bb_pos = (last['close'] - bbl.iloc[-1]) / (bbu.iloc[-1] - bbl.iloc[-1])

        atr_pct = None
        try: atr_pct = float(atr.iloc[-1] / last['close'] * 100.0)
        except Exception: pass

        features = {
            "label": label,
            "last_close": sf(last['close']),
            "atr": sf(atr.iloc[-1]), "atr_pct": atr_pct,
            "rsi": sf(rsi.iloc[-1]), "adx": sf(adx.iloc[-1]),
            "ema20": sf(ema20.iloc[-1]), "ema50": sf(ema50.iloc[-1]), "ema200": sf(ema200.iloc[-1]) if len(ema200) else None,
            "macd": sf(macd.iloc[-1]), "macd_signal": sf(macds.iloc[-1]), "macd_hist": sf(macdh.iloc[-1]),
            "stoch_k": sf(stoch_k.iloc[-1]), "stoch_d": sf(stoch_d.iloc[-1]),
            "bb_lower": sf(bbl.iloc[-1]), "bb_mid": sf(bbm.iloc[-1]), "bb_upper": sf(bbu.iloc[-1]), "bb_pos": bb_pos,
            "obv": sf(obv.iloc[-1]), "roc7": sf(roc7.iloc[-1]), "vwap": sf(vwap.iloc[-1]),
            "rvol10": sf(rvol10.iloc[-1]),
            "price_slope10": float(price_slope10) if not pd.isna(price_slope10) else None,
            "vol_slope10": float(vol_slope10) if not pd.isna(vol_slope10) else None,
        }
    except Exception as e:
        logger.warning(f"compute_timeframe_features({label}) error: {e}")
        features = {}
    return features

def detect_support_level_simple(df: pd.DataFrame, window: int = 20):
    try:
        recent = df['close'].iloc[-window:]
        loc_mins = [recent.iloc[i] for i in range(1, len(recent)-1)
                    if recent.iloc[i] < recent.iloc[i-1] and recent.iloc[i] < recent.iloc[i+1]]
        if not loc_mins: return None, None
        support = float(min(loc_mins))
        price = float(df['close'].iloc[-1])
        dist = (price - support) / support * 100.0 if support > 0 else None
        return support, dist
    except Exception:
        return None, None

def fetch_and_prepare_df(symbol: str, timeframe: str, limit: int = 250):
    ohlcv = fetch_ohlcv_with_retry(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv: return None
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce').ffill().bfill()
    return df

def build_rich_features(symbol: str):
    t = fetch_ticker_safe(symbol)
    last = t.get('last'); bid = t.get('bid'); ask = t.get('ask')
    qvol = t.get('quoteVolume'); bvol = t.get('baseVolume')
    vwap = t.get('vwap'); high = t.get('high'); low = t.get('low')

    ob = fetch_order_book_safe(symbol, limit=100)
    try:
        b0 = ob['bids'][0][0]; a0 = ob['asks'][0][0]
        b0v = ob['bids'][0][1]; a0v = ob['asks'][0][1]
    except Exception:
        b0 = a0 = b0v = a0v = None

    try:
        bid_vol_sum = float(sum(v for _, v in ob.get('bids', [])))
        ask_vol_sum = float(sum(v for _, v in ob.get('asks', [])))
        mid = (b0 + a0) / 2.0 if (b0 and a0) else None
        depth_usdt = ((bid_vol_sum + ask_vol_sum) * mid) if (mid and bid_vol_sum and ask_vol_sum) else None
        imbalance = (bid_vol_sum / ask_vol_sum) if ask_vol_sum not in (None, 0) else None
        spr = percent_spread(ob)
    except Exception:
        bid_vol_sum = ask_vol_sum = depth_usdt = imbalance = None
        spr = float('inf')

    tf_map = {}
    for tf in ['15m', '1h', '4h']:
        df = fetch_and_prepare_df(symbol, tf, limit=250)
        if df is not None and len(df) >= 30:
            tf_map[tf] = compute_timeframe_features(df, tf)
            if tf == '1h':
                sup, sup_dist = detect_support_level_simple(df, window=20)
        else:
            tf_map[tf] = {}
            if tf == '1h': sup, sup_dist = (None, None)

    features = {
        "general": {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "last": float(last) if last is not None else None,
            "bid": float(bid) if bid is not None else None,
            "ask": float(ask) if ask is not None else None,
            "vwap": float(vwap) if vwap is not None else None,
            "high_24h": float(high) if high is not None else None,
            "low_24h": float(low) if low is not None else None,
            "quote_volume_24h": float(qvol) if qvol is not None else None,
            "base_volume_24h": float(bvol) if bvol is not None else None
        },
        "orderbook": {
            "best_bid": b0, "best_ask": a0, "best_bid_vol": b0v, "best_ask_vol": a0v,
            "spread_pct": spr if spr != float('inf') else None,
            "bid_vol_sum": bid_vol_sum, "ask_vol_sum": ask_vol_sum,
            "depth_usdt": depth_usdt, "imbalance": imbalance
        },
        "timeframes": tf_map,
        "support_1h": {"support": sup, "distance_pct": sup_dist}
    }
    return features

# =========================
# Learning state
# =========================
def get_learn_params(symbol_base: str):
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("SELECT rsi_min, rsi_max, adx_min, rvol_base FROM learn_params WHERE symbol=?", (symbol_base,))
    row = cur.fetchone(); conn.close()
    if not row:
        return {"rsi_min": RSI_MIN_DEFAULT, "rsi_max": RSI_MAX_DEFAULT,
                "adx_min": ADX_MIN_DEFAULT, "rvol_base": RVOL_BASE_DEFAULT}
    return {"rsi_min": float(row[0]) if row[0] is not None else RSI_MIN_DEFAULT,
            "rsi_max": float(row[1]) if row[1] is not None else RSI_MAX_DEFAULT,
            "adx_min": float(row[2]) if row[2] is not None else ADX_MIN_DEFAULT,
            "rvol_base": float(row[3]) if row[3] is not None else RVOL_BASE_DEFAULT}

def set_learn_params(symbol_base: str, rsi_min: float, rsi_max: float, adx_min: float, rvol_base: float):
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("""
        INSERT INTO learn_params (symbol, rsi_min, rsi_max, adx_min, rvol_base, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            rsi_min=excluded.rsi_min, rsi_max=excluded.rsi_max,
            adx_min=excluded.adx_min, rvol_base=excluded.rvol_base,
            updated_at=excluded.updated_at
    """, (symbol_base, rsi_min, rsi_max, adx_min, rvol_base, datetime.now(timezone.utc).isoformat()))
    conn.commit(); conn.close()

def smooth_nudge(cur, target, lo, hi):
    nxt = cur + LEARN_RATE * (target - cur)
    return clamp(nxt, lo, hi)

# =========================
# Controller (score gate)
# =========================
def get_score_gate():
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("SELECT score_gate FROM controller WHERE id=1")
    row = cur.fetchone(); conn.close()
    return float(row[0]) if row and row[0] is not None else SCORE_GATE_START

def set_score_gate(new_gate: float):
    ng = clamp(float(new_gate), SCORE_GATE_MIN, SCORE_GATE_MAX)
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("UPDATE controller SET score_gate=?, updated_at=? WHERE id=1",
                (ng, datetime.now(timezone.utc).isoformat()))
    conn.commit(); conn.close()
    return ng

def log_decision(symbol: str, score: float, action: str, executed: bool):
    try:
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("INSERT INTO decision_log (ts, symbol, score, action, executed) VALUES (?, ?, ?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(), symbol, float(score), action, 1 if executed else 0))
        # keep DB small: delete oldest beyond window * 8
        cur.execute("SELECT COUNT(*) FROM decision_log"); n = cur.fetchone()[0]
        if n > DECISION_WINDOW * 8:
            cur.execute("DELETE FROM decision_log WHERE id IN (SELECT id FROM decision_log ORDER BY id ASC LIMIT ?)", (n - DECISION_WINDOW*8,))
        conn.commit(); conn.close()
    except Exception as e:
        logger.debug(f"log_decision error: {e}")

def controller_autotune():
    """Run once per cycle, adjust score gate to achieve target buy ratio on recent window."""
    try:
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("SELECT executed FROM decision_log ORDER BY id DESC LIMIT ?", (DECISION_WINDOW,))
        rows = cur.fetchall(); conn.close()
        if not rows or len(rows) < max(20, DECISION_WINDOW // 4):  # need a minimum sample
            return
        execs = sum(1 for (e,) in rows if e == 1)
        ratio = execs / len(rows)
        gate = get_score_gate()
        if ratio < (TARGET_BUY_RATIO - BUY_RATIO_DELTA):
            gate = set_score_gate(gate - GATE_STEP)
            logger.info(f"[controller] Low buy ratio {ratio:.2f} < target {TARGET_BUY_RATIO:.2f}. Decreasing score_gate -> {gate:.2f}")
        elif ratio > (TARGET_BUY_RATIO + BUY_RATIO_DELTA):
            gate = set_score_gate(gate + GATE_STEP)
            logger.info(f"[controller] High buy ratio {ratio:.2f} > target {TARGET_BUY_RATIO:.2f}. Increasing score_gate -> {gate:.2f}")
    except Exception as e:
        logger.debug(f"controller_autotune error: {e}")

# =========================
# Decision (hybrid)
# =========================
def fetch_and_prepare_data_hybrid(symbol: str, limit: int = 200, timeframe: str = DECISION_TIMEFRAME):
    ohlcv = fetch_ohlcv_with_retry(symbol, timeframe, limit=limit)
    if not ohlcv: return None
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce').ffill().bfill()
    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['RSI']   = ta.rsi(df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14'] if adx is not None and not adx.empty else np.nan
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    rv_mean = df['volume'].rolling(10).mean()
    df['RVOL10'] = df['volume'] / rv_mean

    if len(df) >= 10:
        x = np.arange(10)
        df.loc[df.index[-1], 'PRICE_SLOPE10'] = linregress(x, df['close'].iloc[-10:]).slope
        df.loc[df.index[-1], 'VOL_SLOPE10']   = linregress(x, df['volume'].iloc[-10:]).slope
    else:
        df['PRICE_SLOPE10'] = np.nan; df['VOL_SLOPE10'] = np.nan

    return df

def hybrid_decision(symbol: str):
    base = symbol.split('/')[0]
    lp = get_learn_params(base)
    RSI_MIN = lp["rsi_min"]; RSI_MAX = lp["rsi_max"]
    ADX_MIN = lp["adx_min"]; RVOL_BASE = lp["rvol_base"]

    # Liquidity
    try:
        t = exchange.fetch_ticker(symbol)
        qvol = float(t.get('quoteVolume', 0.0) or 0.0)
        if qvol < MIN_QUOTE_VOL_24H_DEFAULT:
            return "hold", 50, 0.0, f"low 24h quote vol: {qvol:.0f}"
    except Exception as e:
        return "hold", 50, 0.0, f"ticker error: {e}"

    # Spread
    try:
        ob = exchange.fetch_order_book(symbol, limit=50)
        if percent_spread(ob) > SPREAD_MAX_PCT_DEFAULT:
            return "hold", 50, 0.0, "spread too wide"
    except Exception as e:
        return "hold", 50, 0.0, f"orderbook error: {e}"

    df = fetch_and_prepare_data_hybrid(symbol)
    if df is None or len(df) < 60:
        return "hold", 50, 0.0, "not enough candles"

    row = df.iloc[-1]
    score_gate = get_score_gate()

    # Uptrend lane
    in_lane = bool(row['EMA20'] > row['EMA50'] and row['close'] > row['EMA20'])

    # Momentum override if not in lane
    adx = float(row['ADX']) if pd.notna(row['ADX']) else 0.0
    rvol = float(row['RVOL10']) if pd.notna(row['RVOL10']) else 0.0
    price_slope = float(row.get('PRICE_SLOPE10', 0.0) or 0.0)
    override = (not in_lane) and (adx >= ADX_MIN + 10) and (rvol >= RVOL_BASE * 1.5) and (price_slope > 0)

    if not (in_lane or override):
        return "hold", 55, 1.0, "not in uptrend lane; no override"

    # Flexible scoring
    score = 0.0; notes = []
    if rvol >= RVOL_BASE: score += 2.0; notes.append(f"RVOL≥{RVOL_BASE:.2f} ({rvol:.2f})")
    if rvol >= RVOL_BASE + 0.5: score += 1.0
    if rvol >= RVOL_BASE + 1.5: score += 1.0

    if price_slope > 0: score += 1.0; notes.append("price slope>0")

    rsi = float(row['RSI']) if pd.notna(row['RSI']) else 50.0
    if RSI_MIN <= rsi <= RSI_MAX: score += 1.0; notes.append(f"RSI in band ({rsi:.1f})")
    if rsi >= (RSI_MIN + RSI_MAX)/2: score += 0.5

    if adx >= ADX_MIN: score += 1.0; notes.append(f"ADX≥{ADX_MIN:.0f} ({adx:.1f})")
    vol_slope = float(row.get('VOL_SLOPE10', 0.0) or 0.0)
    if vol_slope > 0: score += 0.5

    # Exploration jitter
    if random.random() < EPSILON_EXPLORE:
        jitter = random.uniform(-0.3, 0.3)
        score += jitter
        notes.append(f"jitter {jitter:+.2f}")

    msg = f"score={score:.1f} gate={score_gate:.1f} | " + ", ".join(notes) + f" | RSI={rsi:.1f} ADX={adx:.1f}"

    if score >= score_gate:
        conf = int(min(92, 70 + max(0.0, score - score_gate)*6))
        return "buy", conf, score, msg

    # not enough score → hold
    return "hold", 60 if score >= (score_gate - 0.5) else 50, score, msg

# =========================
# Feature persistence & Telegram bundles
# =========================
def save_trade_features(trade_id: str, symbol: str, side: str, features: dict):
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("INSERT INTO trade_features (trade_id, ts, symbol, side, features_json) VALUES (?,?,?,?,?)",
                (trade_id, datetime.now(timezone.utc).isoformat(), symbol, side, json.dumps(features, ensure_ascii=False)))
    conn.commit(); conn.close()

def send_feature_bundle_telegram(trade_id: str, symbol: str, side: str, features: dict, extra_lines: str = ""):
    try:
        gen = features.get("general", {}); ob = features.get("orderbook", {})
        tf1h = features.get("timeframes", {}).get("1h", {})
        tf15 = features.get("timeframes", {}).get("15m", {})
        tf4h = features.get("timeframes", {}).get("4h", {})
        sup = features.get("support_1h", {})

        summary = (
            f"*{side.upper()} FEATURES* {symbol}\n"
            f"Trade: `{trade_id}`\n"
            f"{extra_lines}"
            f"—\n"
            f"*Market*\n"
            f"Last: `{gen.get('last')}`  24hQVol: `{abbr(gen.get('quote_volume_24h'))}`  Spread: `{pct((ob.get('spread_pct') or 0)*100)}`\n"
            f"Depth≈ `{abbr(ob.get('depth_usdt'))}`  Imb: `{(lambda x: f'{x:.2f}' if x not in (None, float('inf')) else '—')(ob.get('imbalance'))}`  "
            f"SupportΔ: `{(lambda x: f'{x:.2f}%' if x is not None else '—')(sup.get('distance_pct'))}`\n"
            f"—\n"
            f"*1h*  RSI:`{tf1h.get('rsi')}` ADX:`{tf1h.get('adx')}` ATR%:`{(lambda v: f'{v:.2f}%' if v else '—')(tf1h.get('atr_pct'))}` RVOL10:`{tf1h.get('rvol10')}`\n"
            f"*15m* RSI:`{tf15.get('rsi')}` RVOL10:`{tf15.get('rvol10')}` MACD-h:`{tf15.get('macd_hist')}`\n"
            f"*4h*  RSI:`{tf4h.get('rsi')}` ADX:`{tf4h.get('adx')}`\n"
        )
        send_telegram_message(summary)

        fname = f"{side}_{trade_id.replace(':','-').replace('/','_')}_features.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2, ensure_ascii=False)
        send_telegram_document(fname, caption=f"{symbol} {side} features (trade {trade_id})")
        try: os.remove(fname)
        except Exception: pass
    except Exception as e:
        logger.error(f"send_feature_bundle_telegram error: {e}")

# =========================
# Execution & trailing
# =========================
def execute_order_buy(symbol: str, amount: float, signals: dict):
    try:
        order = exchange.create_market_buy_order(symbol, amount)
        price = order.get("price", fetch_price(symbol))
        filled = order.get("filled", amount)
        if not price: return None
        trade_id = f"{symbol}-{datetime.now(timezone.utc).isoformat().replace(':','-')}"

        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("""
            INSERT INTO transactions (symbol, side, price, amount, ts, trade_id, adx, rsi, rvol, atr, score, confidence, status)
            VALUES (?, 'buy', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (symbol, float(price), float(filled),
              datetime.now(timezone.utc).isoformat(), trade_id,
              signals.get('adx'), signals.get('rsi'), signals.get('rvol'), signals.get('atr'),
              signals.get('score'), signals.get('confidence')))
        conn.commit(); conn.close()

        send_telegram_message(f"✅ BUY {symbol}\nPrice: {price}\nAmount: {filled}\nConf: {signals.get('confidence', 0)}%\nScore: {signals.get('score', 0):.1f}")

        features = build_rich_features(symbol)
        save_trade_features(trade_id, symbol, 'entry', features)
        send_feature_bundle_telegram(trade_id, symbol, 'entry', features)

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
        sell_ts = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("""
            INSERT INTO transactions (symbol, side, price, amount, ts, trade_id, status)
            VALUES (?, 'sell', ?, ?, ?, ?, 'closed')
        """, (symbol, float(sell_price), float(amount), sell_ts, trade_id))
        cur.execute("UPDATE transactions SET status='closed' WHERE trade_id=? AND side='buy'", (trade_id,))
        conn.commit(); conn.close()

        send_telegram_message(f"✅ SELL {symbol}\nPrice: {sell_price}\nAmount: {amount}")

        features_exit = build_rich_features(symbol)
        save_trade_features(trade_id, symbol, 'exit', features_exit)
        send_feature_bundle_telegram(trade_id, symbol, 'exit', features_exit)

        analyze_and_learn(trade_id, sell_price)
        logger.info(f"Sold {symbol} @ {sell_price} (trade {trade_id})")
    except Exception as e:
        logger.error(f"sell_symbol error {symbol}: {e}")

def dynamic_trailing_stop(symbol: str, amount: float, purchase_price: float, trade_id: str, atr_abs: float):
    def loop():
        try:
            highest = purchase_price
            atr_pct = (atr_abs / purchase_price) * 100 if purchase_price > 0 and atr_abs else 2.0
            trail_pct = min(max(1.5 * atr_pct, 2.0), 8.0)
            take_profit = purchase_price * 1.05  # optional TP

            while True:
                price = fetch_price(symbol)
                if not price:
                    time.sleep(30); continue

                if price > highest: highest = price
                stop_price = highest * (1 - trail_pct/100.0)
                logger.debug(f"[trail] {symbol} price={price:.6f} high={highest:.6f} stop={stop_price:.6f} trail={trail_pct:.2f}%")

                if price <= stop_price or price >= take_profit:
                    sell_symbol(symbol, amount, trade_id)
                    break
                time.sleep(30)
        except Exception as e:
            logger.error(f"Trailing error {symbol}: {e}")
            try: sell_symbol(symbol, amount, trade_id)
            except Exception: pass
    threading.Thread(target=loop, daemon=True).start()

# =========================
# Post-trade analysis & learning
# =========================
def fetch_trade_legs(trade_id: str):
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("""
        SELECT symbol, price, amount, ts, adx, rsi, rvol, atr, score, confidence
        FROM transactions
        WHERE trade_id=? AND side='buy' LIMIT 1
    """, (trade_id,)); buy = cur.fetchone()

    cur.execute("""
        SELECT price, amount, ts
        FROM transactions
        WHERE trade_id=? AND side='sell'
        ORDER BY ts DESC LIMIT 1
    """, (trade_id,)); sell = cur.fetchone()

    cur.execute("SELECT side, features_json FROM trade_features WHERE trade_id=?", (trade_id,))
    feats = cur.fetchall(); conn.close()

    feats_entry = feats_exit = {}
    for side, fj in feats:
        if side == 'entry': feats_entry = json.loads(fj)
        elif side == 'exit': feats_exit = json.loads(fj)
    return buy, sell, feats_entry, feats_exit

def analyze_and_learn(trade_id: str, sell_price: float = None):
    try:
        buy, sell, feats_entry, feats_exit = fetch_trade_legs(trade_id)
        if not buy or not sell:
            logger.warning(f"analyze_and_learn: incomplete legs for {trade_id}")
            return

        symbol, buy_price, amount, buy_ts, adx_e, rsi_e, rvol_e, atr_e, score_e, conf_e = buy
        sell_p, sell_amt, sell_ts = sell
        sell_price = sell_price or sell_p

        pnl_usdt = (sell_price - buy_price) * amount
        pnl_pct = (sell_price / buy_price - 1.0) * 100.0 if buy_price else 0.0
        dur_sec = int((datetime.fromisoformat(sell_ts.replace("Z","")).timestamp()
                      - datetime.fromisoformat(buy_ts.replace("Z","")).timestamp()))

        tf1h_x = feats_exit.get("timeframes", {}).get("1h", {})
        rsi_x = tf1h_x.get("rsi"); adx_x = tf1h_x.get("adx"); rvol_x = tf1h_x.get("rvol10")

        analysis = {
            "trade_id": trade_id, "symbol": symbol,
            "pnl_usdt": round(pnl_usdt, 6), "pnl_pct": round(pnl_pct, 4), "duration_sec": dur_sec,
            "entry": {"price": buy_price, "amount": amount, "ts": buy_ts, "rsi": rsi_e, "adx": adx_e, "rvol": rvol_e, "atr": atr_e, "score": score_e, "confidence": conf_e},
            "exit":  {"price": sell_price, "amount": sell_amt, "ts": sell_ts, "rsi": rsi_x, "adx": adx_x, "rvol": rvol_x}
        }

        # soft learning
        base = symbol.split('/')[0]
        lp = get_learn_params(base)
        rsi_min = lp["rsi_min"]; rsi_max = lp["rsi_max"]
        adx_min = lp["adx_min"]; rvol_base = lp["rvol_base"]
        adjustments = {}
        win = pnl_usdt > 0.0

        if not win and rsi_e is not None and rsi_e >= (rsi_max - 1):
            new_rsi_max = smooth_nudge(rsi_max, rsi_max - 1, *RSI_MAX_RANGE)
            if new_rsi_max != rsi_max:
                adjustments["rsimax"] = {"old": rsi_max, "new": new_rsi_max}
                rsi_max = new_rsi_max
        elif win and rsi_e is not None and rsi_e <= (rsi_min + 2):
            new_rsi_min = clamp(rsi_min - 1, 35, RSI_MAX_RANGE[0]-5)
            if new_rsi_min != rsi_min:
                adjustments["rsimin"] = {"old": rsi_min, "new": new_rsi_min}
                rsi_min = new_rsi_min

        if not win and (adx_e is not None) and adx_e < adx_min:
            new_adx_min = smooth_nudge(adx_min, adx_min + 1, *ADX_MIN_RANGE)
            if new_adx_min != adx_min:
                adjustments["adxmin"] = {"old": adx_min, "new": new_adx_min}
                adx_min = new_adx_min
        elif win and (adx_e is not None) and adx_e >= adx_min + 5:
            new_adx_min = smooth_nudge(adx_min, adx_min - 1, *ADX_MIN_RANGE)
            if new_adx_min != adx_min:
                adjustments["adxmin"] = {"old": adx_min, "new": new_adx_min}
                adx_min = new_adx_min

        if not win and (rvol_e is not None) and rvol_e < rvol_base:
            new_rvol = smooth_nudge(rvol_base, rvol_base + 0.10, *RVOL_BASE_RANGE)
            if new_rvol != rvol_base:
                adjustments["rvolbase"] = {"old": rvol_base, "new": new_rvol}
                rvol_base = new_rvol
        elif win and (rvol_e is not None) and rvol_e >= (rvol_base * 1.2):
            new_rvol = smooth_nudge(rvol_base, rvol_base - 0.05, *RVOL_BASE_RANGE)
            if new_rvol != rvol_base:
                adjustments["rvolbase"] = {"old": rvol_base, "new": new_rvol}
                rvol_base = new_rvol

        if adjustments:
            set_learn_params(base, rsi_min, rsi_max, adx_min, rvol_base)

        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO trade_analysis (trade_id, ts, symbol, pnl_usdt, pnl_pct, duration_sec,
                                                   entry_snapshot_json, exit_snapshot_json, adjustments_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, datetime.now(timezone.utc).isoformat(), symbol,
              float(analysis["pnl_usdt"]), float(analysis["pnl_pct"]), int(analysis["duration_sec"]),
              json.dumps(analysis["entry"], ensure_ascii=False),
              json.dumps(analysis["exit"], ensure_ascii=False),
              json.dumps(adjustments, ensure_ascii=False)))
        conn.commit(); conn.close()

        outcome = "WIN ✅" if win else "LOSS ❌"
        lines = [
            f"📊 *Trade Analysis* {outcome}  {symbol}",
            f"Trade: `{trade_id}`",
            f"PnL: `{analysis['pnl_usdt']:.4f} USDT`  (`{analysis['pnl_pct']:.2f}%`)  Duration: `{analysis['duration_sec']}s`",
            f"Entry: RSI `{analysis['entry']['rsi']}`, ADX `{analysis['entry']['adx']}`, RVOL `{analysis['entry']['rvol']}`",
            f"Exit:  RSI `{analysis['exit']['rsi']}`, ADX `{analysis['exit']['adx']}`, RVOL `{analysis['exit']['rvol']}`",
        ]
        if adjustments:
            adj_txt = ", ".join([f"{k}:{v['old']}→{v['new']}" for k,v in adjustments.items()])
            lines.append(f"Learned nudges → {adj_txt}")
        send_telegram_message("\n".join(lines))

        fname = f"analysis_{trade_id.replace(':','-').replace('/','_')}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump({"analysis": analysis, "adjustments": adjustments,
                       "entry_features": feats_entry, "exit_features": feats_exit}, f, indent=2, ensure_ascii=False)
        send_telegram_document(fname, caption=f"{symbol} trade analysis (trade {trade_id})")
        try: os.remove(fname)
        except Exception: pass

    except Exception as e:
        logger.error(f"analyze_and_learn error {trade_id}: {e}", exc_info=True)

# =========================
# Sizing
# =========================
def size_position(price: float, usdt_balance: float, confidence: int) -> float:
    if not price or price <= 0: return 0.0
    conf_mult = (confidence - 50) / 50.0  # 0..0.84 approx
    conf_mult = max(0.0, min(conf_mult, 0.84))
    # Slightly boost when controller starved (done by gate; keeping sizing simple)
    budget = usdt_balance * (RISK_FRACTION * (0.6 + 0.4 * conf_mult))
    amount = max(MIN_NOTIONAL / price, budget / price)
    return amount

# =========================
# Trading step + report
# =========================
buy_lock = threading.Lock()

def trade_once_with_report(symbol: str):
    report = {"symbol": symbol, "action": "hold", "confidence": 50, "score": 0.0, "note": "", "executed": False}
    try:
        if has_open_position(symbol):
            report["note"] = "already holding"; logger.info(f"{symbol}: {report['note']}"); log_decision(symbol, 0.0, "hold", False); return report

        open_trades = get_open_trades_count()
        if open_trades >= MAX_OPEN_TRADES:
            report["note"] = "max open trades"; logger.info("Max open trades."); log_decision(symbol, 0.0, "hold", False); return report

        balances = exchange.fetch_balance()
        usdt = float(balances.get('free', {}).get('USDT', 0.0))
        if usdt - RESERVE_USDT < MIN_NOTIONAL:
            report["note"] = f"insufficient USDT after reserve ({usdt:.2f})"; logger.info(report["note"]); log_decision(symbol, 0.0, "hold", False); return report

        action, conf, score, note = hybrid_decision(symbol)
        report.update({"action": action, "confidence": conf, "score": float(score), "note": note})
        logger.info(f"{symbol}: decision={action} conf={conf} score={score:.1f} note={note}")

        if action != "buy":
            log_decision(symbol, score, "hold", False); return report

        with buy_lock:
            open_trades = get_open_trades_count()
            if open_trades >= MAX_OPEN_TRADES:
                report["note"] = "max open trades (post-lock)"; log_decision(symbol, score, "hold", False); return report

            balances = exchange.fetch_balance()
            usdt = float(balances.get('free', {}).get('USDT', 0.0))
            if usdt - RESERVE_USDT < MIN_NOTIONAL:
                report["note"] = "insufficient USDT (post-lock)"; log_decision(symbol, score, "hold", False); return report

            price = fetch_price(symbol)
            if not price or price <= 0:
                report["note"] = "invalid price"; log_decision(symbol, score, "hold", False); return report

            df = fetch_and_prepare_data_hybrid(symbol, limit=200, timeframe=DECISION_TIMEFRAME)
            row = df.iloc[-1]
            atr_abs = float(row['ATR']) if pd.notna(row['ATR']) else price * 0.02

            amount = size_position(price, usdt, conf)
            trade_val = amount * price
            if trade_val < MIN_NOTIONAL and trade_val < (usdt - RESERVE_USDT):
                report["note"] = f"trade value {trade_val:.2f} < MIN_NOTIONAL"; log_decision(symbol, score, "hold", False); return report

            order = execute_order_buy(symbol, amount, {
                'adx': float(row['ADX']) if pd.notna(row['ADX']) else None,
                'rsi': float(row['RSI']) if pd.notna(row['RSI']) else None,
                'rvol': float(row['RVOL10']) if pd.notna(row['RVOL10']) else None,
                'atr': atr_abs, 'score': float(score), 'confidence': int(conf)
            })

            if order:
                report["executed"] = True
                log_decision(symbol, score, "buy", True)
                logger.info(f"{symbol}: BUY @ {order['price']} (conf {conf}%) — {note}")
                dynamic_trailing_stop(symbol, order['filled'], order['price'], order['trade_id'], atr_abs)
            else:
                log_decision(symbol, score, "buy", False)
        return report
    except Exception as e:
        logger.error(f"trade_once error {symbol}: {e}", exc_info=True)
        return report

def print_cycle_summary(decisions: list):
    try:
        balances = exchange.fetch_balance(); usdt = float(balances.get('free', {}).get('USDT', 0.0))
    except Exception: usdt = float('nan')
    open_trades = get_open_trades_count()
    buys = sum(1 for d in decisions if d.get("executed"))
    logger.info("─" * 72)
    logger.info(f"Cycle summary | USDT={usdt:.2f} | open_trades={open_trades} | buys_this_cycle={buys}")
    for d in decisions:
        logger.info(f"  {d['symbol']:<8} action={d['action']:<4} exec={str(d['executed']):<5} conf={d['confidence']:<3} score={d['score']:.1f} note={d['note']}")
    logger.info("─" * 72)

def write_status_json(decisions: list, path: str = "status.json"):
    try:
        try:
            balances = exchange.fetch_balance(); usdt = float(balances.get('free', {}).get('USDT', 0.0))
        except Exception: usdt = None
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "open_trades": get_open_trades_count(),
            "usdt": usdt,
            "score_gate": get_score_gate(),
            "decisions": decisions,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        logger.warning(f"write_status_json error: {e}")

# =========================
# Reconcile orphans
# =========================
def reconcile_orphan_open_buys():
    try:
        try: exchange.load_markets()
        except Exception as e: logger.warning(f"load_markets warn (reconcile): {e}")

        balances = exchange.fetch_balance() or {}; free = balances.get("free", {}) or {}
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("SELECT symbol, trade_id, amount FROM transactions WHERE side='buy' AND status='open'")
        rows = cur.fetchall()

        closed = 0
        for symbol, trade_id, buy_amt in rows:
            base = symbol.split('/')[0]
            held = float(free.get(base, 0) or 0.0)
            min_amt = None
            try:
                if symbol in getattr(exchange, "markets", {}):
                    min_amt = (exchange.markets[symbol].get("limits", {}) or {}).get("amount", {}).get("min", None)
            except Exception: pass
            dust_eps = float(min_amt) if min_amt else 1e-10
            if held <= dust_eps:
                cur.execute("UPDATE transactions SET status='closed_orphan' WHERE trade_id=? AND side='buy'", (trade_id,))
                conn.commit(); closed += 1
                send_telegram_message(f"🧹 *Startup Reconcile*: Cerrado en BD trade `{trade_id}` ({symbol}) por falta de balance (status=closed_orphan)")
                logger.info(f"Reconciled orphan BUY -> closed_orphan: {trade_id} {symbol}")
        conn.close()
        if closed: logger.info(f"Reconcile: {closed} BUYs huérfanos cerrados.")
    except Exception as e:
        logger.error(f"reconcile_orphan_open_buys error: {e}", exc_info=True)

# =========================
# STARTUP CLEANUP
# =========================
def startup_cleanup():
    """
    On start:
      - Sell non-USDT balances (if value >= MIN_NOTIONAL).
      - If a BUY 'open' exists for that symbol, reuse its trade_id and close it.
      - Insert SELL row; Telegram; capture exit features; analyze if related BUY exists.
      - Reconcile 'open' BUYs with zero real balance -> 'closed_orphan'.
    """
    logger.info("Startup cleanup: closing non-USDT balances and syncing DB states...")
    try:
        try: exchange.load_markets()
        except Exception as e: logger.warning(f"load_markets warning: {e}")

        balances = exchange.fetch_balance() or {}; free = balances.get("free", {}) or {}
        non_usdt = {a: amt for a, amt in free.items() if a != "USDT" and amt and amt > 0}

        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()

        for asset, amt in non_usdt.items():
            symbol = f"{asset}/USDT"
            if symbol not in getattr(exchange, "markets", {}):
                logger.info(f"Skip {asset}: market {symbol} not found"); continue

            mkt = exchange.markets[symbol]
            min_amt = None
            try: min_amt = (mkt.get("limits", {}) or {}).get("amount", {}).get("min", None)
            except Exception: pass
            if min_amt and amt < float(min_amt):
                logger.info(f"Skip {symbol}: amount {amt} < min {min_amt}"); continue

            px = fetch_price(symbol)
            if not px or px <= 0: logger.info(f"Skip {symbol}: no price on cleanup"); continue

            try: adj_amt = float(exchange.amount_to_precision(symbol, amt))
            except Exception: adj_amt = float(amt)
            if adj_amt <= 0: logger.info(f"Skip {symbol}: adjusted amount <= 0"); continue

            trade_value = adj_amt * px
            if trade_value < MIN_NOTIONAL:
                logger.info(f"Skip {symbol}: value {trade_value:.2f} < MIN_NOTIONAL {MIN_NOTIONAL}"); continue

            try:
                order = exchange.create_market_sell_order(symbol, adj_amt)
                sell_price = order.get("price", px) or px
                ts = datetime.now(timezone.utc).isoformat()

                cur.execute("""
                    SELECT trade_id FROM transactions
                    WHERE symbol=? AND side='buy' AND status='open'
                    ORDER BY ts ASC LIMIT 1
                """, (symbol,))
                row = cur.fetchone()
                trade_id = row[0] if row else f"startup_{asset}"

                cur.execute("""
                    INSERT INTO transactions (symbol, side, price, amount, ts, trade_id, status)
                    VALUES (?, 'sell', ?, ?, ?, ?, 'closed')
                """, (symbol, float(sell_price), float(adj_amt), ts, trade_id))

                if row:
                    cur.execute("UPDATE transactions SET status='closed' WHERE trade_id=? AND side='buy'", (trade_id,))
                conn.commit()

                send_telegram_message(f"🔒 *Startup Cleanup*: Vendido `{symbol}` cantidad `{adj_amt}` a `{sell_price}`")

                exit_feats = build_rich_features(symbol)
                save_trade_features(trade_id, symbol, 'exit', exit_feats)

                if row:
                    analyze_and_learn(trade_id, sell_price)

                logger.info(f"Startup sold {symbol}: amt={adj_amt}, px={sell_price}, trade_id={trade_id}")
            except Exception as e:
                logger.error(f"Error selling {symbol} in startup_cleanup: {e}", exc_info=True)

        conn.close()
        reconcile_orphan_open_buys()
        logger.info("Startup cleanup done.")
    except Exception as e:
        logger.error(f"startup_cleanup fatal: {e}", exc_info=True)

# =========================
# Main loop
# =========================
if __name__ == "__main__":
    logger.info("Starting hybrid trader...")
    try:
        try: exchange.load_markets()
        except Exception as e: logger.warning(f"load_markets warning: {e}")

        startup_cleanup()

        while True:
            check_log_rotation()
            cycle_decisions = []
            for sym in SELECTED_CRYPTOS:
                rep = trade_once_with_report(sym)
                cycle_decisions.append(rep)
                time.sleep(1)   # pacing
            controller_autotune()         # adjust score gate to keep buys flowing
            print_cycle_summary(cycle_decisions)
            write_status_json(cycle_decisions)
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down gracefully.")
        logging.shutdown()
        for h in logger.handlers[:]:
            try: h.close()
            except Exception: pass
            logger.removeHandler(h)
