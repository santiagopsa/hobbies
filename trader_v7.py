#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Crypto Spot Trader (Binance via ccxt) — r4.2 (Volatility-aware)

Cambios r4.2 sobre r4.1:
- Perfilado de volatilidad por símbolo (stable / medium / unstable) vía ATR% 30D.
- Umbrales por régimen: RSI/ADX/RVOL/score_gate_offset y re-entry pad dinámico.
- Confirmación reforzada en inestables (EMA50 + VWAP + pad sobre último exit).
- HARD block extra: si 4h RSI alto y 15m no muestra empuje real (MACDh>0 y RVOL15≥1.3) → HOLD.
- Sizing y trailing/TP adaptativos a la volatilidad.
- Min-hold antes de permitir exit por precio/indicadores (evita salidas de 3s).
- Mantiene tus flujos, aprendizaje, controller, startup_cleanup, reconcile, etc.
"""

import os, time, json, threading, sqlite3, logging, logging.handlers, random, math
from datetime import datetime, timezone, date
import ccxt, numpy as np, pandas as pd, pandas_ta as ta, requests
from dotenv import load_dotenv
from scipy.stats import linregress
from typing import Tuple

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
RISK_FRACTION = 0.18  # base fraction per trade (modulated)

# Decision params (defaults)
DECISION_TIMEFRAME = "1h"
SPREAD_MAX_PCT_DEFAULT = 0.005   # 0.5%
MIN_QUOTE_VOL_24H_DEFAULT = 3_000_000

ADX_MIN_DEFAULT = 20
RSI_MIN_DEFAULT, RSI_MAX_DEFAULT = 45, 72
RVOL_BASE_DEFAULT = 1.5
SCORE_GATE_START = 4.0
SCORE_GATE_MAX = 6.0
SCORE_GATE_HARD_MIN = 2.0  # floor del controller

# Learning ranges & rates
ADX_MIN_RANGE = (15, 35)
RSI_MAX_RANGE = (60, 80)
RVOL_BASE_RANGE = (1.2, 3.0)
LEARN_RATE = 0.15
LEARN_MIN_SAMPLES = 6
LEARN_DAILY_CLIP = 0.5
MEAN_REV_WEIGHT = 0.02

# Buy-flow controller
TARGET_BUY_RATIO = 0.3
BUY_RATIO_DELTA = 0.05
GATE_STEP = 0.15
EPSILON_EXPLORE = 0.03
DECISION_WINDOW = 120

# Adaptive jitter
JITTER_BASE = EPSILON_EXPLORE
JITTER_MAX = 0.08

# --- Fees / edge guard ---
# Taker fee in basis points (0.10% => 10 bps). Override with env if you have VIP/BNB discounts.
FEE_BPS_PER_SIDE = float(os.getenv("FEE_BPS_PER_SIDE", "10"))
# How much edge over round-trip cost we demand before allowing non-stop exits.
# 1.3 means 30% safety margin over fees to cover spread + slippage.
EDGE_SAFETY_MULT = float(os.getenv("EDGE_SAFETY_MULT", "1.3"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Exit mode & cadence
EXIT_MODE = os.getenv("EXIT_MODE", "hybrid")  # "price" | "indicators" | "hybrid"
EXIT_CHECK_EVERY_SEC = int(os.getenv("EXIT_CHECK_EVERY_SEC", "30"))
TRAIL_TP_MULT = float(os.getenv("TRAIL_TP_MULT", "1.05"))
# >>> PATCH START: REBOUND/TRAIL/TIME knobs
# --- Exit anti-whipsaw / rebound guard ---
REBOUND_GUARD_ENABLED = True
REBOUND_WAIT_BARS_15M = 2          # espera confirmación 2 velas 15m antes de vender
REBOUND_MIN_RSI_BOUNCE = 5.0       # cancel exit si RSI15 sube >= +5
REBOUND_EMA_RECLAIM = True         # cancel exit si cierra sobre EMA20 15m
REBOUND_USE_5M_DIVERGENCE = True   # opcional: divergencia alcista 5m cancela sell

# --- Volatility grace / trailing hysteresis ---
RVOL_SPIKE_GRACE = True
RVOL_SPIKE_THRESHOLD = 2.0         # si RVOL10 en últimas 3 velas > 2 → ampliar trailing
RVOL_SPIKE_TRAIL_BONUS = 0.5       # +0.5% al trail por 15 minutos

# --- Time-in-trade stop con excepción de "tape mejorando" ---
TIME_STOP_HOURS = 6
TIME_STOP_EXTEND_HOURS = 3
TAPE_IMPROVING_ADX_SLOPE_MIN = 0.0 # >0 = ADX1h subiendo
TAPE_IMPROVING_VWAP_REQ = True     # close 1h > VWAP 1h para extender

# --- Profit quality guard (evitar micro-takes post-fees) ---
MIN_GAIN_OVER_FEES_MULT = 1.6      # exigir 1.6× fees totales antes de permitir exit por precio
MIN_HOLD_SECONDS = 900             # 15 min mínimos globales para exits (excepto stop duro)
# >>> PATCH END

# >>> CHAN PATCH CONFIG START (ANCHOR A)
# ——— Chandelier / estructura / BE / time-stop por velas ———
CHAN_ATR_LEN = 22          # ATR para Chandelier (n)
CHAN_LEN_HIGH = 22         # HH(n), típico 22
CHAN_K_STABLE  = 3.0
CHAN_K_MEDIUM  = 2.7
CHAN_K_UNSTABLE= 2.3

# Tighten/widen dinámicos del K
SOFT_TIGHTEN_K = 0.5       # reducción temporal de k al detectar debilidad leve (RSI/MACDh)
RVOL_SPIKE_K_BONUS = 0.4   # aumento temporal de k si hay spike de RVOL (gracia a la volatilidad)
RVOL_K_BONUS_MINUTES = 15

# Break-even cuando el trade alcanza X R
BE_R_MULT = 1            # go BE at 1.0R to reduce scratches

# Dead-tape hard block
DEAD_TAPE_RVOL10_HARD = 0.20


# Tiers para "dejar correr" apretando el k (sin TP duro)
TIER2_R_MULT = 3.0         # a partir de 3R, aprieta más el k
TIER2_K_TIGHTEN = 0.5      # reducción extra de k

# Estructura: Donchian + EMA
DONCHIAN_LEN_EXIT = 20     # N-bar low para salida de tendencia si se pierde estructura

# Time-stop por barras del TF 1h
TIME_STOP_BARS_1H = 12     # si en 12 velas 1h no progresa (o no cumple mejora), salir
TIME_STOP_EXTEND_BARS = 6  # prórroga si "tape mejorando" (ADX↑ y close>VWAP)
# >>> CHAN PATCH CONFIG END


# Momentum penalties/blocks
RSI_OVERBOUGHT_4H = 78.0
PENALTY_4H_RSI = 0.7
PENALTY_15M_WEAK = 0.5

# Cooldowns / re-entry
COOLDOWN_MIN_AFTER_LOSS = 10
POST_LOSS_SIZING_WINDOW_SEC = 2 * 3600
POST_LOSS_CONF_CAP = 78
POST_LOSS_SIZING_FACTOR = 0.70
REENTRY_BLOCK_MIN = 5
REENTRY_ABOVE_LAST_EXIT_PAD = 0.0015  # 0.15% (por defecto; se ajusta por régimen)

# Volatility / crash protection
INIT_STOP_ATR_MULT = 1.4
PORTFOLIO_MAX_UTIL = 0.75
CRASH_HALT_DROP_PCT = 3.5
CRASH_HALT_WINDOW_MIN = 15

# Crash-halt per-symbol override
CRASH_HALT_ENABLE_OVERRIDE = True
CRASH_HALT_OVERRIDE_MIN_ADX15 = 27.0
CRASH_HALT_OVERRIDE_REQUIRE_EMA20 = True

# RVOL sanity
RVOL_MEAN_MIN = 1e-6
RVOL_VALUE_MIN = 0.25

# No-buy-under-sell thresholds
NBUS_RSI15_OB = 75.0
NBUS_MACDH15_NEG = -0.002
# Logger
logger = logging.getLogger("hybrid_trader")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
file_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_PATH, when="midnight", interval=1, backupCount=14, utc=True
)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(fmt); logger.addHandler(file_handler)
console = logging.StreamHandler(); console.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
console.setFormatter(fmt); logger.addHandler(console)

def check_log_rotation(max_size_mb=1):
    try:
        if os.path.getsize(file_handler.baseFilename) > max_size_mb * 1024 * 1024:
            file_handler.doRollover()
    except Exception:
        pass

# =========================
# Runtime state
# =========================
LAST_LOSS_INFO = {}           # symbol -> {"ts": iso, "pnl_usdt": float}
LAST_SELL_INFO = {}           # symbol -> {"ts": iso, "price": float, "source": "trail|manual|startup"}
TRADE_ANALYTICS_COUNT = {}    # base -> int
LAST_PARAM_UPDATE = {}        # base -> yyyy-mm-dd

LAST_TRADE_CLOSE = {}  # symbol -> ts ISO
POST_TRADE_COOLDOWN_SEC = int(os.getenv("POST_TRADE_COOLDOWN_SEC", "120"))


# =========================
# DB schema (with migration)
# =========================
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
            adx REAL, rsi REAL, rvol REAL, atr REAL,
            score REAL, confidence INTEGER,
            status TEXT DEFAULT 'open'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
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
            adjustments_json TEXT,
            learn_enabled INTEGER DEFAULT 1
        )
    """)

    # MIGRACIÓN: añade learn_enabled si falta
    try:
        cur.execute("PRAGMA table_info(trade_analysis)")
        cols = {r[1] for r in cur.fetchall()}
        if "learn_enabled" not in cols:
            logger.info("Migrating: ADD COLUMN trade_analysis.learn_enabled INTEGER DEFAULT 1")
            cur.execute("ALTER TABLE trade_analysis ADD COLUMN learn_enabled INTEGER DEFAULT 1")
    except Exception as e:
        logger.warning(f"Migration check failed: {e}")

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
# Binance filter helpers
# =========================
def _bn_filters(symbol):
    m = exchange.markets.get(symbol, {})
    step = min_qty = min_notional = None
    try:
        for f in m.get('info', {}).get('filters', []):
            if f.get('filterType') == 'LOT_SIZE':
                step = float(f['stepSize'])
                min_qty = float(f['minQty'])
            if f.get('filterType') in ('NOTIONAL', 'MIN_NOTIONAL'):
                min_notional = float(f.get('minNotional') or f.get('minNotional', 0))
    except Exception:
        pass
    return step, min_qty, min_notional

def _floor_to_step(x, step):
    if not step or step <= 0:
        return x
    return math.floor(x / step) * step

def _prepare_sell_amount(symbol, desired_amt, px):
    step, min_qty, min_notional = _bn_filters(symbol)
    amt = desired_amt * 0.999  # safety buffer
    if step:
        amt = _floor_to_step(amt, step)
    if min_qty and amt < min_qty:
        return 0.0, "amount < minQty"
    if min_notional and px and (amt * px) < min_notional:
        return 0.0, "notional < minNotional"
    return amt, None

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
# >>> PATCH START: REBOUND HELPERS
def _ema(series, length=20):
    try:
        return ta.ema(series, length=length)
    except Exception:
        return None

def bullish_divergence(df, rsi_len=14):
    """
    Divergencia alcista simple en 5m: precio hace low más bajo, RSI hace low más alto.
    df: DataFrame con close; calcula RSI interno.
    """
    try:
        rsi = ta.rsi(df['close'], length=rsi_len)
        c = df['close'].iloc[-20:]
        r = rsi.iloc[-20:]
        # mínimos recientes (asegurando orden temporal)
        p1 = c.idxmin()                        # low más reciente
        p2 = c.iloc[:-1].idxmin()              # low anterior
        if p2 >= p1 and len(c) >= 3:
            p2 = c.iloc[:-2].idxmin()
        if p2 is None or p1 is None:
            return False
        price_lower_low = c.loc[p1] < c.loc[p2]
        rsi_higher_low = r.loc[p1] > r.loc[p2]
        return bool(price_lower_low and rsi_higher_low)
    except Exception:
        return False
# >>> PATCH END

# >>> CHAN HELPERS START (ANCHOR B)
def chandelier_stop_long(df: pd.DataFrame, atr_len=CHAN_ATR_LEN, hh_len=CHAN_LEN_HIGH, k=3.0):
    """
    Chandelier exit clásico para largos: stop = HH(hh_len) - k * ATR(atr_len).
    Usa columnas 'high','low','close' en el TF actual (recomendado 1h).
    """
    if df is None or len(df) < max(atr_len, hh_len) + 2:
        return None
    try:
        atr = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
        hh  = df['high'].rolling(hh_len).max()
        c_stop = hh - k * atr
        v = float(c_stop.iloc[-1])
        return v if np.isfinite(v) else None
    except Exception:
        return None

def donchian_lower(df: pd.DataFrame, length=DONCHIAN_LEN_EXIT):
    """Lower band Donchian: mínimo de 'low' en las últimas N velas."""
    if df is None or len(df) < length + 2:
        return None
    try:
        low_n = df['low'].rolling(length).min().iloc[-1]
        return float(low_n) if np.isfinite(low_n) else None
    except Exception:
        return None

def count_closed_bars_since(df: pd.DataFrame, ts_open: datetime) -> int:
    """# de velas cerradas con índice > ts_open (DataFrame index = timestamps)."""
    if df is None or df.empty:
        return 0
    try:
        return int((df.index > pd.Timestamp(ts_open)).sum())
    except Exception:
        return 0
# >>> CHAN HELPERS END


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

def _fmt(v, decimals=2, pct=False, suffix=""):
    try:
        if v is None or (isinstance(v, float) and (np.isnan(v) or not np.isfinite(v))):
            return "—"
        if pct:
            return f"{float(v):.{decimals}f}%"
        return f"{float(v):.{decimals}f}{suffix}"
    except Exception:
        return "—"

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

        rvv = rvol10.iloc[-1]
        rv_ok = not pd.isna(rvv) and np.isfinite(rvv) and rv_mean.iloc[-1] and rv_mean.iloc[-1] > RVOL_MEAN_MIN
        rvv = float(rvv) if rv_ok else None

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
            "rvol10": rvv,
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
    sup, sup_dist = (None, None)
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

def quick_tf_snapshot(symbol: str, timeframe: str, limit: int = 120):
    ohlcv = fetch_ohlcv_with_retry(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv: return {}
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce').ffill().bfill()
    snap = {}
    try: snap['EMA20'] = float(ta.ema(df['close'], length=20).iloc[-1])
    except Exception: pass
    try: snap['RSI'] = float(ta.rsi(df['close'], length=14).iloc[-1])
    except Exception: pass
    try:
        macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
        snap['MACDh'] = float(macd_df['MACDh_12_26_9'].iloc[-1])
    except Exception: pass
    try:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        snap['ADX'] = float(adx_df['ADX_14'].iloc[-1])
    except Exception: pass
    try:
        rv_mean = df['volume'].rolling(10).mean().iloc[-1]
        snap['RVOL10'] = float(df['volume'].iloc[-1] / rv_mean) if rv_mean and rv_mean > RVOL_MEAN_MIN else None
    except Exception: pass
    try: snap['last'] = float(df['close'].iloc[-1])
    except Exception: pass
    try:
        snap['VWAP'] = float(ta.vwap(df['high'], df['low'], df['close'], df['volume']).iloc[-1])
    except Exception: pass
    try:
        snap['candle_ts'] = df.index[-1].isoformat()
    except Exception: pass
    return snap

# =========================
# Volatility profiling (NEW)
# =========================
VOL_CACHE = {}  # symbol -> {"class": "stable|medium|unstable", "atrp_30d": float, "ts": iso}

def _atrp_median(df: pd.DataFrame) -> float:
    atr = ta.atr(df['high'], df['low'], df['close'], length=14)
    if atr is None or atr.empty: return None
    atrp = (atr / df['close'] * 100.0).dropna()
    return float(atrp.tail(200).median()) if len(atrp) else None

def classify_symbol(symbol: str) -> dict:
    now = datetime.now(timezone.utc)
    c = VOL_CACHE.get(symbol)
    if c and (now - datetime.fromisoformat(c["ts"])).total_seconds() < 3600:
        return c
    df = fetch_and_prepare_df(symbol, "4h", limit=400)
    if df is None or len(df) < 120:
        cls = {"class": "medium", "atrp_30d": None, "ts": now.isoformat()}
        VOL_CACHE[symbol] = cls; return cls
    atrp = _atrp_median(df)
    if atrp is None:
        klass = "medium"
    elif atrp < 1.2:
        klass = "stable"
    elif atrp > 2.5:
        klass = "unstable"
    else:
        klass = "medium"
    cls = {"class": klass, "atrp_30d": atrp, "ts": now.isoformat()}
    VOL_CACHE[symbol] = cls
    return cls

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
    if not row or row[0] is None:
        return SCORE_GATE_START
    return max(float(row[0]), SCORE_GATE_HARD_MIN)

def set_score_gate(new_gate: float):
    ng = clamp(float(new_gate), SCORE_GATE_MIN, SCORE_GATE_MAX)
    ng = max(ng, SCORE_GATE_HARD_MIN)
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("UPDATE controller SET score_gate=?, updated_at=? WHERE id=1",
                (ng, datetime.now(timezone.utc).isoformat()))
    conn.commit(); conn.close()
    return ng

def log_decision(symbol: str, score: float, action: str, executed: bool):
    try:
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("INSERT INTO decision_log (ts, symbol, score, action, executed) VALUES (?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), symbol, float(score), action, 1 if executed else 0))
        cur.execute("SELECT COUNT(*) FROM decision_log"); n = cur.fetchone()[0]
        if n > DECISION_WINDOW * 8:
            cur.execute("DELETE FROM decision_log WHERE id IN (SELECT id FROM decision_log ORDER BY id ASC LIMIT ?)", (n - DECISION_WINDOW*8,))
        conn.commit(); conn.close()
    except Exception as e:
        logger.debug(f"log_decision error: {e}")

def controller_autotune():
    try:
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("SELECT executed FROM decision_log ORDER BY id DESC LIMIT ?", (DECISION_WINDOW,))
        rows = cur.fetchall(); conn.close()
        if not rows or len(rows) < max(20, DECISION_WINDOW // 4):
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

def _recent_buy_ratio():
    try:
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("SELECT executed FROM decision_log ORDER BY id DESC LIMIT ?", (DECISION_WINDOW,))
        rows = cur.fetchall(); conn.close()
        if not rows:
            return 0.0
        return sum(1 for (e,) in rows if e == 1) / len(rows)
    except Exception:
        return 0.0

# =========================
# Data prep
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
    rv_ok = rv_mean.iloc[-1] and rv_mean.iloc[-1] > RVOL_MEAN_MIN
    df['RVOL10'] = (df['volume'] / rv_mean) if rv_ok else np.nan
    if len(df) >= 10:
        x = np.arange(10)
        df.loc[df.index[-1], 'PRICE_SLOPE10'] = linregress(x, df['close'].iloc[-10:]).slope
        df.loc[df.index[-1], 'VOL_SLOPE10']   = linregress(x, df['volume'].iloc[-10:]).slope
    else:
        df['PRICE_SLOPE10'] = np.nan; df['VOL_SLOPE10'] = np.nan
    try:
        df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    except Exception:
        df['VWAP'] = np.nan
    return df

def estimate_slippage_pct(symbol: str, notional: float = MIN_NOTIONAL):
    ob = fetch_order_book_safe(symbol, limit=50)
    try:
        best_ask = ob['asks'][0][0]; best_ask_vol = ob['asks'][0][1]
        best_bid = ob['bids'][0][0]; best_bid_vol = ob['bids'][0][1]
        spr_pct = percent_spread(ob) * 100.0  # p.p.

        order_qty = notional / max(best_ask, 1e-9)
        # Bump en puntos porcentuales (cap 0.15%)
        bump_pct = min(0.15, (order_qty / (best_ask_vol + 1e-9)) * 0.2)

        return spr_pct + bump_pct
    except Exception:
        return 0.05

    
def required_edge_pct() -> float:
    """Minimum pct gain vs entry to overcome round-trip fees (+ safety)."""
    rt_fee = 2.0 * (FEE_BPS_PER_SIDE / 10000.0)  # e.g. 0.002 for 20 bps
    return rt_fee * EDGE_SAFETY_MULT * 100.0     # return in %


# =========================
# No-Buy-Under-Sell gate
# =========================
# === ANCHOR-NBUS: simplified NBUS gate (no ADX hard block) ===
def is_selling_condition_now(symbol: str) -> Tuple[bool, str]:
    """
    NBUS (No-Buy-Under-Sell) — only block obvious 'buy-the-top' traps.
    - Block if 15m is *clearly* overheated AND rolling over: RSI15 >= 75 AND MACDh15 <= -0.002
    - DO NOT block on 1h ADX < 20 (we already evaluate trend strength elsewhere)
    """
    try:
        df15 = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=80)
        if df15 is None or len(df15) == 0:
            return False, "no 15m data"

        rsi15 = None
        macdh15 = None

        # RSI15
        try:
            if 'RSI' in df15 and not pd.isna(df15['RSI'].iloc[-1]):
                rsi15 = float(df15['RSI'].iloc[-1])
        except Exception:
            rsi15 = None

        # MACDh15
        try:
            macd15 = ta.macd(df15['close'], fast=12, slow=26, signal=9)
            if macd15 is not None and not macd15.empty and 'MACDh_12_26_9' in macd15:
                mh = macd15['MACDh_12_26_9'].iloc[-1]
                if not pd.isna(mh):
                    macdh15 = float(mh)
        except Exception:
            macdh15 = None

        # Only one blocking rule remains (tougher than before)
        if (rsi15 is not None and rsi15 >= 75.0) and (macdh15 is not None and macdh15 <= -0.002):
            return True, f"15m overheated & rolling over (RSI15={rsi15:.1f}, MACDh15={macdh15:.3f})"

        return False, "ok"
    except Exception as e:
        logger.debug(f"is_selling_condition_now error {symbol}: {e}")
        return False, "error"

# =========================
# Decision (volatility aware)
# =========================
def hybrid_decision(symbol: str):
    base = symbol.split('/')[0]
    lp = get_learn_params(base)

    # ===== learned/base thresholds =====
    RSI_MIN = lp["rsi_min"]; RSI_MAX = lp["rsi_max"]
    ADX_MIN = lp["adx_min"]; RVOL_BASE = lp["rvol_base"]

    # ===== volatility profile (ANCHOR-6: soften "unstable") =====
    prof = classify_symbol(symbol)
    klass = prof["class"]

    RSI_MIN_K, RSI_MAX_K = RSI_MIN, RSI_MAX
    ADX_MIN_K = ADX_MIN
    RVOL_BASE_K = RVOL_BASE
    SCORE_GATE_OFFSET = 0.0
    REENTRY_PAD = REENTRY_ABOVE_LAST_EXIT_PAD

    if klass == "stable":
        RSI_MIN_K, RSI_MAX_K = max(RSI_MIN, 52), min(RSI_MAX, 68)
        ADX_MIN_K = max(18, ADX_MIN - 2)
        RVOL_BASE_K = max(0.8, RVOL_BASE * 0.8)
        SCORE_GATE_OFFSET = -0.2
        REENTRY_PAD = 0.0015
    elif klass == "unstable":
        # soften vs. your previous +0.3 gate bump and tighter RSI
        RSI_MIN_K, RSI_MAX_K = max(48, RSI_MIN - 2), min(66, RSI_MAX)   # was 65
        ADX_MIN_K = max(23, ADX_MIN)                                     # was 25
        RVOL_BASE_K = max(1.05, RVOL_BASE)                               # was >=1.1
        SCORE_GATE_OFFSET = +0.1                                         # was +0.3
        REENTRY_PAD = 0.0030                                             # was 0.0035

    blocks = []
    level = "NONE"  # NONE | SOFT | HARD

    # ===== liquidity / spread guards (unchanged) =====
    try:
        t = exchange.fetch_ticker(symbol)
        qvol = float(t.get('quoteVolume', 0.0) or 0.0)
        if qvol < MIN_QUOTE_VOL_24H_DEFAULT:
            blocks.append(f"low 24h quote vol: {qvol:.0f}")
            level = "HARD"
    except Exception as e:
        blocks.append(f"ticker error: {e}")
        level = "HARD"

    try:
        ob = exchange.fetch_order_book(symbol, limit=50)
        spr_p = percent_spread(ob)
        if spr_p > SPREAD_MAX_PCT_DEFAULT:
            blocks.append("spread too wide")
            level = "HARD"
    except Exception as e:
        blocks.append(f"orderbook error: {e}")
        level = "HARD"
        spr_p = 0.0

    # ===== core TF (1h) =====
    df = fetch_and_prepare_data_hybrid(symbol)
    if df is None or len(df) < 60:
        blocks.append("not enough candles")
        level = "HARD"
        return "hold", 50, 0.0, " | ".join(blocks)

    row = df.iloc[-1]
    score_gate = get_score_gate() + SCORE_GATE_OFFSET

    # ===== anti-scalp vs fees (ANCHOR-7: keep but a touch looser) =====
    atr_abs_now = float(row['ATR']) if pd.notna(row['ATR']) else None
    atr_pct_now = (atr_abs_now / row['close'] * 100.0) if (atr_abs_now and row['close']) else None
    if atr_pct_now is not None:
        needed = required_edge_pct() * 1.15  # was 1.2
        if atr_pct_now < needed:
            blocks.append(f"anti-scalp: ATR% {atr_pct_now:.2f} < needed {needed:.2f}")
            if level != "HARD": level = "SOFT"

    # ===== local 1h features =====
    in_lane = bool(row['EMA20'] > row['EMA50'] and row['close'] > row['EMA20'])
    adx = float(row['ADX']) if pd.notna(row['ADX']) else 0.0
    rvol_1h = float(row['RVOL10']) if pd.notna(row['RVOL10']) else None
    price_slope = float(row.get('PRICE_SLOPE10', 0.0) or 0.0)

    # ===== fast snapshots (needed for anchors 2,8) =====
    tf15 = quick_tf_snapshot(symbol, '15m', limit=120)
    tf4h = quick_tf_snapshot(symbol, '4h',  limit=120)
    rv15 = tf15.get('RVOL10')

    # ===== 15m weakness/4h OB hard-blocks (keep) =====
    try:
        rsi_15 = tf15.get('RSI'); macdh_15 = tf15.get('MACDh')
        if (macdh_15 is not None and macdh_15 < 0) and (rsi_15 is not None and rsi_15 < 48):
            blocks.append("HARD block: 15m weakness")
            level = "HARD"
    except Exception:
        pass
    try:
        rsi_4h = tf4h.get('RSI')
        if rsi_4h is not None and rsi_4h >= (RSI_OVERBOUGHT_4H + 2.0):
            blocks.append("HARD block: 4h too overbought")
            level = "HARD"
        if klass == "unstable" and rsi_4h is not None and rsi_4h >= 68:
            rv_ok = (rv15 or 0) >= 1.3
            mh_ok = (tf15.get('MACDh') or 0) > 0
            if not (mh_ok and rv_ok):
                blocks.append("HARD block: 4h high RSI but 15m lacks RVOL/MACDh")
                level = "HARD"
    except Exception:
        pass

    # ===== NBUS rework (ANCHOR-2): relax ADX and OB gate =====
    # old call removed; inline light version:
    # ===== NBUS gate (use helper) =====
    try:
        sell_cond, reason = is_selling_condition_now(symbol)
        if sell_cond:
            blocks.append(f"NBUS: {reason}")
            level = "HARD"
    except Exception as e:
        logger.debug(f"NBUS helper error {symbol}: {e}")


    # ===== post-loss cooldown (ANCHOR-3: softer & more permissive) =====
    try:
        info = LAST_LOSS_INFO.get(symbol)
        if info and info.get("ts"):
            now = datetime.now(timezone.utc)
            last_dt = datetime.fromisoformat(info["ts"].replace("Z",""))
            within_cooldown = last_dt and (now - last_dt).total_seconds() < COOLDOWN_MIN_AFTER_LOSS * 60
            if within_cooldown:
                # easier override: RVOL15 > 1.5 or ADX15 > 30 and MACDh15 > 0
                df15_full = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=120)
                post_loss_override = False
                if df15_full is not None and len(df15_full) > 30:
                    adx15_df = ta.adx(df15_full['high'], df15_full['low'], df15_full['close'], length=14)
                    adx15 = float(adx15_df['ADX_14'].iloc[-1]) if adx15_df is not None else None
                    rvol15_full = None
                    try:
                        rv_mean = df15_full['volume'].rolling(10).mean().iloc[-1]
                        if rv_mean and rv_mean > RVOL_MEAN_MIN:
                            rvol15_full = float(df15_full['volume'].iloc[-1] / rv_mean)
                    except Exception:
                        pass
                    macd15 = ta.macd(df15_full['close'], fast=12, slow=26, signal=9)
                    macdh15 = float(macd15['MACDh_12_26_9'].iloc[-1]) if macd15 is not None else None
                    post_loss_override = (
                        (rvol15_full is not None and rvol15_full > 1.5) or
                        ((adx15 or 0) > 30 and (macdh15 or 0) > 0)
                    )
                if not post_loss_override:
                    blocks.append("post-loss cooldown")
                    level = "HARD"
    except Exception:
        pass

    # ===== re-entry block (ANCHOR-4: shorter) =====
    try:
        info = LAST_SELL_INFO.get(symbol)
        if info and info.get("ts"):
            last_dt = datetime.fromisoformat(info["ts"].replace("Z",""))
            if last_dt and (datetime.now(timezone.utc) - last_dt).total_seconds() < (REENTRY_BLOCK_MIN * 60):
                blocks.append(f"re-entry block ({REENTRY_BLOCK_MIN}m)")
                level = "HARD"
    except Exception:
        pass

    # ===== confirmation (EMA/VWAP/exitPad) — keep, but less strict on VWAP for stable =====
    try:
        last = float(row['close'])
        ema20_1h = float(row['EMA20']) if pd.notna(row['EMA20']) else None
        ema50_1h = float(row['EMA50']) if pd.notna(row['EMA50']) else None
        vwap_1h = float(row['VWAP']) if pd.notna(row['VWAP']) else None

        ema_ok = (ema20_1h is None or last > ema20_1h)
        if klass == "unstable":
            ema_ok = ema_ok and (ema50_1h is None or last > ema50_1h)

        # allow dip-buys on stable regime even if slightly below VWAP
        if klass == "stable":
            vwap_ok = True if vwap_1h is None else (last >= vwap_1h * 0.998)
        else:
            vwap_ok = (vwap_1h is None) or (last >= vwap_1h)

        le = LAST_SELL_INFO.get(symbol)
        exit_ok = True
        if le and le.get("price") is not None:
            exit_ok = last >= float(le["price"]) * (1.0 + REENTRY_PAD)

        if not (ema_ok and vwap_ok and exit_ok):
            blocks.append("confirmation failed (EMA/VWAP/exitPad)")
            if level != "HARD": level = "SOFT"
    except Exception:
        pass

    # ===== lane/override logic (ANCHOR-5: make override a bit easier) =====
    override = (not in_lane) and (adx >= ADX_MIN_K + 8) and ((rvol_1h or 0) >= (RVOL_BASE_K * 1.4)) and (price_slope > 0)
    if not (in_lane or override):
        blocks.append("not in uptrend lane; no override")
        if level != "HARD": level = "SOFT"

    # ===== scoring =====
    score = 0.0; notes = []

    # RVOL contribution uses ANY of (1h,15m) so early wake-ups don’t get punished (ANCHOR-8)
    rvol_any = None
    for _rv in (rvol_1h, rv15):
        if _rv is not None and np.isfinite(_rv):
            rvol_any = _rv if (rvol_any is None) else max(rvol_any, _rv)

    if rvol_any is not None and rvol_any >= RVOL_BASE_K:
        score += 2.0; notes.append(f"RVOL≥{RVOL_BASE_K:.2f} ({rvol_any:.2f})")
        if rvol_any >= RVOL_BASE_K + 0.5: score += 1.0
        if rvol_any >= RVOL_BASE_K + 1.5: score += 1.0

    if price_slope > 0:
        score += 1.0; notes.append("price slope>0")

    rsi = float(row['RSI']) if pd.notna(row['RSI']) else 50.0
    if RSI_MIN_K <= rsi <= RSI_MAX_K:
        score += 1.0; notes.append(f"RSI in band ({rsi:.1f})")
    if rsi >= (RSI_MIN_K + RSI_MAX_K)/2:
        score += 0.5

    if adx >= ADX_MIN_K:
        score += 1.0; notes.append(f"ADX≥{ADX_MIN_K:.0f} ({adx:.1f})")

    vol_slope = float(row.get('VOL_SLOPE10', 0.0) or 0.0)
    if vol_slope > 0: score += 0.5

    try:
        spr_pct = spr_p * 100.0
        if spr_pct > 0.10:
            score -= 0.2; notes.append(f"spread penalty {spr_pct:.2f}%")
    except Exception:
        pass

    # jitter (unchanged)
    try:
        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("SELECT executed FROM decision_log ORDER BY id DESC LIMIT ?", (DECISION_WINDOW,))
        rows = cur.fetchall(); conn.close()
        execs = sum(1 for (e,) in rows if e == 1)
        ratio_recent = execs / len(rows) if rows else 0.0
    except Exception:
        ratio_recent = TARGET_BUY_RATIO
    explore_p = JITTER_BASE
    if ratio_recent < (TARGET_BUY_RATIO * 0.6):
        explore_p = min(JITTER_MAX, JITTER_BASE + 0.03)
    if random.random() < explore_p:
        jitter = random.uniform(-0.3, 0.3)
        score += jitter
        notes.append(f"jitter {jitter:+.2f} (p={explore_p:.2f})")

    raw_score = max(0.0, score)

    # executability and starvation logic
    buy_ratio = _recent_buy_ratio()
    starvation = buy_ratio < TARGET_BUY_RATIO
    relax_soft = starvation  # can set True to force testing

    if level == "HARD":
        exec_allowed = False
    elif level == "SOFT":
        exec_allowed = relax_soft
        if exec_allowed: notes.append("soft override (starvation)")
    else:
        exec_allowed = True

    score_for_gate = raw_score if exec_allowed else 0.0

    # ANCHOR-8b: skip gate bump if 15m is waking up
    fifteen_waking = (rv15 is not None and rv15 >= 0.9 and (tf15.get('MACDh') or 0) > 0)
    if ((rvol_1h is None) or (rvol_1h < RVOL_BASE_K)) and not fifteen_waking:
        score_gate += 0.3

    # final action
    gate = score_gate
    if exec_allowed and raw_score >= gate:
        action = "buy"
    else:
        action = "hold"

    conf = int(clamp(50 + 10*(1 if row['EMA20'] > row['EMA50'] else 0) + 5*(1 if price_slope>0 else 0), 0, 100))
    if symbol in LAST_LOSS_INFO:
        conf = min(conf, POST_LOSS_CONF_CAP)

    note = f"raw={raw_score:.1f} gate={gate:.1f} score={score_for_gate:.1f} | " + ", ".join(notes)
    note += f" | class={klass}"
    if blocks:
        note += f" | blocks=[{'; '.join(blocks)}] level={level} exec={exec_allowed}"

    return action, conf, score_for_gate, note

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

        def fmt_pct(x): 
            try: return f"{float(x):.2f}%"
            except: return "—"

        summary = (
            f"*{side.upper()} FEATURES* {symbol}\n"
            f"Trade: `{trade_id}`\n"
            f"{extra_lines}"
            f"—\n"
            f"*Market*\n"
            f"Last: `{gen.get('last')}`  24hQVol: `{abbr(gen.get('quote_volume_24h'))}`  Spread: `{fmt_pct((ob.get('spread_pct') or 0)*100)}`\n"
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

def sell_symbol(symbol: str, amount: float, trade_id: str, source: str = "trail"):
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

        LAST_SELL_INFO[symbol] = {"ts": sell_ts, "price": float(sell_price), "source": source}
        LAST_TRADE_CLOSE[symbol] = sell_ts

        

        features_exit = build_rich_features(symbol)
        save_trade_features(trade_id, symbol, 'exit', features_exit)

        # EXIT FEATURES
        try:
            gen = features_exit.get("general", {})
            ob  = features_exit.get("orderbook", {})
            tf1h = features_exit.get("timeframes", {}).get("1h", {})
            tf15 = features_exit.get("timeframes", {}).get("15m", {})
            tf4h = features_exit.get("timeframes", {}).get("4h", {})

            exit_summary = (
                f"📤 *EXIT FEATURES* {symbol}\n"
                f"Trade: `{trade_id}`  |  Price: `{_fmt(sell_price)}`  Amount: `{_fmt(amount)}`\n"
                f"—\n"
                f"*Market*\n"
                f"Last:`{_fmt(gen.get('last'))}`  24hQVol:`{abbr(gen.get('quote_volume_24h'))}`  "
                f"Spread:`{_fmt((ob.get('spread_pct') or 0)*100, pct=True)}`  Imb:`{_fmt(ob.get('imbalance'))}`\n"
                f"—\n"
                f"*1h*   RSI:`{_fmt(tf1h.get('rsi'))}`  ADX:`{_fmt(tf1h.get('adx'))}`  RVOL10:`{_fmt(tf1h.get('rvol10'))}`  ATR%:`{_fmt(tf1h.get('atr_pct'))}`\n"
                f"*15m*  RSI:`{_fmt(tf15.get('rsi'))}`  MACDh:`{_fmt(tf15.get('macd_hist'))}`\n"
                f"*4h*   RSI:`{_fmt(tf4h.get('rsi'))}`  ADX:`{_fmt(tf4h.get('adx'))}`\n"
            )
            send_telegram_message(exit_summary)

            send_feature_bundle_telegram(
                trade_id, symbol, 'exit', features_exit,
                extra_lines="(exit snapshot adjunto como JSON)\n"
            )

            logger.info(
                f"[exit-features] {symbol} trade={trade_id} | "
                f"1h RSI={_fmt(tf1h.get('rsi'))} ADX={_fmt(tf1h.get('adx'))} RVOL10={_fmt(tf1h.get('rvol10'))} "
                f"| 15m RSI={_fmt(tf15.get('rsi'))} MACDh={_fmt(tf15.get('macd_hist'))} "
                f"| spread={_fmt((ob.get('spread_pct') or 0)*100, pct=True)} imb={_fmt(ob.get('imbalance'))}"
            )
        except Exception as e:
            logger.warning(f"exit features reporting error {symbol}: {e}")

        analyze_and_learn(trade_id, sell_price, learn_enabled=(source != "startup"))

        logger.info(f"Sold {symbol} @ {sell_price} (trade {trade_id})")
    except Exception as e:
        logger.error(f"sell_symbol error {symbol}: {e}")

def dynamic_trailing_stop(symbol: str, amount: float, purchase_price: float, trade_id: str, atr_abs: float):
    def loop():
        def trade_is_closed() -> bool:
            try:
                conn = sqlite3.connect(DB_NAME)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM transactions WHERE trade_id=? AND side='sell' LIMIT 1", (trade_id,))
                n = cur.fetchone()[0]
                conn.close()
                return n > 0
            except Exception:
                return False

        try:
            if not symbol or amount <= 0 or not purchase_price:
                logger.info(f"[trailing] Invalid params, stopping. sym={symbol} amt={amount} px={purchase_price}")
                return

            # Perfil / parámetros base
            prof_local = classify_symbol(symbol)
            klass_local = prof_local["class"]
            if   klass_local == "unstable": base_k = CHAN_K_UNSTABLE
            elif klass_local == "stable":   base_k = CHAN_K_STABLE
            else:                           base_k = CHAN_K_MEDIUM

            opened_ts = datetime.now(timezone.utc)
            # Stop inicial (protección dura, igual que antes pero explícito)
            initial_stop = purchase_price - (INIT_STOP_ATR_MULT * atr_abs)
            if initial_stop >= purchase_price:
                # fallback por si ATR_abs inválido
                initial_stop = purchase_price * 0.985

            # "R" inicial con ese stop
            initial_R = max(purchase_price - initial_stop, purchase_price * 0.003)  # evita R ~0

            # Estados temporales
            rv_grace_until = None
            soft_tighten_until = None

            # Stop ratcheado (no retrocede)
            stop_price = initial_stop

            # Rebound guard (usa tu lógica ya existente)
            rebound_pending_until = None
            last_exit_trigger_reason = None

            while True:
                if trade_is_closed():
                    logger.info(f"[trailing] {symbol} trade {trade_id} already closed — stopping trailing thread.")
                    return

                price = fetch_price(symbol)
                if not price:
                    time.sleep(EXIT_CHECK_EVERY_SEC)
                    continue

                held = (datetime.now(timezone.utc) - opened_ts).total_seconds()
                held_long_enough = held >= max(60, MIN_HOLD_SECONDS)

                # ===== Cálculos en 1h (estructura, Chandelier, ADX/VWAP/tiempo) =====
                df1h = fetch_and_prepare_data_hybrid(symbol, timeframe="1h", limit=max(120, CHAN_LEN_HIGH + CHAN_ATR_LEN + 10))
                if df1h is None or len(df1h) < (CHAN_LEN_HIGH + CHAN_ATR_LEN + 5):
                    time.sleep(EXIT_CHECK_EVERY_SEC)
                    continue

                # Candles y features 1h
                close1h = float(df1h['close'].iloc[-1])
                ema20_1h = float(df1h['EMA20'].iloc[-1]) if 'EMA20' in df1h and pd.notna(df1h['EMA20'].iloc[-1]) else None
                vwap1h = float(df1h['VWAP'].iloc[-1]) if 'VWAP' in df1h and pd.notna(df1h['VWAP'].iloc[-1]) else None
                adx1h = float(df1h['ADX'].iloc[-1]) if 'ADX' in df1h and pd.notna(df1h['ADX'].iloc[-1]) else None

                # K efectivo
                k_eff = base_k

                # — RVOL grace: ampliar k temporalmente
                if RVOL_SPIKE_GRACE:
                    try:
                        df15 = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                        if df15 is not None and len(df15) > 20:
                            rv_mean = df15['volume'].rolling(10).mean().iloc[-1]
                            r3 = []
                            for i in range(3):
                                rv = df15['volume'].iloc[-1 - i] / rv_mean if rv_mean and rv_mean > RVOL_MEAN_MIN else None
                                if rv is not None: r3.append(float(rv))
                            if r3 and max(r3) >= RVOL_SPIKE_THRESHOLD:
                                rv_grace_until = time.time() + RVOL_K_BONUS_MINUTES * 60
                        if rv_grace_until and time.time() <= rv_grace_until:
                            k_eff += RVOL_SPIKE_K_BONUS
                    except Exception:
                        pass

                # — Soft flags por 15m (no venden; aprietan k)
                try:
                    df15 = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                    if df15 is not None and len(df15) > 30:
                        rsi15 = float(df15['RSI'].iloc[-1]) if 'RSI' in df15 else None
                        macd15 = ta.macd(df15['close'], fast=12, slow=26, signal=9)
                        macdh15 = float(macd15['MACDh_12_26_9'].iloc[-1]) if macd15 is not None else None
                        if rsi15 is not None and macdh15 is not None:
                            if (rsi15 >= NBUS_RSI15_OB and macdh15 < 0):  # sobrecompra + debilidad
                                soft_tighten_until = time.time() + 2 * 15 * 60
                    if soft_tighten_until and time.time() <= soft_tighten_until:
                        k_eff = max(1.2, k_eff - SOFT_TIGHTEN_K)
                except Exception:
                    pass

                # — Tiers por múltiplos de R (dejar correr, sin TP duro, sólo apretar k)
                gain = price - purchase_price
                if gain >= BE_R_MULT * initial_R:
                    # sube stop a break-even + fees
                    be_stop = purchase_price * (1.0 + required_edge_pct()/100.0)
                    stop_price = max(stop_price, be_stop)
                if gain >= TIER2_R_MULT * initial_R:
                    k_eff = max(1.2, k_eff - TIER2_K_TIGHTEN)

                # ===== Chandelier y ratchet del stop =====
                c_stop = chandelier_stop_long(df1h, atr_len=CHAN_ATR_LEN, hh_len=CHAN_LEN_HIGH, k=k_eff)
                if c_stop is not None:
                    # nunca bajar el stop (ratchet)
                    stop_price = max(stop_price, c_stop, initial_stop)

                # ===== Salida por estructura (Donchian + EMA20) =====
                structural_exit = False
                if held_long_enough:
                    d_low = donchian_lower(df1h, length=DONCHIAN_LEN_EXIT)
                    if d_low is not None and ema20_1h is not None:
                        if (close1h < ema20_1h) and (close1h < d_low):
                            structural_exit = True

                # ===== Time-stop por # velas (con prórroga si tape mejora) =====
                try:
                    n_bars = count_closed_bars_since(df1h, opened_ts)
                    if n_bars >= TIME_STOP_BARS_1H:
                        extend = False
                        # ADX slope reciente y close>VWAP
                        if n_bars < (TIME_STOP_BARS_1H + TIME_STOP_EXTEND_BARS):
                            try:
                                adx_series = ta.adx(df1h['high'], df1h['low'], df1h['close'], length=14)['ADX_14']
                                x = np.arange(6)
                                adx_slope = linregress(x, adx_series.iloc[-6:]).slope
                            except Exception:
                                adx_slope = None
                            improving = True
                            if TAPE_IMPROVING_ADX_SLOPE_MIN is not None:
                                improving = improving and (adx_slope is not None and adx_slope > TAPE_IMPROVING_ADX_SLOPE_MIN)
                            if TAPE_IMPROVING_VWAP_REQ:
                                improving = improving and (vwap1h is not None and close1h > vwap1h)
                            extend = bool(improving)
                        if not extend:
                            logger.info(f"[time-stop] {symbol} exit after {n_bars} bars 1h (no improvement)")
                            sell_symbol(symbol, amount, trade_id, source="trail")
                            return
                except Exception as e:
                    logger.debug(f"time-stop bars check error {symbol}: {e}")

                # ===== Disparador por precio (Chandelier/ratchet) =====
                # Política de fees: si el stop cae en ganancia pequeña, exige cubrir fees; si es pérdida, siempre vende.
                if price <= stop_price:
                    if price >= purchase_price:
                        edge_needed_pct = max(required_edge_pct()*MIN_GAIN_OVER_FEES_MULT/EDGE_SAFETY_MULT, required_edge_pct())
                        gain_pct = (price / purchase_price - 1.0) * 100.0
                        if gain_pct < edge_needed_pct:
                            # Demasiado micro-take; mantener si no hay estructura rota
                            if structural_exit:
                                sell_symbol(symbol, amount, trade_id, source="trail")
                                return
                            # de lo contrario, seguimos observando (el stop podrá subir en próximas velas)
                        else:
                            # OK fees cubiertos
                            sell_symbol(symbol, amount, trade_id, source="trail")
                            return
                    else:
                        # Stop de pérdida: salir siempre
                        sell_symbol(symbol, amount, trade_id, source="trail")
                        return

                # ===== Rebound Guard (si hubiese un trigger "duro" por estructura) =====
                if structural_exit and REBOUND_GUARD_ENABLED:
                    try:
                        if rebound_pending_until is None:
                            rebound_pending_until = time.time() + REBOUND_WAIT_BARS_15M * 15 * 60
                            last_exit_trigger_reason = 'structure'
                            logger.info(f"[rebound-guard] {symbol} structural exit pending for {REBOUND_WAIT_BARS_15M}x15m")

                        df15 = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                        cancel_exit = False
                        if df15 is not None and len(df15) > 21:
                            rsi_series = ta.rsi(df15['close'], length=14)
                            rsi_now = float(rsi_series.iloc[-1]) if rsi_series is not None else None
                            rsi_prev = float(rsi_series.iloc[-3]) if rsi_series is not None else None
                            ema20_15 = _ema(df15['close'], 20)
                            ema_ok = False
                            if REBOUND_EMA_RECLAIM and ema20_15 is not None:
                                ema_ok = float(df15['close'].iloc[-1]) > float(ema20_15.iloc[-1])
                            rsi_bounce = False
                            if rsi_now is not None and rsi_prev is not None:
                                rsi_bounce = (rsi_now - rsi_prev) >= REBOUND_MIN_RSI_BOUNCE

                            div_ok = False
                            if REBOUND_USE_5M_DIVERGENCE:
                                df5 = fetch_and_prepare_df(symbol, "5m", limit=120)
                                if df5 is not None and len(df5) > 30:
                                    div_ok = bullish_divergence(df5)

                            if rsi_bounce or ema_ok or div_ok:
                                cancel_exit = True

                        if cancel_exit:
                            # Si hay rebote, no vendemos y solo apretamos un poco más el k
                            base_k = max(1.2, base_k - 0.2)
                            rebound_pending_until = None
                            logger.info(f"[rebound-guard] {symbol} cancel structural exit — tighten k to {base_k:.2f}")
                        elif time.time() >= rebound_pending_until:
                            sell_symbol(symbol, amount, trade_id, source="trail")
                            return
                    except Exception as e:
                        logger.debug(f"rebound-guard(structure) error {symbol}: {e}")
                        sell_symbol(symbol, amount, trade_id, source="trail")
                        return

                time.sleep(EXIT_CHECK_EVERY_SEC)

        except Exception as e:
            logger.error(f"Trailing error {symbol}: {e}")
            try:
                if not trade_is_closed():
                    sell_symbol(symbol, amount, trade_id, source="trail")
            except Exception:
                pass
            return

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

def analyze_and_learn(trade_id: str, sell_price: float = None, learn_enabled: bool = True):
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

        adjustments = {}
        win = pnl_usdt > 0.0
        base = symbol.split('/')[0]
        lp = get_learn_params(base)
        rsi_min = lp["rsi_min"]; rsi_max = lp["rsi_max"]
        adx_min = lp["adx_min"]; rvol_base = lp["rvol_base"]

        TRADE_ANALYTICS_COUNT[base] = TRADE_ANALYTICS_COUNT.get(base, 0) + 1
        do_learn = learn_enabled and (TRADE_ANALYTICS_COUNT[base] >= LEARN_MIN_SAMPLES)

        if do_learn:
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

            def _clip_delta(cur, new, max_abs=LEARN_DAILY_CLIP):
                return clamp(new, cur - max_abs, cur + max_abs)

            rsi_min = _clip_delta(lp["rsi_min"], rsi_min)
            rsi_max = _clip_delta(lp["rsi_max"], rsi_max)
            adx_min = _clip_delta(lp["adx_min"], adx_min)
            rvol_base = _clip_delta(lp["rvol_base"], rvol_base)

            # mean reversion a defaults
            rsi_min = (1 - MEAN_REV_WEIGHT)*rsi_min + MEAN_REV_WEIGHT*RSI_MIN_DEFAULT
            rsi_max = (1 - MEAN_REV_WEIGHT)*rsi_max + MEAN_REV_WEIGHT*RSI_MAX_DEFAULT
            adx_min = (1 - MEAN_REV_WEIGHT)*adx_min + MEAN_REV_WEIGHT*ADX_MIN_DEFAULT
            rvol_base = (1 - MEAN_REV_WEIGHT)*rvol_base + MEAN_REV_WEIGHT*RVOL_BASE_DEFAULT

            if adjustments:
                set_learn_params(base, rsi_min, rsi_max, adx_min, rvol_base)
                LAST_PARAM_UPDATE[base] = date.today().isoformat()

        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO trade_analysis (trade_id, ts, symbol, pnl_usdt, pnl_pct, duration_sec,
                                                   entry_snapshot_json, exit_snapshot_json, adjustments_json, learn_enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, datetime.now(timezone.utc).isoformat(), symbol,
              float(analysis["pnl_usdt"]), float(analysis["pnl_pct"]), int(analysis["duration_sec"]),
              json.dumps(analysis["entry"], ensure_ascii=False),
              json.dumps(analysis["exit"], ensure_ascii=False),
              json.dumps(adjustments, ensure_ascii=False),
              1 if learn_enabled else 0))
        conn.commit(); conn.close()

        outcome = "WIN ✅" if win else "LOSS ❌"
        lines = [
            f"📊 *Trade Analysis* {outcome}  {symbol}",
            f"Trade: `{trade_id}`",
            f"PnL: `{analysis['pnl_usdt']:.4f} USDT`  (`{analysis['pnl_pct']:.2f}%`)  Duration: `{analysis['duration_sec']}s`",
            f"*Entry* → RSI `{analysis['entry']['rsi']}`  ADX `{analysis['entry']['adx']}`  RVOL `{analysis['entry']['rvol']}`",
            f"*Exit*  → RSI `{analysis['exit']['rsi']}`  ADX `{analysis['exit']['adx']}`  RVOL `{analysis['exit']['rvol']}`",
        ]
        if adjustments:
            adj_txt = ", ".join([f"{k}:{v['old']}→{v['new']}" for k,v in adjustments.items()])
            lines.append(f"Learned nudges → {adj_txt}")
        if not learn_enabled:
            lines.append("_Note: startup/maintenance exit — learning disabled_")
        send_telegram_message("\n".join(lines))

        try:
            if pnl_usdt <= 0:
                LAST_LOSS_INFO[symbol] = {"ts": datetime.now(timezone.utc).isoformat(), "pnl_usdt": float(pnl_usdt)}
        except Exception:
            pass

    except Exception as e:
        logger.error(f"analyze_and_learn error {trade_id}: {e}", exc_info=True)

# =========================
# Sizing (volatility-aware)
# =========================
# =========================
# Sizing (volatility-aware)
# =========================
def size_position(price: float, usdt_balance: float, confidence: int, symbol: str = None) -> float:
    if not price or price <= 0: 
        return 0.0

    # base fraction por confianza
    conf_mult = (confidence - 50) / 50.0
    conf_mult = max(0.0, min(conf_mult, 0.84))
    base_frac = RISK_FRACTION * (0.6 + 0.4 * conf_mult)

    # penalización temporal post-pérdida
    if symbol and symbol in LAST_LOSS_INFO:
        try:
            last_dt = datetime.fromisoformat(LAST_LOSS_INFO[symbol]["ts"].replace("Z",""))
            if (datetime.now(timezone.utc) - last_dt).total_seconds() < POST_LOSS_SIZING_WINDOW_SEC:
                base_frac *= POST_LOSS_SIZING_FACTOR
        except Exception:
            pass

    # slippage shave
    slip_pct = estimate_slippage_pct(symbol, notional=MIN_NOTIONAL) if symbol else 0.05
    shave = clamp(1.0 - (slip_pct/100.0)*0.5, 0.9, 1.0)

    # ajuste por volatilidad (1/sqrt(ATR%)), cap en [0.6, 1.4]
    vol_adj = 1.0
    try:
        prof = classify_symbol(symbol) if symbol else None
        atrp = (prof or {}).get("atrp_30d")
        if atrp:
            vol_adj = clamp(1.0 / max(0.8, math.sqrt(atrp)), 0.6, 1.4)
    except Exception:
        pass

    # === Soft sizing bias by regime + 1h RSI (no blocking) ===
    try:
        sym_cls = classify_symbol(symbol) if symbol else None
        klass_sz = (sym_cls or {}).get("class", "medium")
        rsi1h = None
        snap1h = quick_tf_snapshot(symbol, '1h', limit=80) if symbol else {}
        rsi1h = float(snap1h.get('RSI')) if snap1h and snap1h.get('RSI') is not None else None

        size_bias = 1.0
        if klass_sz == "unstable" and rsi1h is not None:
            if rsi1h < 55:      # mushy → trim
                size_bias *= 0.70
            elif rsi1h < 60:    # borderline → mild trim
                size_bias *= 0.85
        elif klass_sz == "stable" and rsi1h is not None:
            if rsi1h < 52:
                size_bias *= 0.85

        size_bias = clamp(size_bias, 0.6, 1.1)
    except Exception:
        size_bias = 1.0

    # final budget
    budget = usdt_balance * base_frac * shave * vol_adj * size_bias
    amount = max(MIN_NOTIONAL / price, budget / price)
    return amount

# =========================
# Portfolio util & crash halt
# =========================
def portfolio_utilization():
    try:
        bal = exchange.fetch_balance() or {}
        total = float(bal.get('total', {}).get('USDT', 0.0) or 0.0)
        free  = float(bal.get('free',  {}).get('USDT', 0.0) or 0.0)
        used  = max(total - free, 0.0)
        return used / (total + 1e-9), total, free, used
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def crash_halt():
    try:
        df = fetch_and_prepare_df("BTC/USDT", "5m", limit=max(6, CRASH_HALT_WINDOW_MIN//5 + 2))
        if df is None or len(df) < 4: return False
        recent = df['close'].iloc[-1]
        past = df['close'].iloc[-(CRASH_HALT_WINDOW_MIN//5 + 1)]
        drop = (recent/past - 1.0) * 100.0
        return drop <= -CRASH_HALT_DROP_PCT
    except Exception:
        return False

def strong_symbol_momentum_15m(symbol: str) -> bool:
    if not CRASH_HALT_ENABLE_OVERRIDE:
        return False
    try:
        ohlcv = fetch_ohlcv_with_retry(symbol, timeframe="15m", limit=120)
        if not ohlcv:
            return False
        df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms'); df.set_index('ts', inplace=True)
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce').ffill().bfill()

        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is None or adx_df.empty:
            return False
        adx15 = float(adx_df['ADX_14'].iloc[-1])
        ema20_15 = float(ta.ema(df['close'], length=20).iloc[-1])
        last15 = float(df['close'].iloc[-1])

        cond_adx = adx15 > CRASH_HALT_OVERRIDE_MIN_ADX15
        cond_ema = (last15 > ema20_15) if CRASH_HALT_OVERRIDE_REQUIRE_EMA20 else True
        ok = bool(cond_adx and cond_ema)
        if ok:
            logger.info(f"[crash-override] {symbol}: ADX15={adx15:.1f} EMA20OK={cond_ema} — allowing buy despite BTC dip.")
        else:
            logger.info(f"[crash-override] {symbol}: blocked (ADX15={adx15:.1f}, last>EMA20={last15>ema20_15})")
        return ok
    except Exception as e:
        logger.debug(f"strong_symbol_momentum_15m error {symbol}: {e}")
        return False

# =========================
# Trading step + report
# =========================
buy_lock = threading.Lock()

def has_open_position(symbol: str) -> bool:
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transactions WHERE symbol=? AND side='buy' AND status='open'", (symbol,))
    cnt = cur.fetchone()[0]; conn.close()
    return cnt > 0

def get_open_trades_count() -> int:
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transactions WHERE side='buy' AND status='open'")
    cnt = cur.fetchone()[0]; conn.close()
    return int(cnt)

def trade_once_with_report(symbol: str):
    report = {"symbol": symbol, "action": "hold", "confidence": 50, "score": 0.0, "note": "", "executed": False}
    try:
        if has_open_position(symbol):
            report["note"] = "already holding"; logger.info(f"{symbol}: {report['note']}"); log_decision(symbol, 0.0, "hold", False); return report

        util, total, free, used = portfolio_utilization()
        if util > PORTFOLIO_MAX_UTIL:
            report["note"] = f"portfolio util {util:.2f} > max"; logger.info(report["note"]); log_decision(symbol, 0.0, "hold", False); return report

        open_trades = get_open_trades_count()
        if open_trades >= MAX_OPEN_TRADES:
            report["note"] = "max open trades"; logger.info("Max open trades."); log_decision(symbol, 0.0, "hold", False); return report

        balances = exchange.fetch_balance()
        usdt = float(balances.get('free', {}).get('USDT', 0.0))
        if usdt - RESERVE_USDT < MIN_NOTIONAL:
            report["note"] = f"insufficient USDT after reserve ({usdt:.2f})"; logger.info(report["note"]); log_decision(symbol, 0.0, "hold", False); return report

        # Decision
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

            amount = size_position(price, usdt, conf, symbol)
            trade_val = amount * price
            if trade_val < MIN_NOTIONAL and trade_val < (usdt - RESERVE_USDT):
                report["note"] = f"trade value {trade_val:.2f} < MIN_NOTIONAL"; log_decision(symbol, score, "hold", False); return report

            order = execute_order_buy(symbol, amount, {
                'adx': float(row['ADX']) if pd.notna(row['ADX']) else None,
                'rsi': float(row['RSI']) if pd.notna(row['RSI']) else None,
                'rvol': (float(row['RVOL10']) if pd.notna(row['RVOL10']) else None),
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
    logger.info("Startup cleanup: closing non-USDT balances and syncing DB states...")
    try:
        try: 
            exchange.load_markets()
        except Exception as e: 
            logger.warning(f"load_markets warning: {e}")

        balances = exchange.fetch_balance() or {} 
        free = balances.get("free", {}) or {}
        non_usdt = {a: amt for a, amt in free.items() if a != "USDT" and amt and amt > 0}

        conn = sqlite3.connect(DB_NAME); cur = conn.cursor()

        for asset, amt in non_usdt.items():
            symbol = f"{asset}/USDT"
            if symbol not in getattr(exchange, "markets", {}):
                logger.info(f"Skip {asset}: market {symbol} not found")
                continue

            px = fetch_price(symbol)
            if not px or px <= 0:
                logger.info(f"Skip {symbol}: no price on cleanup")
                continue

            free_bal = float(free.get(asset, 0) or 0)
            desired = min(free_bal, amt)
            sell_amt, reason = _prepare_sell_amount(symbol, desired, px)
            if sell_amt <= 0:
                logger.info(f"Skip {symbol}: {reason} (free={free_bal}, desired={desired})")
                continue

            try:
                order = exchange.create_market_sell_order(symbol, sell_amt)
            except ccxt.InsufficientFunds:
                sell_amt_retry, reason2 = _prepare_sell_amount(symbol, sell_amt * 0.995, px)
                if sell_amt_retry > 0:
                    try:
                        order = exchange.create_market_sell_order(symbol, sell_amt_retry)
                        sell_amt = sell_amt_retry
                    except ccxt.InsufficientFunds:
                        logger.info(f"Skip {symbol}: Insufficient after retry (amt={sell_amt_retry})")
                        continue
                else:
                    logger.info(f"Skip {symbol}: {reason2} after insufficient funds retry")
                    continue

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
            """, (symbol, float(sell_price), float(sell_amt), ts, trade_id))

            if row:
                cur.execute("UPDATE transactions SET status='closed' WHERE trade_id=? AND side='buy'", (trade_id,))
            conn.commit()

            send_telegram_message(f"🔒 *Startup Cleanup*: Vendido `{symbol}` cantidad `{sell_amt}` a `{sell_price}`")

            exit_feats = build_rich_features(symbol)
            save_trade_features(trade_id, symbol, 'exit', exit_feats)

            analyze_and_learn(trade_id, sell_price, learn_enabled=False)
            LAST_SELL_INFO[symbol] = {"ts": ts, "price": float(sell_price), "source": "startup"}

            logger.info(f"Startup sold {symbol}: amt={sell_amt}, px={sell_price}, trade_id={trade_id}")

        conn.close()
        reconcile_orphan_open_buys()
        logger.info("Startup cleanup done.")
    except Exception as e:
        logger.error(f"startup_cleanup fatal: {e}", exc_info=True)


# =========================
# Main loop (r4.2)
# =========================
if __name__ == "__main__":
    logger.info("Starting hybrid trader (r4.2)…")
    try:
        try: exchange.load_markets()
        except Exception as e: logger.warning(f"load_markets warning: {e}")

        startup_cleanup()

        while True:
            check_log_rotation()

            crash_active = crash_halt()
            cycle_decisions = []

            for sym in SELECTED_CRYPTOS:
                if crash_active:
                    if sym == "BTC/USDT":
                        logger.warning("⚠️ Crash halt active — skipping BTC/USDT.")
                        continue
                    if not strong_symbol_momentum_15m(sym):
                        logger.warning(f"⚠️ Crash halt active — skipping {sym} (no momentum override).")
                        continue

                rep = trade_once_with_report(sym)
                cycle_decisions.append(rep)
                time.sleep(1)

            controller_autotune()
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

           
