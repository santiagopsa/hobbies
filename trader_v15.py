

# =========================
# trader_v15.py
# =========================
from __future__ import annotations
from pathlib import Path
import os, time, json, threading, sqlite3, logging, logging.handlers, random, math
from datetime import datetime, timezone, date, timedelta  # >>> PATCH: add timedelta
import ccxt, numpy as np, pandas as pd, pandas_ta as ta, requests
from dotenv import load_dotenv
from scipy.stats import linregress
from typing import Tuple

# =========================
# Config & initialization
# =========================

load_dotenv()

# >>> STRATEGY PROFILE (ANCHOR SP0)
# Selector de estrategia
STRATEGY_PROFILES = {"USDT_MOMENTUM", "BTC_PARK", "AUTO_HYBRID"}

STRATEGY_PROFILE = (os.getenv("STRATEGY_PROFILE", "USDT_MOMENTUM") or "USDT_MOMENTUM").strip().upper()
if STRATEGY_PROFILE not in STRATEGY_PROFILES:
    STRATEGY_PROFILE = "USDT_MOMENTUM"  # fallback seguro


# Parking config (solo aplica a BTC_PARK)
PARK_PCT = float(os.getenv("PARK_PCT", "0.90"))   # % del USDT libre (después de RESERVE_USDT) a parquear
PARK_STATE = {"active": False}                    # estado en memoria (simple)

# >>> BTC PARK FLOOR (ANCHOR PKF)
PARK_PROTECT_ENABLED = True
PARK_TRAIL_PCT = float(os.getenv("PARK_TRAIL_PCT", "0.03"))     # 3% desde el high
PARK_REENTRY_PAD_BPS = int(os.getenv("PARK_REENTRY_PAD_BPS", "30"))  # +0.30% para re-entrada
PARK_MIN_HOLD_MIN = int(os.getenv("PARK_MIN_HOLD_MIN", "15"))   # evitar churn


# Target de parking cuando el entorno está “calm/safe”
AUTO_PARK_PCT = float(os.getenv("AUTO_PARK_PCT", "0.60"))      # 60% del capital libre (tras reserva) hacia BTC
AUTO_PARK_DEADBAND_BPS = int(os.getenv("AUTO_PARK_DEADBAND_BPS", "50"))  # banda muerta ±0.50% para evitar micro-churn

# Anti-chop extra (puedes sobreescribir en .env)
HYBRID_GLOBAL_COOLDOWN_SEC = int(os.getenv("HYBRID_GLOBAL_COOLDOWN_SEC", "180"))  # espera mínima entre cambios
HYBRID_MIN_REVERSAL_BPS    = int(os.getenv("HYBRID_MIN_REVERSAL_BPS", "20"))     # 0.20% de movimiento para revertir
HYBRID_MAX_ACTIONS_5M      = int(os.getenv("HYBRID_MAX_ACTIONS_5M", "4"))        # tope de acciones en 5 minutos

# Condiciones “calm/safe”
AUTO_SAFE_FGI = int(os.getenv("AUTO_SAFE_FGI", "60"))           # FGI ≥ 60
AUTO_SAFE_ATRP_BTC = float(os.getenv("AUTO_SAFE_ATRP_BTC", "1.2"))  # ATR% 30d de BTC ≤ 1.2%

DB_NAME = "trading_real.db"
LOG_PATH = os.path.expanduser("~/hobbies/trading.log")

TOP_COINS = ['BTC','ETH','BNB','SOL','XRP','DOGE','TON','ADA','TRX','AVAX']
SELECTED_CRYPTOS = [f"{c}/USDT" for c in TOP_COINS]

# >>> REGIME/BREADTH CONFIG (ANCHOR RBG)
REGIME_LEADERS = ["BTC/USDT", "ETH/USDT"]
BREADTH_COINS  = [f"{c}/USDT" for c in TOP_COINS]
# >>> REGIME/BREADTH TUNING (ANCHOR RBG)
BREADTH_MIN_COUNT = 4          # antes 6
BREADTH_RSI_MIN_1H = 55.0      # antes 60.0
BREADTH_REQUIRE_EMA20 = False  # antes True (lo hacemos "soft")

REGIME_LEADER_RSI_MIN = 52.0
REGIME_LEADER_ADX_MIN = 20.0
BREADTH_CACHE_TTL_SEC = 120  # seconds

# >>> STRONG TREND TRAILING (let winners breathe)
STRONG_TREND_ADX_1H = 30.0
STRONG_TREND_ADX_4H = 30.0
STRONG_TREND_K_STABLE   = 3.0   # floor for k when trend is strong
STRONG_TREND_K_MEDIUM   = 3.0
STRONG_TREND_K_UNSTABLE = 2.5


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
# >>> PATCH START: gate & cooldowns
SCORE_GATE_START = 5.0
SCORE_GATE_MAX = 6.0
SCORE_GATE_HARD_MIN = 4.5  # tougher floor to avoid low-quality buys

# re-entry/cooldowns
COOLDOWN_AFTER_COLLAPSE_MIN = 60   # minutes — after a volume-collapse exit
COOLDOWN_AFTER_SCRATCHES_MIN = 60  # minutes — after 2 scratches in <2h
# >>> PATCH END

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
TIME_STOP_HOURS = 3
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
TIME_STOP_BARS_1H = 3     # si en 3 velas 1h no progresa (o no cumple mejora), salir
TIME_STOP_EXTEND_BARS = 6  # prórroga si "tape mejorando" (ADX↑ y close>VWAP)
# >>> CHAN PATCH CONFIG END


# Momentum penalties/blocks
RSI_OVERBOUGHT_4H = 78.0
PENALTY_4H_RSI = 0.7
PENALTY_15M_WEAK = 0.5

# >>> SCRATCH FILTER CONFIG (ANCHOR SF1)
ADX_SCRATCH_MIN = 18.0           # require ADX >= 18 OR positive slope if 15 <= ADX < 18
ADX_SCRATCH_SLOPE_BARS = 6       # slope lookback (1h ADX)
ADX_SCRATCH_SLOPE_MIN = 0.0      # must be > 0 when ADX < 18


# Cooldowns / re-entry
COOLDOWN_MIN_AFTER_LOSS = 10
POST_LOSS_SIZING_WINDOW_SEC = 2 * 3600
POST_LOSS_CONF_CAP = 78
POST_LOSS_SIZING_FACTOR = 0.70
REENTRY_BLOCK_MIN = 15  # unify with top-level patch
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

# >>> PREFLIGHT KNOBS (ANCHOR PF0)
PREFLIGHT_ADX_HARD_MIN = 12.0   # antes 15
PREFLIGHT_RVOL1H_MIN   = 0.95   # antes 1.00


# No-buy-under-sell thresholds
NBUS_RSI15_OB = 75.0
NBUS_MACDH15_NEG = -0.002

# >>> PATCH A1: BEAR/THIN-TAPE & COLLAPSE KNOBS
# Bearish-context gate (4h) & Thin-tape RVOL floors
RSI4H_HARD_MIN = 45.0       # HARD block if 4h RSI <45 and trending down
RSI4H_SOFT_MIN = 50.0       # SOFT block if 45≤RSI<50 and trending down unless RVOL strong
RSI4H_SLOPE_BARS = 6        # slope lookback (4h bars)

RVOL_1H_MIN  = 0.80         # base floors to avoid thin tape at entry
RVOL_15M_MIN = 0.60
RVOL_FLOOR_STABLE_BONUS   = -0.10  # stable: allow 0.10 lower
RVOL_FLOOR_UNSTABLE_BONUS = +0.10  # unstable: require 0.10 higher

# Volume-collapse exit (post-entry)
RVOL_COLLAPSE_EXIT_ENABLED = True
RVOL_COLLAPSE_EXIT = 0.50   # if 15m RVOL < 0.50 after min hold → exit

# >>> FGI CONFIG (ANCHOR FGI0)
FGI_API_URL = "https://api.alternative.me/fng/?limit=2&format=json"
FGI_CACHE_TTL_SEC = 600   # 10 min
FGI_EXTREME_FEAR = 20
FGI_FEAR = 35
FGI_GREED = 60
FGI_EXTREME_GREED = 75

_FGI_CACHE = {"ts": 0, "value": None, "prev": None}

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
# >>> PATCH START: scratch log
SCRATCH_LOG = {}  # symbol -> [epoch_seconds] of recent scratch exits
# >>> PATCH END
# >>> PARK GUARD STATE (ANCHOR PKF_STATE)
PARK_GUARD = {"anchor": None, "high": None, "last_change": 0}


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


def fetch_fear_greed_index() -> dict:
    """
    Retorna {"value": int|None, "prev": int|None, "ts": epoch}
    Con caché para evitar rate-limit/timeouts.
    """
    now = time.time()
    if now - _FGI_CACHE["ts"] < FGI_CACHE_TTL_SEC and _FGI_CACHE["value"] is not None:
        return {"value": _FGI_CACHE["value"], "prev": _FGI_CACHE["prev"], "ts": _FGI_CACHE["ts"]}
    try:
        r = requests.get(FGI_API_URL, timeout=6)
        j = r.json() if r is not None else {}
        data = (j or {}).get("data", [])
        cur = int(data[0]["value"]) if data and "value" in data[0] else None
        prev = int(data[1]["value"]) if len(data) > 1 and "value" in data[1] else None
        _FGI_CACHE.update({"ts": now, "value": cur, "prev": prev})
        return {"value": cur, "prev": prev, "ts": now}
    except Exception:
        # Mantén último valor de caché si existe
        return {"value": _FGI_CACHE.get("value"), "prev": _FGI_CACHE.get("prev"), "ts": _FGI_CACHE.get("ts", 0)}

def is_market_bullish() -> tuple[bool, int|None, str]:
    """
    Usa F&G para determinar si el contexto global es propicio.
    True si >= FGI_GREED o si (>=50 y mejorando vs prev).
    """
    f = fetch_fear_greed_index()
    v, p = f.get("value"), f.get("prev")
    if v is None:
        return (True, None, "no-fgi")  # falla segura
    if v >= FGI_GREED:
        return (True, v, "greed")
    if v >= 50 and (p is not None) and v > p:
        return (True, v, "improving")
    return (False, v, "fear")

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

def _prepare_buy_amount(symbol, desired_amt, px):
    """
    Ajusta el amount de compra a los filtros de Binance.
    - Pisa al stepSize
    - Verifica minQty
    - Verifica minNotional (cantidad * precio)
    Retorna (amount_ok, reason_or_None)
    """
    step, min_qty, min_notional = _bn_filters(symbol)
    amt = max(0.0, float(desired_amt or 0.0))
    if step:
        amt = _floor_to_step(amt, step)
    if min_qty and amt < min_qty:
        return 0.0, "amount < minQty"
    if min_notional and px and (amt * px) < min_notional:
        return 0.0, "notional < minNotional"
    return amt, None

def _prepare_sell_amount(symbol, desired_amt, px):
    """
    Ajusta el amount de venta a los filtros de Binance.
    - Pisa al stepSize
    - Verifica minQty
    - Verifica minNotional (cantidad * precio)
    Retorna (amount_ok, reason_or_None)
    """
    step, min_qty, min_notional = _bn_filters(symbol)
    amt = max(0.0, float(desired_amt or 0.0))
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

# >>> CALM/SAFE DETECTOR (ANCHOR AH1)
def is_calm_safe_env() -> tuple[bool, str]:
    """
    true si el entorno global es alcista y estable:
      - market_regime_ok() == True
      - is_market_bullish() con FGI >= AUTO_SAFE_FGI
      - BTC estable: ATR% 30d ≤ AUTO_SAFE_ATRP_BTC
      - Tendencia sana 1h: EMA20 > EMA50 y close > EMA20
    """
    try:
        reg_ok = market_regime_ok()
    except Exception:
        reg_ok = True  # fail-open

    mk_bull, fgi_v, tag = is_market_bullish()
    if not (reg_ok and mk_bull and (fgi_v is not None and fgi_v >= AUTO_SAFE_FGI)):
        return (False, f"reg={reg_ok},fgi={fgi_v},{tag}")

    try:
        prof_btc = classify_symbol("BTC/USDT") or {}
        atrp = float(prof_btc.get("atrp_30d") or 0.0)
        if atrp <= 0 or atrp > AUTO_SAFE_ATRP_BTC:
            return (False, f"atrp_btc={atrp:.2f}>limit")
    except Exception:
        return (False, "atrp_err")

    try:
        d1 = fetch_and_prepare_data_hybrid("BTC/USDT", timeframe="1h", limit=120)
        if d1 is None or len(d1) < 30:
            return (False, "no-1h")
        ema20 = float(d1["EMA20"].iloc[-1])
        ema50 = float(d1["EMA50"].iloc[-1])
        close = float(d1["close"].iloc[-1])
        if not (ema20 > ema50 and close > ema20):
            return (False, "ema-structure-weak")
    except Exception:
        return (False, "ema_err")

    return (True, f"fgi≥{AUTO_SAFE_FGI},atrp≤{AUTO_SAFE_ATRP_BTC}")

# >>> STRATEGY HELPERS (ANCHOR SP1)
def strategy_profile() -> str:
    return STRATEGY_PROFILE

def _free_balances() -> tuple[float, float]:
    """USDT free, BTC free"""
    try:
        b = exchange.fetch_balance() or {}
        usdt = float(b.get('free', {}).get('USDT', 0.0) or 0.0)
        btc  = float(b.get('free', {}).get('BTC', 0.0)  or 0.0)
        return usdt, btc
    except Exception:
        return 0.0, 0.0

# >>> STRATEGY HELPERS (ANCHOR AH2) — reemplaza la función existente si ya la tenías
def park_to_btc_if_needed() -> bool:
    """
    Parquear a BTC (en realidad salir a USDT en BTC/USDT) cuando se rompe un piso dinámico,
    y “des-parquear” (reentrar) sólo cuando:
      1) ha pasado un tiempo mínimo en parking (PARK_MIN_HOLD_MIN), y
      2) el precio recupera al menos hasta (high * (1 - PARK_REENTRY_PAD_BPS/10000)).

    Requiere vars globales:
      - PARK_PROTECT_ENABLED (bool)
      - PARK_GUARD (dict con claves: anchor, high, floor, last_change, reason, parked)
      - PARK_TRAIL_BPS (int, bps para trailing del piso)
      - PARK_REENTRY_PAD_BPS (int, bps para “reclaim”)
      - PARK_MIN_HOLD_MIN (int, minutos de bloqueo antes de re-entrar)
      - MIN_NOTIONAL (float, notional mínimo del exchange)
      - MIN_RESERVE_USDT (float, USDT a reservar sin usar)
    """
    import time
    from datetime import datetime, timezone

    global PARK_GUARD

    symbol = "BTC/USDT"

    # --- helpers locales ---------------------------
    def _init_guard():
        """Asegura estructura y valores por defecto del guard."""
        nonlocal PARK_GUARD
        if not isinstance(PARK_GUARD, dict):
            PARK_GUARD = {}
        PARK_GUARD.setdefault("anchor", None)
        PARK_GUARD.setdefault("high", None)
        PARK_GUARD.setdefault("floor", None)
        PARK_GUARD.setdefault("last_change", None)  # epoch seconds
        PARK_GUARD.setdefault("reason", None)
        PARK_GUARD.setdefault("parked", False)

    def _floor_guard_active(btc_val_usdt: float) -> bool:
        # Evita ruido si la posición es demasiado pequeña
        return PARK_PROTECT_ENABLED and btc_val_usdt >= max(MIN_NOTIONAL, 25)

    def _update_anchor_and_floor(last_px: float):
        """Actualiza high/anchor y recalcula floor con trailing bps."""
        g = PARK_GUARD
        if g["high"] is None or last_px > g["high"]:
            g["high"] = last_px
            g["anchor"] = last_px
            g["last_change"] = time.time()
        # trailing floor desde el HIGH
        g["floor"] = (g["high"] or last_px) * (1.0 - (PARK_TRAIL_BPS * 0.0001))

    def _reentry_trigger_price(current_px: float) -> float:
        """Precio de 'reclaim' para poder re-entrar (pad por debajo del high)."""
        g = PARK_GUARD
        base = g["high"] or current_px
        return base * (1.0 - (PARK_REENTRY_PAD_BPS * 0.0001))

    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # --- inicio -----------------------------------
    _init_guard()

    # Balances y precio actual
    usdt_free, btc_free = _free_balances()
    btc_px = float(fetch_price(symbol) or 0.0)
    if btc_px <= 0:
        return False

    btc_val = btc_free * btc_px  # valor de la posición BTC en USDT
    now_epoch = time.time()

    # 1) Si tengo BTC (posición) y el guard está activo, mantener trailing y vender si rompe el piso
    if _floor_guard_active(btc_val):
        _update_anchor_and_floor(btc_px)

        # Rompe el piso => vender a USDT (parquear)
        floor = PARK_GUARD.get("floor") or 0.0
        if btc_px <= floor and btc_free > 0.0:
            # Respetar reserva de USDT: no vender el BTC equivalente a MIN_RESERVE_USDT
            reserve_btc = max(0.0, MIN_RESERVE_USDT / btc_px)
            qty_to_sell = max(0.0, btc_free - reserve_btc)

            if qty_to_sell * btc_px >= MIN_NOTIONAL:
                trade_id = f"PARK-FLOOR-{_now_iso()}"

                # Enviar snapshot de features de salida
                try:
                    features = build_rich_features(symbol)
                    extra = f"Price: {btc_px:,.2f}  Amount: {qty_to_sell:.8f}"
                    send_feature_bundle_telegram(
                        side="exit",
                        symbol=symbol,
                        trade_id=trade_id,
                        features=features,
                        extra_lines=extra,
                    )
                except Exception as e:
                    logger.warning(f"[park] no se pudo enviar features de salida: {e}")

                # Vender
                try:
                    sell_symbol(symbol, qty_to_sell, trade_id=trade_id, source="park-floor")
                    PARK_GUARD["last_change"] = now_epoch
                    PARK_GUARD["reason"] = "floor_break"
                    PARK_GUARD["parked"] = True
                    return True
                except Exception as e:
                    logger.error(f"[park] error al vender BTC por floor-break: {e}")
                    # aunque falle, no devolvemos True
            # si la qty no alcanza el mínimo, no hacemos nada
        # si no rompe el piso, no hacemos nada

    # 2) Si estoy aparcado (sin BTC) y tengo USDT suficientes, evaluar des-parking (re-entrada)
    #    Condiciones: cooldown y reclaim del precio
    if PARK_PROTECT_ENABLED and btc_free <= 0.0 and usdt_free >= (MIN_NOTIONAL + 1.0):
        last_change = PARK_GUARD.get("last_change")
        hold_secs = (now_epoch - last_change) if last_change else None
        min_hold_secs = float(PARK_MIN_HOLD_MIN) * 60.0

        # (a) cooldown: esperar al menos PARK_MIN_HOLD_MIN min tras el parking
        if hold_secs is not None and hold_secs < min_hold_secs:
            # Demasiado pronto para re-entrar
            return False

        # (b) reclaim: precio actual debe estar por encima del trigger de re-entrada
        reclaim_px = _reentry_trigger_price(btc_px)
        if btc_px >= reclaim_px:
            # Notional a usar: todo lo que supere la reserva
            buy_usdt = max(0.0, usdt_free - MIN_RESERVE_USDT)
            if buy_usdt >= MIN_NOTIONAL:
                qty_to_buy = round(buy_usdt / btc_px, 8)  # granularidad suficiente p/ BTC

                trade_id = f"PARK-UNWIND-{_now_iso()}"

                # Ejecutar la compra primero (para que el order fill dispare el mensaje "BUY")
                try:
                    execute_order_buy(
                        symbol,
                        qty_to_buy,
                        signals={"confidence": 60, "score": 0.0},
                    )
                except Exception as e:
                    logger.error(f"[unwind] error al recomprar BTC: {e}")
                    return False

                # Enviar snapshot de features de entrada
                try:
                    features = build_rich_features(symbol)
                    extra = f"Price: {btc_px:,.2f}  Amount: {qty_to_buy:.8f}"
                    send_feature_bundle_telegram(
                        side="entry",
                        symbol=symbol,
                        trade_id=trade_id,
                        features=features,
                        extra_lines=extra,
                    )
                except Exception as e:
                    logger.warning(f"[unwind] no se pudo enviar features de entrada: {e}")

                # Reset del guard
                PARK_GUARD["parked"] = False
                PARK_GUARD["anchor"] = btc_px
                PARK_GUARD["high"] = btc_px
                PARK_GUARD["floor"] = btc_px * (1.0 - (PARK_TRAIL_BPS * 0.0001))
                PARK_GUARD["last_change"] = now_epoch
                PARK_GUARD["reason"] = "reclaim"

                return True

    return False



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
    if df is None or df.empty: return 0
    try:
        ts = pd.Timestamp(ts_open)
        if ts.tzinfo is None: ts = ts.tz_localize('UTC')
        idx = df.index
        if idx.tz is None: idx = idx.tz_localize('UTC')
        return int((idx > ts).sum())
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
    """
    Construye features de un DataFrame OHLCV (indexado por timestamp).
    Nota: StochRSI y Fisher se guardan en 'extras' y se mezclan al final
    para evitar ser sobrescritos cuando se reasigna 'features'.
    """
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
        macd  = macd_df['MACD_12_26_9']  if macd_df is not None and not macd_df.empty else pd.Series(index=df.index, dtype=float)
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

        # ---- StochRSI + Fisher: guardar en extras y mezclar al final ----
        extras = {}
        try:
            srs = ta.stochrsi(df['close'], length=14)
            if srs is not None and not srs.empty:
                extras["stochrsi_k"] = float(srs.iloc[-1, 0])  # 'STOCHRSIk_14_14_3_3'
                extras["stochrsi_d"] = float(srs.iloc[-1, 1])  # 'STOCHRSId_14_14_3_3'
        except Exception:
            pass

        try:
            fdf = ta.fisher(df['high'], df['low'], length=9)
            if fdf is not None and not fdf.empty:
                # pandas_ta típicamente: 'FISHERT_9' y 'FISHERTs_9'
                extras["fisher_t"]  = float(fdf.iloc[-1, 0])
                extras["fisher_ts"] = float(fdf.iloc[-1, 1])
        except Exception:
            pass
        # -----------------------------------------------------------------

        roc7 = ta.roc(df['close'], length=7)
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])

        # RVOL10
        rv_mean = df['volume'].rolling(10).mean()
        rv_last = rv_mean.iloc[-1]
        rv_ok = (pd.notna(rv_last)) and (rv_last > RVOL_MEAN_MIN)
        df['RVOL10'] = (df['volume'] / rv_mean) if rv_ok else np.nan

        # Slopes de 10 barras
        price_slope10 = vol_slope10 = np.nan
        if len(df) >= 10:
            x = np.arange(10)
            price_slope10 = linregress(x, df['close'].iloc[-10:]).slope
            vol_slope10   = linregress(x, df['volume'].iloc[-10:]).slope

        last = df.iloc[-1]

        def sf(x):
            try:
                if pd.isna(x):
                    return None
                return float(x)
            except Exception:
                return None

        # Posición dentro de las bandas de Bollinger (0..1 aprox)
        bb_pos = None
        try:
            if (not pd.isna(bbu.iloc[-1]) and not pd.isna(bbl.iloc[-1]) and (bbu.iloc[-1] - bbl.iloc[-1]) != 0):
                bb_pos = (last['close'] - bbl.iloc[-1]) / (bbu.iloc[-1] - bbl.iloc[-1])
        except Exception:
            bb_pos = None

        # ATR como %
        atr_pct = None
        try:
            atr_pct = float(atr.iloc[-1] / last['close'] * 100.0)
        except Exception:
            pass

        # RVOL último (solo si la media es válida)
        try:
            rv_last_val = df['RVOL10'].iloc[-1]
            rvv = float(rv_last_val) if pd.notna(rv_last_val) else None
        except Exception:
            rvv = None

        # Ensamblar features (mezclando 'extras' al final)
        features = {
            "label": label,
            "last_close": sf(last['close']),
            "atr": sf(atr.iloc[-1]), "atr_pct": atr_pct,
            "rsi": sf(rsi.iloc[-1]), "adx": sf(adx.iloc[-1]),
            "ema20": sf(ema20.iloc[-1]), "ema50": sf(ema50.iloc[-1]),
            "ema200": sf(ema200.iloc[-1]) if len(ema200) else None,
            "macd": sf(macd.iloc[-1]), "macd_signal": sf(macds.iloc[-1]),
            "macd_hist": sf(macdh.iloc[-1]),
            "stoch_k": sf(stoch_k.iloc[-1]), "stoch_d": sf(stoch_d.iloc[-1]),
            "bb_lower": sf(bbl.iloc[-1]), "bb_mid": sf(bbm.iloc[-1]),
            "bb_upper": sf(bbu.iloc[-1]), "bb_pos": bb_pos,
            "obv": sf(obv.iloc[-1]), "roc7": sf(roc7.iloc[-1]), "vwap": sf(vwap.iloc[-1]),
            "rvol10": rvv,
            "price_slope10": float(price_slope10) if not pd.isna(price_slope10) else None,
            "vol_slope10": float(vol_slope10) if not pd.isna(vol_slope10) else None,
            **extras,  # ← aquí se mezclan StochRSI y Fisher calculados arriba
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
    try:
        willr = ta.willr(df['high'], df['low'], df['close'], length=14)
        snap['WILLR'] = float(willr.iloc[-1]) if willr is not None else None
    except Exception:
        pass
    try:
        srs = ta.stochrsi(df['close'], length=14)
        snap['STOCHRSIk'] = float(srs.iloc[-1, 0]) if srs is not None and not srs.empty else None
    except Exception:
        pass
    return snap

# === Breadth cache / regime helpers ===
_BREADTH_CACHE = {"ts": 0, "ok": True, "count": 0, "leaders_ok": True}

def _tf1h_closed(symbol: str):
    snap = quick_tf_snapshot(symbol, '1h', limit=80) or {}
    return {
        "rsi": snap.get("RSI"),
        "adx": snap.get("ADX"),
        "ema20": snap.get("EMA20"),
        "last": snap.get("last"),
    }

def compute_breadth() -> tuple:
    now = time.time()
    if now - _BREADTH_CACHE["ts"] < BREADTH_CACHE_TTL_SEC:
        return _BREADTH_CACHE["leaders_ok"], _BREADTH_CACHE["count"]

    leaders_ok = False
    try:
        flags = []
        for lead in REGIME_LEADERS:
            s = _tf1h_closed(lead)
            if s["rsi"] is None or s["adx"] is None:
                continue
            flags.append(s["rsi"] >= REGIME_LEADER_RSI_MIN and s["adx"] >= REGIME_LEADER_ADX_MIN)
        leaders_ok = any(flags)
    except Exception:
        leaders_ok = False

    breadth = 0
    try:
        for sym in BREADTH_COINS:
            s = _tf1h_closed(sym)
            if s["rsi"] is None or s["last"] is None or (BREADTH_REQUIRE_EMA20 and s["ema20"] is None):
                continue
            ema_ok = True if not BREADTH_REQUIRE_EMA20 else (s["last"] >= s["ema20"])
            if (s["rsi"] >= BREADTH_RSI_MIN_1H) and ema_ok:
                breadth += 1
    except Exception:
        pass

    _BREADTH_CACHE.update({
        "ts": now,
        "ok": leaders_ok and breadth >= BREADTH_MIN_COUNT,
        "count": breadth,
        "leaders_ok": leaders_ok
    })
    return leaders_ok, breadth


def market_regime_ok() -> bool:
    leaders_ok, breadth = compute_breadth()
    return bool(leaders_ok and breadth >= BREADTH_MIN_COUNT)


# >>> PATCH A2: 4h RSI slope & trend flags + RVOL floors
def rsi_slope(df: pd.DataFrame, length: int = 14, bars: int = 6) -> float:
    """Return linear slope of RSI over last `bars` candles (positive = rising)."""
    try:
        r = ta.rsi(df['close'], length=length)
        seg = r.dropna().iloc[-bars:]
        if len(seg) < bars:
            return 0.0
        x = np.arange(len(seg))
        return float(linregress(x, seg.values).slope)
    except Exception:
        return 0.0

# >>> SCRATCH FILTER HELPERS (ANCHOR SF2)
def series_slope_last_n(series: pd.Series, bars: int = 6) -> float:
    """Linear slope over the last `bars` points (positive => rising)."""
    try:
        s = pd.Series(series).dropna()
        seg = s.iloc[-bars:]
        if len(seg) < bars:
            return 0.0
        x = np.arange(len(seg))
        return float(linregress(x, seg.values).slope)
    except Exception:
        return 0.0

def preflight_buy_guard(symbol: str) -> Tuple[bool, str]:
    """
    Final check right BEFORE placing a buy:
      1) HARD block if 1h ADX < PREFLIGHT_ADX_HARD_MIN.
      2) If PREFLIGHT_ADX_HARD_MIN ≤ ADX < ADX_SCRATCH_MIN, require:
         - 15m MACDh > 0 (short-term impulse)
         - close_1h > EMA20_1h (minimal structure)
      3) HARD block if 1h RVOL10 < PREFLIGHT_RVOL1H_MIN.

    Extra adjustments:
      - Overextension guard: if RSI1h > 75, require 4h ADX rising AND MACDh15 accelerating > 0.
      - Pullback timing on 15m: RSI15 in [55..65] AND MACDh15 accelerating > 0.
      - Pair filter (SOL/DOGE): require alignment of 1h/4h and pullback_ok.
    """
    # --- Load 1h
    df1h = fetch_and_prepare_data_hybrid(symbol, timeframe="1h", limit=80)
    if df1h is None or len(df1h) < 30:
        return False, "no 1h data"

    try:
        adx_1h     = float(df1h['ADX'].iloc[-1]) if pd.notna(df1h['ADX'].iloc[-1]) else None
        rvol1h_raw = df1h['RVOL10'].iloc[-1]
        rvol1h     = float(rvol1h_raw) if pd.notna(rvol1h_raw) else None
        ema20_1h   = float(df1h['EMA20'].iloc[-1]) if pd.notna(df1h['EMA20'].iloc[-1]) else None
        close_1h   = float(df1h['close'].iloc[-1])
    except Exception:
        return False, "bad 1h fields"

    # (1) ADX hard floor
    if adx_1h is None or adx_1h < PREFLIGHT_ADX_HARD_MIN:
        return False, f"1h ADX {None if adx_1h is None else f'{adx_1h:.1f}'} < {PREFLIGHT_ADX_HARD_MIN:.0f}"

    # Preload 15m for scratch zone and later checks
    df15 = None
    # (2) Scratch zone: ADX in [PREFLIGHT_ADX_HARD_MIN .. ADX_SCRATCH_MIN)
    if adx_1h < ADX_SCRATCH_MIN:
        df15 = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=80)
        if df15 is None or len(df15) < 35:
            return False, "no 15m data for scratch check"

        # 15m MACDh
        try:
            macd15_df = ta.macd(df15['close'], fast=12, slow=26, signal=9)
            macdh15 = float(macd15_df['MACDh_12_26_9'].iloc[-1]) if macd15_df is not None and not macd15_df.empty else None
        except Exception:
            macdh15 = None

        # Need impulse on 15m and structure on 1h
        if (macdh15 is None or macdh15 <= 0.0) or (close_1h is None or ema20_1h is None or not (close_1h > ema20_1h)):
            return False, f"scratch needs MACDh15>0 and close1h>EMA20 (ADX {adx_1h:.1f} < {ADX_SCRATCH_MIN:.0f})"

    # (3) RVOL 1h floor
    if rvol1h is None or rvol1h < PREFLIGHT_RVOL1H_MIN:
        return False, f"1h RVOL {None if rvol1h is None else f'{rvol1h:.2f}'} < {PREFLIGHT_RVOL1H_MIN:.2f}"

    # ===== Extra adjustments / timing filters =====
    # Ensure df15 is available
    if df15 is None:
        df15 = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=80)

    # Load 4h
    df4h = fetch_and_prepare_data_hybrid(symbol, timeframe="4h", limit=120)
    if df4h is None or df15 is None or len(df4h) < 30 or len(df15) < 35:
        return False, "no 4h/15m data for ajustes"

    # RSI 1h / 15m
    try:
        rsi1h = float(ta.rsi(df1h['close'], length=14).iloc[-1])
    except Exception:
        rsi1h = None
    try:
        rsi15 = float(ta.rsi(df15['close'], length=14).iloc[-1])
    except Exception:
        rsi15 = None

    # 4h ADX and rising flag
    try:
        adx4h_df = ta.adx(df4h['high'], df4h['low'], df4h['close'], length=14)
        adx4h = float(adx4h_df['ADX_14'].iloc[-1]) if adx4h_df is not None and not adx4h_df.empty else None
        adx4h_prev = float(adx4h_df['ADX_14'].iloc[-2]) if adx4h_df is not None and len(adx4h_df) >= 2 else None
        adx4h_rising = (adx4h is not None and adx4h_prev is not None and adx4h > adx4h_prev)
    except Exception:
        adx4h = None
        adx4h_rising = False

    # 15m MACD acceleration
    try:
        macd15_df = ta.macd(df15['close'], fast=12, slow=26, signal=9)
        macdh15_now = float(macd15_df['MACDh_12_26_9'].iloc[-1]) if macd15_df is not None and not macd15_df.empty else None
        macdh15_prev = float(macd15_df['MACDh_12_26_9'].iloc[-2]) if macd15_df is not None and len(macd15_df) >= 2 else None
        macdh15_accel = (macdh15_now is not None and macdh15_prev is not None and macdh15_now > macdh15_prev and macdh15_now > 0.0)
    except Exception:
        macdh15_now = None
        macdh15_accel = False

    # (A) Overextension guard on 1h
    if rsi1h is not None and rsi1h > 75.0:
        if not (adx4h_rising and macdh15_accel):
            return False, f"overextended_1h_rsi>{rsi1h:.1f} without confirmation (ADX4h rising & MACDh15 accelerating)"

    # (B) Pullback timing on 15m
    pullback_ok = (rsi15 is not None and 55.0 <= rsi15 <= 65.0) and macdh15_accel
    if not pullback_ok:
        return False, "pullback_15m_not_clear (want RSI15 in 55–65 and MACDh15 accelerating>0)"

    # (C) Pair filter for SOL/DOGE (need added alignment)
    base = symbol.split('/')[0]
    if base in {'SOL', 'DOGE'}:
        try:
            rsi4h = float(ta.rsi(df4h['close'], length=14).iloc[-1])
        except Exception:
            rsi4h = None
        ok_align = (rsi1h is not None and rsi1h >= 63.0) and (rsi4h is not None and rsi4h >= 55.0) and adx4h_rising and pullback_ok
        if not ok_align:
            return False, "pair_filter: SOL/DOGE requires 1h/4h alignment + clear 15m pullback"

    return True, "ok"





def rsi_trend_flags(df: pd.DataFrame, length: int = 14, fast: int = 2, slow: int = 3) -> tuple:
    """
    Returns (is_up, is_down) using a tiny MA crossover on 4h RSI.
    - is_up  => fast MA > slow MA and last RSI > prev RSI
    - is_down=> fast MA < slow MA and last RSI < prev RSI
    """
    try:
        r = ta.rsi(df['close'], length=length).dropna()
        if len(r) < max(fast, slow) + 2:
            return (False, False)
        r_fast = ta.sma(r, length=fast)
        r_slow = ta.sma(r, length=slow)
        is_up = bool(r_fast.iloc[-1] > r_slow.iloc[-1] and r.iloc[-1] > r.iloc[-2])
        is_down = bool(r_fast.iloc[-1] < r_slow.iloc[-1] and r.iloc[-1] < r.iloc[-2])
        return (is_up, is_down)
    except Exception:
        return (False, False)

def rvol_floors_by_regime(klass: str) -> tuple:
    """Return (rvol_1h_min, rvol_15m_min) adjusted by volatility class."""
    d1 = d15 = 0.0
    if klass == "stable":
        d1 = d15 = RVOL_FLOOR_STABLE_BONUS
    elif klass == "unstable":
        d1 = d15 = RVOL_FLOOR_UNSTABLE_BONUS
    return max(0.0, RVOL_1H_MIN + d1), max(0.0, RVOL_15M_MIN + d15)


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
    ng = clamp(float(new_gate), SCORE_GATE_HARD_MIN, SCORE_GATE_MAX)
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

# >>> DOWNTREND (ANCHOR DT0)
def is_downtrend(symbol: str) -> tuple[bool, float, str]:
    """
    Señal local por símbolo (4h/1h):
    - down si RSI4h < 50 con slope<=0, o estructura EMA20<EMA50 y close<EMA20 en 1h.
    - severidad en [0..1] (0.3 leve, 0.6 media, 0.9 fuerte).
    """
    try:
        df4 = fetch_and_prepare_data_hybrid(symbol, timeframe="4h", limit=120)
        df1 = fetch_and_prepare_data_hybrid(symbol, timeframe="1h", limit=120)
        if df4 is None or len(df4) < 30 or df1 is None or len(df1) < 30:
            return (False, 0.0, "no-data")

        rsi4 = float(df4["RSI"].iloc[-1]) if not pd.isna(df4["RSI"].iloc[-1]) else None
        slope4 = series_slope_last_n(df4["RSI"], bars=6) if "RSI" in df4 else 0.0

        ema20_1h = float(df1["EMA20"].iloc[-1]) if not pd.isna(df1["EMA20"].iloc[-1]) else None
        ema50_1h = float(df1["EMA50"].iloc[-1]) if not pd.isna(df1["EMA50"].iloc[-1]) else None
        close_1h = float(df1["close"].iloc[-1])

        ema_down = (ema20_1h is not None and ema50_1h is not None and ema20_1h < ema50_1h and close_1h < ema20_1h)
        rsi_down = (rsi4 is not None and rsi4 < 50.0 and slope4 <= 0.0)

        if not (ema_down or rsi_down):
            return (False, 0.0, "neutral")

        # severidad
        sev = 0.0
        if ema_down: sev += 0.5
        if rsi4 is not None:
            if rsi4 < 42: sev += 0.3
            elif rsi4 < 47: sev += 0.15
        if slope4 < -0.2: sev += 0.2

        rsi4_note = f"{rsi4:.1f}" if rsi4 is not None else "—"
        return (True, float(clamp(sev, 0.2, 0.95)), f"rsi4={rsi4_note},slope={slope4:.2f},ema_down={ema_down}")

    except Exception:
        return (False, 0.0, "error")

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
        if (rsi15 is not None and rsi15 >= NBUS_RSI15_OB) and (macdh15 is not None and macdh15 <= NBUS_MACDH15_NEG):
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
    # local copies to avoid mutating globals
    reentry_pad_local = REENTRY_ABOVE_LAST_EXIT_PAD
    reentry_block_min_local = REENTRY_BLOCK_MIN

    if klass == "stable":
        RSI_MIN_K, RSI_MAX_K = max(RSI_MIN, 52), min(RSI_MAX, 68)
        ADX_MIN_K = max(18, ADX_MIN - 2)
        RVOL_BASE_K = max(0.8, RVOL_BASE * 0.8)
        SCORE_GATE_OFFSET = -0.2
        reentry_pad_local = 0.0015
    elif klass == "unstable":
        # soften vs. your previous +0.3 gate bump and tighter RSI
        RSI_MIN_K, RSI_MAX_K = max(48, RSI_MIN - 2), min(66, RSI_MAX)   # was 65
        ADX_MIN_K = max(23, ADX_MIN)                                     # was 25
        RVOL_BASE_K = max(1.05, RVOL_BASE)                               # was >=1.1
        SCORE_GATE_OFFSET = +0.1                                         # was +0.3
        reentry_pad_local = 0.0030                                       # was 0.0035

    blocks = []
    level = "NONE"  # NONE | SOFT | HARD

    # >>> F&G + Downtrend modulation (ANCHOR FGI1)
    try:
        mk_bull, fgi_v, fgi_tag = is_market_bullish()
        down, down_sev, down_note = is_downtrend(symbol)

        # Global (F&G): aflojar en greed, apretar en fear
        if fgi_v is not None:
            if fgi_v >= FGI_EXTREME_GREED:
                ADX_MIN_K = max(ADX_MIN_K - 2, 15)
                RVOL_BASE_K = max(0.9, RVOL_BASE_K * 0.9)
                SCORE_GATE_OFFSET += -0.25
            elif fgi_v >= FGI_GREED:
                ADX_MIN_K = max(ADX_MIN_K - 1, 16)
                RVOL_BASE_K = max(1.0, RVOL_BASE_K * 0.95)
                SCORE_GATE_OFFSET += -0.15
            elif fgi_v <= FGI_EXTREME_FEAR:
                ADX_MIN_K = max(ADX_MIN_K + 3, ADX_MIN_K)  # endurece
                RVOL_BASE_K = max(1.1, RVOL_BASE_K * 1.10)
                SCORE_GATE_OFFSET += +0.35
            elif fgi_v <= FGI_FEAR:
                ADX_MIN_K = max(ADX_MIN_K + 1, ADX_MIN_K)
                RVOL_BASE_K = max(1.05, RVOL_BASE_K * 1.05)
                SCORE_GATE_OFFSET += +0.20

        # Local (downtrend por símbolo): apretar gates y re-entry
        if down:
            bump = 1.0 + 0.35 * down_sev
            ADX_MIN_K = max(ADX_MIN_K, 20 + 6*down_sev)
            RVOL_BASE_K = max(RVOL_BASE_K, 1.1 * bump)
            SCORE_GATE_OFFSET += +0.20 + 0.20*down_sev
            reentry_pad_local = max(reentry_pad_local, 0.003 + 0.004*down_sev)
            reentry_block_min_local = max(reentry_block_min_local, int(20 + 20*down_sev))
        else:
            # En subida local: permite un poco más de entradas
            SCORE_GATE_OFFSET += -0.05

        note_parts = []
        if fgi_v is not None: note_parts.append(f"FGI={fgi_v}({fgi_tag})")
        if down: note_parts.append(f"downtrend({down_sev:.2f})")
        if note_parts:
            blocks.append("mod:" + ",".join(note_parts))
    except Exception as e:
        blocks.append(f"fgi/down-mod error: {e}")

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

    # ===== market regime & breadth gate (NEW) =====
    if not market_regime_ok():
        blocks.append(
            f"regime off: BTC/ETH not trending or breadth < {BREADTH_MIN_COUNT} "
            f"(RSI≥{BREADTH_RSI_MIN_1H:.0f}{' & >EMA20' if BREADTH_REQUIRE_EMA20 else ''})"
        )

        level = "HARD"

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

    try:
        adx_slope_1h = series_slope_last_n(df['ADX'], ADX_SCRATCH_SLOPE_BARS)
    except Exception:
        adx_slope_1h = 0.0

    # >>> NEW: ADX floor + gray zone (replaces old scratch block + <15 floor)
    # Hard block if ADX < 17.5. Gray zone 17.5–22 allowed only if slope>0 and 4h ADX≥25.
    tf4h_for_adx = quick_tf_snapshot(symbol, '4h', limit=120)
    adx4h_for_gate = tf4h_for_adx.get('ADX')

    if adx is not None and adx < 17.5:
        blocks.append(f"HARD block: 1h ADX {adx:.1f} < 17.5")
        level = "HARD"
    elif adx is not None and 17.5 <= adx < 22.0:
        if adx_slope_1h <= 0.0 or (adx4h_for_gate is None or adx4h_for_gate < 25.0):
            blocks.append(f"HARD block: ADX {adx:.1f} in gray zone needs slope>0 and 4h ADX≥25 (slope={adx_slope_1h:.4f}, 4h={adx4h_for_gate})")
            level = "HARD"

    # Keep minimal volume sanity
    if (rvol_1h is not None) and (rvol_1h < 1.00):
        blocks.append(f"HARD block: 1h RVOL {rvol_1h:.2f} < 1.00")
        level = "HARD"

    # ===== fast snapshots (needed for anchors 2,8) =====
    tf15 = quick_tf_snapshot(symbol, '15m', limit=120) or {}
    tf4h = quick_tf_snapshot(symbol, '4h',  limit=120) or {}
    rv15 = tf15.get('RVOL10')

    # >>> PATCH A3: Bearish-context (4h) & Thin-tape RVOL floors
    rvol1_floor, rvol15_floor = rvol_floors_by_regime(klass)
    rsi4h = tf4h.get('RSI')
    macdh15 = tf15.get('MACDh')


    # >>> PATCH START: 15m RVOL filter
    rv15_floor_req = max(0.8, rvol15_floor)
    if (rv15 is None or rv15 < rv15_floor_req) and not ((rvol_1h or 0) >= 1.3 and (macdh15 or 0) > 0):
        blocks.append(f"SOFT block: 15m RVOL weak (<{rv15_floor_req:.2f}) without 1h RVOL≥1.3 & MACDh15>0")
        if level != "HARD": level = "SOFT"
    # >>> PATCH END


    # Compute 4h RSI slope and MA-based trend flags
    df4h_full = fetch_and_prepare_df(symbol, "4h", limit=300)
    rsi4h_slope = rsi_slope(df4h_full, length=14, bars=RSI4H_SLOPE_BARS) if df4h_full is not None else 0.0
    rsi4h_up, rsi4h_down = rsi_trend_flags(df4h_full, length=14, fast=2, slow=3) if df4h_full is not None else (False, False)

    # — Bearish HARD block: 4h RSI deeply <45 and trending down (slope<=0 or MA flag down)
    if (rsi4h is not None) and (rsi4h < RSI4H_HARD_MIN) and (rsi4h_slope <= 0 or rsi4h_down):
        blocks.append(f"HARD block: bearish 4h (RSI={rsi4h:.1f}, slope={rsi4h_slope:.4f}, down={rsi4h_down})")
        level = "HARD"

    # — Bearish SOFT block: 45≤RSI<50 and trending down → need stronger RVOL on BOTH TFs
    elif (rsi4h is not None) and (RSI4H_HARD_MIN <= rsi4h < RSI4H_SOFT_MIN) and (rsi4h_slope <= 0 or rsi4h_down):
        need_1h = max(rvol1_floor + 0.3, 1.1)
        need_15 = max(rvol15_floor + 0.3, 0.9)
        if not ((rvol_1h or 0) >= need_1h and (rv15 or 0) >= need_15):
            blocks.append(f"SOFT block: 4h drifting down; RVOL weak (need 1h≥{need_1h:.2f}, 15m≥{need_15:.2f})")
            if level != "HARD": level = "SOFT"

    # >>> PATCH START: 4h drifting down soft guard
    # Extra 4h drifting-down soft block for 45<=RSI4h<47 unless strong proof
    if (rsi4h is not None) and (RSI4H_HARD_MIN <= rsi4h < 47.0) and (rsi4h_slope <= 0 or rsi4h_down):
        try:
            last = float(row['close'])
            ema20_1h = float(row['EMA20']) if pd.notna(row['EMA20']) else None
            ema50_1h = float(row['EMA50']) if pd.notna(row['EMA50']) else None
        except Exception:
            ema20_1h = ema50_1h = None
        proof = ((rvol_1h or 0) >= 1.2) and ((macdh15 or 0) > 0) and (ema20_1h is None or last >= ema20_1h) and (ema50_1h is None or last >= ema50_1h)
        if not proof:
            blocks.append("SOFT block: 4h drifting down (<47) — need RVOL1h≥1.2, MACDh15>0 & EMA20/50 reclaim")
            if level != "HARD": level = "SOFT"
    # >>> PATCH END


    # — Thin-tape SOFT block: BOTH 1h and 15m RVOL under floors
    if ((rvol_1h is None or rvol_1h < rvol1_floor) and (rv15 is None or rv15 < rvol15_floor)):
        blocks.append(f"SOFT block: thin tape (1h RVOL {rvol_1h}, 15m RVOL {rv15}; floors {rvol1_floor:.2f}/{rvol15_floor:.2f})")
        if level != "HARD": level = "SOFT"

    # — Starvation escape hatch: only lift SOFT if 15m is genuinely waking up
    fifteen_waking = (rv15 is not None and rv15 >= max(0.9, rvol15_floor)) and ((macdh15 or 0) > 0)
    if level == "SOFT" and fifteen_waking:
        blocks.append("soft override: 15m waking (MACDh>0 & RVOL ok)")

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

    # >>> PATCH START: overextension guard
    # Wait for pullback/retest to EMA20 15m (+0.4%) unless retest already happened with MACDh15>0
    rsi1h_now = float(row['RSI']) if pd.notna(row['RSI']) else None
    rsi15_now = tf15.get('RSI'); rsi4h_now = tf4h.get('RSI')
    ema20_15 = tf15.get('EMA20'); last15 = tf15.get('last')
    overext = (rsi1h_now is not None and rsi1h_now >= 75.0) and (rsi15_now is not None and rsi15_now >= 74.0)
    if overext and last15 is not None and ema20_15 is not None and (last15 > ema20_15 * 1.004):
        try:
            macdh_15_val = tf15.get('MACDh')
            retest_ok = (last15 <= ema20_15 * 1.001) and (macdh_15_val is not None and macdh_15_val > 0)
        except Exception:
            retest_ok = False
        if not retest_ok:
            blocks.append("HARD block: overextension (1h≥75, 15m≥74, >EMA20_15+0.4%) — wait pullback/retest")
            level = "HARD"
    # >>> PATCH END


    # ===== NBUS rework (ANCHOR-2): relax ADX and OB gate =====
    try:
        sell_cond, reason = is_selling_condition_now(symbol)
        if sell_cond:
            blocks.append(f"NBUS: {reason}")
            level = "HARD"
    except Exception as e:
        logger.debug(f"NBUS helper error {symbol}: {e}")

    # ===== Prepare df15_full for multiple checks =====
    df15_full = None
    try:
        df15_full = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=120)
    except Exception:
        df15_full = None

    # ===== post-loss cooldown (ANCHOR-3: softer & more permissive) =====
    try:
        info = LAST_LOSS_INFO.get(symbol)
        if info and info.get("ts"):
            now = datetime.now(timezone.utc)
            last_dt = datetime.fromisoformat(info["ts"].replace("Z",""))
            within_cooldown = last_dt and (now - last_dt).total_seconds() < COOLDOWN_MIN_AFTER_LOSS * 60
            if within_cooldown:
                # easier override: RVOL15 > 1.5 or ADX15 > 30 and MACDh15 > 0
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
                    macdh15p = float(macd15['MACDh_12_26_9'].iloc[-1]) if macd15 is not None else None
                    post_loss_override = (
                        (rvol15_full is not None and rvol15_full > 1.5) or
                        ((adx15 or 0) > 30 and (macdh15p or 0) > 0)
                    )
                if not post_loss_override:
                    blocks.append("post-loss cooldown")
                    level = "HARD"
    except Exception:
        pass

    # ===== Breakout + retest on 15m (compute early; influences re-entry) =====
    breakout = False
    retest_ok = False


    try:
        if df15_full is None:
            df15_full = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=120)
        if df15_full is not None and len(df15_full) > 21:
            don_hi = df15_full['high'].rolling(20).max().shift(1)
            bo_lvl = float(don_hi.iloc[-2]) if pd.notna(don_hi.iloc[-2]) else None
            bo_prev_close = float(df15_full['close'].iloc[-2])
            bo_now_low = float(df15_full['low'].iloc[-1]); bo_now_close = float(df15_full['close'].iloc[-1])
            breakout = (bo_lvl and bo_prev_close > bo_lvl and (rv15 or 0) >= (rvol15_floor + 0.2))
            retest_ok = (bo_lvl and bo_now_low <= bo_lvl * 1.002 and bo_now_close >= bo_lvl)
            # >>> PATCH START: ADX<22 gate (moved to after flags are computed)
            if adx < 22.0 and not (breakout and retest_ok):
                blocks.append("HARD block: ADX<22 requires breakout+retest")
                level = "HARD"
            # >>> PATCH END

            if breakout and not retest_ok:
                reentry_block_min_local = max(reentry_block_min_local, 20)  # cooldown corto 20min solo para leading
                reentry_pad_local = max(reentry_pad_local, 0.004)          # exige alejarse más
    except Exception:
        pass

    # ===== re-entry block (ANCHOR-4: shorter) =====
    try:
        info = LAST_SELL_INFO.get(symbol)
        if info and info.get("ts"):
            last_dt = datetime.fromisoformat(info["ts"].replace("Z",""))
            if last_dt and (datetime.now(timezone.utc) - last_dt).total_seconds() < (reentry_block_min_local * 60):
                blocks.append(f"re-entry block ({reentry_block_min_local}m)")
                level = "HARD"
    except Exception:
        pass

    # >>> PATCH START: re-entry throttles
    # Post-collapse cooldown (if last sell was 'collapse' and window still active)
    try:
        info_pc = LAST_SELL_INFO.get(symbol)
        if info_pc and info_pc.get("post_collapse_until"):
            if datetime.now(timezone.utc) < datetime.fromisoformat(info_pc["post_collapse_until"].replace("Z","")):
                blocks.append("post-collapse cooldown active")
                level = "HARD"
    except Exception:
        pass

    # Two scratches in <2h → block for 60m
    try:
        now_ts = time.time()
        recent_scratches = [t for t in SCRATCH_LOG.get(symbol, []) if now_ts - t <= 7200]
        if len(recent_scratches) >= 2:
            blocks.append("two scratches in <2h — re-entry blocked (60m)")
            level = "HARD"
    except Exception:
        pass
    # >>> PATCH END


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
            exit_ok = last >= float(le["price"]) * (1.0 + reentry_pad_local)

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

        # >>> PATCH START: MACDh15 penalty (moved here, after score/notes init)
    if (macdh15 is not None) and (macdh15 < 0):
        score -= 0.7; notes.append("MACDh15<0 penalty (-0.7)")
        if adx < 25.0:
            blocks.append("HARD block: ADX<25 with MACDh15<0")
            level = "HARD"
    # >>> PATCH END

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
    # ...ya tienes tf15/tf4h snapshots
    stochk15 = tf15.get('STOCHRSIk')
    willr15 = tf15.get('WILLR')
    # Fisher 1h desde df de 1h ya cargado
    f_t = None
    try:
        fdf = ta.fisher(df['high'], df['low'], length=9)
        if fdf is not None and not fdf.empty:
            f_t = float(fdf.iloc[-1, 0])
    except Exception:
        fdf = None

    # Scoring con guardas de volumen (defensa 1: cap leading_raw)
    leading_raw = 0.0

    # re-eval rvol_any (explicit)
    rvol_any_for_leading = None
    for _rv in (rvol_1h, rv15):
        if _rv is not None and np.isfinite(_rv):
            rvol_any_for_leading = _rv if (rvol_any_for_leading is None) else max(rvol_any_for_leading, _rv)
    rvol_any_for_leading = rvol_any_for_leading or 0  # fallback

    # StochRSI 15m (timing entrada; umbrales por klass + persistencia, defensa 2)
    stoch_persist_ok = False
    if stochk15 is not None:
        try:
            if df15_full is None:
                df15_full = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=120)
            if df15_full is not None and len(df15_full) > 20:
                srs = ta.stochrsi(df15_full['close'], length=14)
                k = float(srs.iloc[-1,0]); k_prev = float(srs.iloc[-2,0])
                stoch_persist_ok = (k < 20 and k_prev < 20)  # 2 velas cerradas
        except Exception:
            stoch_persist_ok = True  # fallback no-block
    stoch_threshold = 18 if klass == "stable" else 20 if klass == "medium" else 22
    if (stochk15 is not None and stochk15 < stoch_threshold and stoch_persist_ok and
        rvol_any_for_leading >= max(RVOL_BASE_K, 1.0) and (tf15.get('MACDh') or 0) > 0):
        leading_raw += 1.0; notes.append("StochRSI15 oversold + vol")

    # Fisher 1h (cambio régimen; umbral por klass)
    fisher_threshold = 0 if klass == "stable" else 0 if klass == "medium" else 0.2
    if f_t is not None and f_t > fisher_threshold:
        leading_raw += 0.7; notes.append("Fisher1h bullish")
        # Extra si sube vs. previa
        try:
            if fdf is not None:
                f_prev = float(fdf.iloc[-2, 0])
                if f_t > f_prev:
                    leading_raw += 0.3
        except Exception:
            pass
        # Si estabas en SOFT por confirmaciones EMA/VWAP, permite override suave
        if level == "SOFT":
            notes.append("soft override (Fisher1h)")
            level = "NONE"

    leading_bonus = min(leading_raw, 1.2)  # cap duro (defensa 1)
    # (FIX) Don't add leading_bonus yet; apply once after MTF alignment below.

    # Defensa 3: Anti-chop BBWidth + ADX squeeze
    try:
        bb = ta.bbands(df['close'], length=20, std=2)
        width = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / df['close']
        last_w = float(width.iloc[-1])
        hist = width.dropna().iloc[-60:]
        bb_rank = float((hist < last_w).mean())  # 0..1
        if (adx or 0) < 20 and bb_rank < 0.30:
            blocks.append(f"chop/squeeze: BBwidth%rank={bb_rank:.2f}, ADX={adx:.1f}")
            level = "HARD"
    except Exception:
        pass

    # Defensa 4: Breakout + retest 15m
    breakout = False
    retest_ok = False
    try:
        if df15_full is None:
            df15_full = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=120)
        if df15_full is not None and len(df15_full) > 21:
            don_hi = df15_full['high'].rolling(20).max().shift(1)
            bo_lvl = float(don_hi.iloc[-2]) if pd.notna(don_hi.iloc[-2]) else None
            bo_prev_close = float(df15_full['close'].iloc[-2])
            bo_now_low = float(df15_full['low'].iloc[-1]); bo_now_close = float(df15_full['close'].iloc[-1])
            breakout = (bo_lvl and bo_prev_close > bo_lvl and (rv15 or 0) >= (rvol15_floor + 0.2))
            retest_ok = (bo_lvl and bo_now_low <= bo_lvl * 1.002 and bo_now_close >= bo_lvl)
            if breakout and retest_ok:
                score += 0.8; notes.append("breakout+retest 15m")
            elif breakout and not retest_ok:
                # Defensa 7: Cooldown breakout fallido (solo para leading)
                reentry_block_min_local = max(reentry_block_min_local, 20)
                reentry_pad_local = max(reentry_pad_local, 0.004)
    except Exception:
        pass

    # Defensa 5: Filtro velas con mecha (15m última vela)
    try:
        if df15_full is not None and len(df15_full) > 1:
            rng = df15_full['high'].iloc[-1] - df15_full['low'].iloc[-1]
            body = abs(df15_full['close'].iloc[-1] - df15_full['open'].iloc[-1])
            near_high = df15_full['close'].iloc[-1] >= df15_full['high'].iloc[-1] * 0.995
            if rng > 0 and (body/rng) < 0.35 and not near_high:
                blocks.append("candle con mecha (falso breakout)")
                if level != "HARD": level = "SOFT"
    except Exception:
        pass

    # Defensa 6: Alineación MTF light (aplica leading_bonus SOLO si alineado)
    if not (row['EMA20'] > row['EMA50'] and adx_slope_1h > 0):
        notes.append("MTF misaligned → sin bonus leading")
        # (FIX) do not add leading_bonus
    else:
        score += leading_bonus  # apply once when aligned

    # RSI band
    if RSI_MIN_K <= rsi <= RSI_MAX_K:
        score += 1.0; notes.append(f"RSI in band ({rsi:.1f})")
    if rsi >= (RSI_MIN_K + RSI_MAX_K)/2:
        score += 0.5

    if adx >= ADX_MIN_K:
        score += 1.0; notes.append(f"ADX≥{ADX_MIN_K:.0f} ({adx:.1f})")

    vol_slope = float(row.get('VOL_SLOPE10', 0.0) or 0.0)
    if vol_slope > 0: score += 0.5

    # Spread penalty
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

    # >>> PATCH START: guarded starvation override
    if level == "HARD":
        exec_allowed = False
    elif level == "SOFT":
        allow_starve = bool(rsi4h_up and (macdh15 or 0) > 0 and (rv15 or 0) >= 1.0)
        exec_allowed = relax_soft and allow_starve
        if exec_allowed: notes.append("soft override (starvation, guarded)")
    else:
        exec_allowed = True
    # >>> PATCH END


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
        info = {"ts": sell_ts, "price": float(sell_price), "source": source}
        if source == "collapse":
            until = datetime.now(timezone.utc) + timedelta(minutes=COOLDOWN_AFTER_COLLAPSE_MIN)
            info["post_collapse_until"] = until.isoformat()
        LAST_SELL_INFO[symbol] = info



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
    """
    Dynamic trailing based on 1h Chandelier + structure, with:
      - Initial ATR stop and R tracking
      - k adapted by volatility class; 'strong trend' mode lets winners breathe
      - RVOL spike grace (15m closed bars)
      - Soft-tighten on 15m (%R/RSI + MACDh), disabled in strong trend
      - BE at 1R; extra tighten ≥ 3R
      - Structure exits: 1h EMA20 loss + Donchian lower break
      - Guards: 15m EMA20 / swing-low; 15m volume collapse (closed bar)
      - Time-stop (3h with MACDh15<0 twice)
      - Rebound-guard to avoid selling right into bounces
    """
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

            # Base k from volatility class
            prof_local = classify_symbol(symbol) or {}
            klass_local = prof_local.get("class", "medium")
            if   klass_local == "unstable": base_k = CHAN_K_UNSTABLE
            elif klass_local == "stable":   base_k = CHAN_K_STABLE
            else:                           base_k = CHAN_K_MEDIUM

            opened_ts = datetime.now(timezone.utc)

            # Initial protective stop and R
            initial_stop = purchase_price - (INIT_STOP_ATR_MULT * atr_abs)
            if initial_stop >= purchase_price:
                initial_stop = purchase_price * 0.985
            initial_R = max(purchase_price - initial_stop, purchase_price * 0.003)

            # State
            stop_price = initial_stop
            rv_grace_until = None
            soft_tighten_until = None
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

                # Global min-hold except hard protective stop
                if not held_long_enough:
                    if price <= initial_stop:
                        sell_symbol(symbol, amount, trade_id, source="trail")
                        return
                    time.sleep(EXIT_CHECK_EVERY_SEC)
                    continue

                # Immediate guard on 15m EMA20 / recent swing-low
                try:
                    df15g = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                    if df15g is not None and len(df15g) >= 20:
                        ema20_15g = float(ta.ema(df15g['close'], length=20).iloc[-1])
                        swing_low_15 = float(df15g['low'].iloc[-5:].min())
                        if price < ema20_15g or price < swing_low_15:
                            sell_symbol(symbol, amount, trade_id, source="ema20_15m/swing_low")
                            return
                except Exception as e:
                    logger.warning(f"[exit][{symbol}] ema20_15m/swing check failed: {e}")

                # 1h context
                df1h = fetch_and_prepare_data_hybrid(symbol, timeframe="1h",
                                                     limit=max(120, CHAN_LEN_HIGH + CHAN_ATR_LEN + 10))
                if df1h is None or len(df1h) < (CHAN_LEN_HIGH + CHAN_ATR_LEN + 5):
                    time.sleep(EXIT_CHECK_EVERY_SEC)
                    continue

                close1h  = float(df1h['close'].iloc[-1])
                ema20_1h = float(df1h['EMA20'].iloc[-1]) if pd.notna(df1h['EMA20'].iloc[-1]) else None
                vwap1h   = float(df1h['VWAP'].iloc[-1]) if pd.notna(df1h['VWAP'].iloc[-1]) else None
                adx1h    = float(df1h['ADX'].iloc[-1])  if pd.notna(df1h['ADX'].iloc[-1])  else None

                # 'Strong trend' mode requires both 1h and 4h ADX strong
                try:
                    snap4h = quick_tf_snapshot(symbol, '4h', limit=120) or {}
                    adx4h = float(snap4h.get('ADX')) if snap4h.get('ADX') is not None else None
                except Exception:
                    adx4h = None
                strong_trend = (adx1h is not None and adx1h >= STRONG_TREND_ADX_1H) and (adx4h is not None and adx4h >= STRONG_TREND_ADX_4H)

                # Effective k
                k_eff = base_k

                # RVOL spike grace (use CLOSED 15m bars)
                if RVOL_SPIKE_GRACE:
                    try:
                        df15v = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                        if df15v is not None and len(df15v) > 20:
                            vol_closed = df15v['volume'].shift(1)  # closed bar
                            rv_mean = vol_closed.rolling(10).mean().iloc[-1]
                            rvals = []
                            for i in (2, 3, 4):  # last 3 closed bars
                                if rv_mean and rv_mean > RVOL_MEAN_MIN:
                                    rvals.append(float(vol_closed.iloc[-i] / rv_mean))
                            if rvals and max(rvals) >= RVOL_SPIKE_THRESHOLD:
                                rv_grace_until = time.time() + RVOL_K_BONUS_MINUTES * 60
                        if rv_grace_until and time.time() <= rv_grace_until:
                            k_eff += RVOL_SPIKE_K_BONUS
                    except Exception:
                        pass

                # Soft tighten via 15m signals (disabled in strong trend)
                if not strong_trend:
                    try:
                        df15t = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                        if df15t is not None and len(df15t) > 20:
                            willr = ta.willr(df15t['high'], df15t['low'], df15t['close'], length=14)
                            w = float(willr.iloc[-1]) if willr is not None else None  # -100..0
                            if w is not None:
                                if w > -20:         # very overbought → tighten
                                    k_eff = max(1.2, k_eff - 0.2)
                                elif w < -80:       # very oversold → a tad more room
                                    k_eff = k_eff + 0.1

                            rsi15 = float(ta.rsi(df15t['close'], length=14).iloc[-1])
                            macd15t = ta.macd(df15t['close'], fast=12, slow=26, signal=9)
                            macdh15t = float(macd15t['MACDh_12_26_9'].iloc[-1]) if macd15t is not None else None
                            if rsi15 >= NBUS_RSI15_OB and (macdh15t or 0) < 0:
                                soft_tighten_until = time.time() + 2 * 15 * 60
                        if soft_tighten_until and time.time() <= soft_tighten_until:
                            k_eff = max(1.2, k_eff - SOFT_TIGHTEN_K)
                    except Exception:
                        pass

                # Strong-trend floor for k (let it run)
                if strong_trend:
                    if   klass_local == "unstable": k_eff = max(k_eff, STRONG_TREND_K_UNSTABLE)
                    elif klass_local == "stable":   k_eff = max(k_eff, STRONG_TREND_K_STABLE)
                    else:                           k_eff = max(k_eff, STRONG_TREND_K_MEDIUM)

                # R tiers: BE and tier2 tighten
                gain = price - purchase_price
                if gain >= BE_R_MULT * initial_R:
                    be_stop = purchase_price * (1.0 + required_edge_pct()/100.0)
                    stop_price = max(stop_price, be_stop)
                if gain >= TIER2_R_MULT * initial_R:
                    k_eff = max(1.2, k_eff - TIER2_K_TIGHTEN)

                # Chandelier + ratchet + early-grace
                c_stop = chandelier_stop_long(df1h, atr_len=CHAN_ATR_LEN, hh_len=CHAN_LEN_HIGH, k=k_eff)
                candidate = stop_price
                if c_stop is not None:
                    candidate = max(candidate, c_stop, initial_stop)
                if (price - purchase_price) < 0.6 * initial_R:
                    grace_cap = purchase_price - 0.2 * initial_R
                    candidate = max(stop_price, min(candidate, grace_cap))
                stop_price = candidate

                # Volume-collapse exit (15m closed) — skip in strong trend
                if RVOL_COLLAPSE_EXIT_ENABLED and not strong_trend:
                    try:
                        df15c = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                        rvol15_closed = None
                        if df15c is not None and len(df15c) > 20:
                            vol_closed = df15c['volume'].shift(1)
                            rv_mean_15 = vol_closed.rolling(10).mean().iloc[-1]
                            if rv_mean_15 and rv_mean_15 > RVOL_MEAN_MIN:
                                rvol15_closed = float(vol_closed.iloc[-1] / rv_mean_15)
                        if held_long_enough and (rvol15_closed is not None) and (rvol15_closed < RVOL_COLLAPSE_EXIT):
                            logger.info(f"[collapse-exit] {symbol} 15m RVOL(closed)={rvol15_closed:.2f} < {RVOL_COLLAPSE_EXIT:.2f} — exiting.")
                            sell_symbol(symbol, amount, trade_id, source="collapse")
                            return
                    except Exception as e:
                        logger.debug(f"collapse-exit check error {symbol}: {e}")

                # Structure exit: EMA20 1h + Donchian lower
                structural_exit = False
                d_low = donchian_lower(df1h, length=DONCHIAN_LEN_EXIT)
                if d_low is not None and ema20_1h is not None:
                    if (close1h < ema20_1h) and (close1h < d_low):
                        if strong_trend:
                            try:
                                adx_series = ta.adx(df1h['high'], df1h['low'], df1h['close'], length=14)['ADX_14']
                                x = np.arange(6)
                                adx_slope = linregress(x, adx_series.iloc[-6:]).slope
                            except Exception:
                                adx_slope = None
                            structural_exit = bool(adx_slope is not None and adx_slope <= 0)
                        else:
                            structural_exit = True

                # Time-stop (3h) if MACDh15 < 0 on two closed bars
                if held >= TIME_STOP_HOURS * 3600:
                    try:
                        df15ts = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                        macd15ts = ta.macd(df15ts['close'], fast=12, slow=26, signal=9)['MACDh_12_26_9']
                        if float(macd15ts.iloc[-1]) < 0.0 and float(macd15ts.iloc[-2]) < 0.0:
                            sell_symbol(symbol, amount, trade_id, source="time_stop_3h_macdh15_neg")
                            return
                    except Exception as e:
                        logger.warning(f"[exit][{symbol}] time-stop MACDh15 check failed: {e}")

                # Optional: bar-based time-stop with extension if tape improving
                try:
                    n_bars = count_closed_bars_since(df1h, opened_ts)
                except Exception:
                    n_bars = None
                if n_bars is not None and n_bars >= TIME_STOP_BARS_1H:
                    tape_improving = True
                    try:
                        adx_series = ta.adx(df1h['high'], df1h['low'], df1h['close'], length=14)['ADX_14']
                        x = np.arange(6)
                        adx_slope = linregress(x, adx_series.iloc[-6:]).slope
                        vwap_ok = (not TAPE_IMPROVING_VWAP_REQ) or (vwap1h is None) or (close1h > vwap1h)
                        tape_improving = (adx_slope is not None and adx_slope >= TAPE_IMPROVING_ADX_SLOPE_MIN) and vwap_ok
                    except Exception:
                        tape_improving = False
                    if not tape_improving or n_bars >= TIME_STOP_EXTEND_BARS:
                        sell_symbol(symbol, amount, trade_id, source="time_stop_bars")
                        return

                # Price-triggered exit (respect micro-take guard)
                if price <= stop_price:
                    if price >= purchase_price:
                        edge_needed_pct = max(
                            required_edge_pct()*MIN_GAIN_OVER_FEES_MULT/EDGE_SAFETY_MULT,
                            required_edge_pct()
                        )
                        gain_pct = (price / purchase_price - 1.0) * 100.0
                        if gain_pct < edge_needed_pct:
                            if structural_exit:
                                sell_symbol(symbol, amount, trade_id, source="trail")
                                return
                        else:
                            sell_symbol(symbol, amount, trade_id, source="trail")
                            return
                    else:
                        sell_symbol(symbol, amount, trade_id, source="trail")
                        return

                # Rebound Guard (for structure-triggered exits)
                if structural_exit and REBOUND_GUARD_ENABLED:
                    try:
                        if rebound_pending_until is None:
                            rebound_pending_until = time.time() + REBOUND_WAIT_BARS_15M * 15 * 60
                            last_exit_trigger_reason = 'structure'
                            logger.info(f"[rebound-guard] {symbol} structural exit pending for {REBOUND_WAIT_BARS_15M}x15m")

                        cancel_exit = False
                        df15r = fetch_and_prepare_data_hybrid(symbol, timeframe="15m", limit=60)
                        if df15r is not None and len(df15r) > 21:
                            rsi_series = ta.rsi(df15r['close'], length=14)
                            rsi_now = float(rsi_series.iloc[-1]) if rsi_series is not None else None
                            rsi_prev = float(rsi_series.iloc[-3]) if rsi_series is not None else None
                            ema20_15_r = ta.ema(df15r['close'], length=20)
                            ema_ok = False
                            if REBOUND_EMA_RECLAIM and ema20_15_r is not None:
                                ema_ok = float(df15r['close'].iloc[-1]) > float(ema20_15_r.iloc[-1])
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
                            # tighten a touch and keep riding
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
        # >>> PATCH START: scratch logging
        try:
            if abs(pnl_pct) <= 0.20:  # scratch threshold (±0.20%)
                now_ts = time.time()
                arr = SCRATCH_LOG.setdefault(symbol, [])
                arr.append(now_ts)
                # keep only last 2h
                SCRATCH_LOG[symbol] = [t for t in arr if now_ts - t <= 7200]
        except Exception:
            pass
        # >>> PATCH END

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
            # ---- RSI nudges (keep your gentle behavior) ----
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

            # ---- ADX nudges (do NOT lower after wins) ----
            if not win and (adx_e is not None) and adx_e < adx_min:
                # Loss in low-trend context → raise floor
                new_adx_min = smooth_nudge(adx_min, adx_min + 1.0, *ADX_MIN_RANGE)
                if new_adx_min != adx_min:
                    adjustments["adxmin"] = {"old": adx_min, "new": new_adx_min}
                    adx_min = new_adx_min
            elif win and (adx_e is not None) and adx_e >= 30.0 and (adx_x or 0) >= 30.0:
                # Optional tiny tilt toward stronger trends on wins in strong trend
                new_adx_min = smooth_nudge(adx_min, adx_min + 0.5, *ADX_MIN_RANGE)
                if new_adx_min != adx_min:
                    adjustments["adxmin"] = {"old": adx_min, "new": new_adx_min}
                    adx_min = new_adx_min

            # ---- RVOL nudges (do NOT lower after wins) ----
            if not win and (rvol_e is not None) and rvol_e < max(1.0, rvol_base):
                # Loss with thin tape → raise base
                new_rvol = smooth_nudge(rvol_base, rvol_base + 0.10, *RVOL_BASE_RANGE)
                if new_rvol != rvol_base:
                    adjustments["rvolbase"] = {"old": rvol_base, "new": new_rvol}
                    rvol_base = new_rvol
            # (No "lower after win" step anymore)

            # ---- daily clip + mean reversion (unchanged) ----
            def _clip_delta(cur, new, max_abs=LEARN_DAILY_CLIP):
                return clamp(new, cur - max_abs, cur + max_abs)

            rsi_min = _clip_delta(lp["rsi_min"], rsi_min)
            rsi_max = _clip_delta(lp["rsi_max"], rsi_max)
            adx_min = _clip_delta(lp["adx_min"], adx_min)
            rvol_base = _clip_delta(lp["rvol_base"], rvol_base)

            # mean reversion to defaults
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
    """
    Volatility-aware position sizing with:
      - Confidence multiplier around RISK_FRACTION
      - Post-loss penalty within POST_LOSS_SIZING_WINDOW_SEC
      - Volatility class adjustment (stable/medium/unstable)
      - Friction guard (spread+slippage) scaling
      - Portfolio utilization and reserve protection
      - Binance filters (stepSize/minQty/minNotional) compliance
    Returns the amount (base units) to buy; 0.0 if not feasible.
    """
    if not price or price <= 0:
        return 0.0
    usdt_balance = float(usdt_balance or 0.0)
    if usdt_balance <= 0:
        return 0.0

    # 0) Respect reserve
    spendable = max(0.0, usdt_balance - RESERVE_USDT)
    if spendable < MIN_NOTIONAL:
        return 0.0

    # 1) Base by confidence (keeps your original curve)
    conf_mult = (confidence - 50) / 50.0
    conf_mult = max(0.0, min(conf_mult, 0.84))
    base_frac = RISK_FRACTION * (0.6 + 0.4 * conf_mult)

    # 2) Post-loss penalty (per symbol)
    if symbol and symbol in LAST_LOSS_INFO:
        try:
            last_dt = datetime.fromisoformat(LAST_LOSS_INFO[symbol]["ts"].replace("Z", ""))
            if (datetime.now(timezone.utc) - last_dt).total_seconds() < POST_LOSS_SIZING_WINDOW_SEC:
                base_frac *= POST_LOSS_SIZING_FACTOR
        except Exception:
            pass

    # 3) Volatility class tilt
    if symbol:
        prof = classify_symbol(symbol) or {}
        klass = prof.get("class", "medium")
        if klass == "stable":
            base_frac *= 1.10
        elif klass == "unstable":
            base_frac *= 0.85

    # 4) Portfolio utilization cap
    base_frac = min(base_frac, PORTFOLIO_MAX_UTIL)

    # Tentative notional
    notional = spendable * base_frac

    # 5) Friction guard (spread + est. slippage for this notional)
    try:
        if symbol:
            fric_pct = estimate_slippage_pct(symbol, notional=notional)  # percentage points
            # scale down if friction is high
            if fric_pct > 0.30:      # >0.30%
                notional *= 0.60
            elif fric_pct > 0.20:    # 0.21–0.30%
                notional *= 0.75
            elif fric_pct > 0.10:    # 0.11–0.20%
                notional *= 0.90
    except Exception:
        pass

    # Keep within spendable and respect MIN_NOTIONAL
    notional = max(MIN_NOTIONAL, min(notional, spendable))

    # 6) Convert to amount and conform to Binance filters
    desired_amt = notional / price
    amt, reason = _prepare_buy_amount(symbol, desired_amt, price) if symbol else (desired_amt, None)

    if amt <= 0:
        # Try to bump to the minimum that satisfies Binance filters (minNotional/minQty)
        step, min_qty, min_notional = _bn_filters(symbol) if symbol else (None, None, None)
        bump_amt = 0.0
        if min_notional:
            bump_amt = max(bump_amt, (min_notional / price))
        if min_qty:
            bump_amt = max(bump_amt, min_qty)
        if bump_amt > 0:
            amt, reason = _prepare_buy_amount(symbol, bump_amt, price) if symbol else (bump_amt, None)

    # Final sanity
    if amt <= 0 or (amt * price) < MIN_NOTIONAL:
        return 0.0

    # Also make sure we don't exceed spendable due to step rounding
    if (amt * price) > spendable:
        # reduce one step if possible
        step, _, _ = _bn_filters(symbol) if symbol else (None, None, None)
        if step:
            adj = max(0.0, amt - step)
            amt, _ = _prepare_buy_amount(symbol, adj, price)
        # if still too large, clip by spendable
        if (amt * price) > spendable:
            amt = (spendable / price)
            amt, _ = _prepare_buy_amount(symbol, amt, price)

    return float(max(0.0, amt))



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
    """
    Ejecuta un ciclo de decisión para un símbolo y, si corresponde, compra con
    verificación de presupuesto + filtros de Binance (stepSize, minQty, minNotional).
    """
    report = {"symbol": symbol, "action": "hold", "confidence": 50, "score": 0.0, "note": "", "executed": False}
    try:
        # 1) Ya hay posición abierta en este símbolo
        if has_open_position(symbol):
            report["note"] = "already holding"
            logger.info(f"{symbol}: {report['note']}")
            log_decision(symbol, 0.0, "hold", False)
            return report

        # 2) Chequeo de utilización de portafolio
        util, total, free, used = portfolio_utilization()
        if util > PORTFOLIO_MAX_UTIL:
            report["note"] = f"portfolio util {util:.2f} > max"
            logger.info(report["note"])
            log_decision(symbol, 0.0, "hold", False)
            return report

        # 3) Límite de operaciones abiertas
        open_trades = get_open_trades_count()
        if open_trades >= MAX_OPEN_TRADES:
            report["note"] = "max open trades"
            logger.info("Max open trades.")
            log_decision(symbol, 0.0, "hold", False)
            return report

        # 4) Liquidez disponible en USDT (respetando reserva)
        balances = exchange.fetch_balance()
        usdt = float(balances.get('free', {}).get('USDT', 0.0))
        free_after_reserve = max(0.0, usdt - RESERVE_USDT)
        if free_after_reserve < MIN_NOTIONAL:
            report["note"] = f"insufficient USDT after reserve ({usdt:.2f})"
            logger.info(report["note"])
            log_decision(symbol, 0.0, "hold", False)
            return report

        # 5) Decisión híbrida
        action, conf, score, note = hybrid_decision(symbol)
        report.update({"action": action, "confidence": conf, "score": float(score), "note": note})
        logger.info(f"{symbol}: decision={action} conf={conf} score={score:.1f} note={note}")

        if action != "buy":
            log_decision(symbol, score, "hold", False)
            return report

        # 6) Zona crítica: ejecutar compra bajo lock
        with buy_lock:
            # Revalidar límites dentro del lock
            open_trades = get_open_trades_count()
            if open_trades >= MAX_OPEN_TRADES:
                report["note"] = "max open trades (post-lock)"
                log_decision(symbol, score, "hold", False)
                return report

            balances = exchange.fetch_balance()
            usdt = float(balances.get('free', {}).get('USDT', 0.0))
            free_after_reserve = max(0.0, usdt - RESERVE_USDT)
            if free_after_reserve < MIN_NOTIONAL:
                report["note"] = "insufficient USDT (post-lock)"
                log_decision(symbol, score, "hold", False)
                return report

            # Precio actual y velas del TF de decisión
            price = fetch_price(symbol)
            if not price or price <= 0:
                report["note"] = "invalid price"
                log_decision(symbol, score, "hold", False)
                return report

            df = fetch_and_prepare_data_hybrid(symbol, limit=200, timeframe=DECISION_TIMEFRAME)
            if df is None or len(df) == 0:
                report["note"] = "no candles at entry check"
                log_decision(symbol, score, "hold", False)
                return report

            row = df.iloc[-1]
            atr_abs = float(row['ATR']) if pd.notna(row['ATR']) else price * 0.02

            # 7) Preflight guard (evita entradas "scratch"/mushy)
            ok, reason = preflight_buy_guard(symbol)
            if not ok:
                report["note"] = f"preflight block: {reason}"
                send_telegram_message(f"🛑 Skipping BUY {symbol} (preflight): {reason}")
                log_decision(symbol, score, "hold", False)
                return report

            # 8) Sizing en función de confianza, slippage y volatilidad
            #    - se capa al presupuesto disponible (free_after_reserve)
            #    - se ajusta a filtros de Binance (step/minQty/minNotional)
            amt_target_raw = size_position(price, usdt, conf, symbol)

            # Cap por presupuesto disponible (ligero margen para fees/slippage)
            max_affordable_amt = (free_after_reserve * 0.999) / price
            amt_target = min(amt_target_raw, max_affordable_amt)

            # Primer ajuste a filtros
            amount, reason = _prepare_buy_amount(symbol, amt_target, price)

            if amount <= 0:
                # Si falló por minNotional/minQty pero hay presupuesto, trata de elevar al mínimo requerido
                step, min_qty, min_notional = _bn_filters(symbol)
                need_amt = 0.0
                if min_notional:
                    need_amt = max(need_amt, float(min_notional) / price)
                if min_qty:
                    need_amt = max(need_amt, float(min_qty))

                if need_amt > 0 and (need_amt * price) <= free_after_reserve:
                    amount, reason = _prepare_buy_amount(symbol, need_amt, price)

            if amount <= 0:
                report["note"] = f"amount rejected by filters: {reason}"
                log_decision(symbol, score, "hold", False)
                return report

            trade_val = amount * price
            if trade_val > free_after_reserve:
                # Ultra-defensa: si después del piso de step/minQty nos pasamos de presupuesto, reduce un pelín
                # y vuelve a pisar al step. Si no entra, aborta.
                shrink_amt = max(0.0, (free_after_reserve * 0.998) / price)
                amount2, reason2 = _prepare_buy_amount(symbol, shrink_amt, price)
                if amount2 <= 0 or (amount2 * price) > free_after_reserve:
                    report["note"] = f"insufficient USDT for filtered amount ({trade_val:.2f} > {free_after_reserve:.2f})"
                    log_decision(symbol, score, "hold", False)
                    return report
                amount = amount2
                trade_val = amount * price

            # 9) Ejecutar compra y lanzar trailing
            order = execute_order_buy(symbol, amount, {
                'adx': float(row['ADX']) if pd.notna(row['ADX']) else None,
                'rsi': float(row['RSI']) if pd.notna(row['RSI']) else None,
                'rvol': (float(row['RVOL10']) if pd.notna(row['RVOL10']) else None),
                'atr': atr_abs,
                'score': float(score),
                'confidence': int(conf)
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
    VERSION = "v15"
    logger.info(f"Starting hybrid trader ({VERSION}, profile={STRATEGY_PROFILE})…")
    try:
        try: exchange.load_markets()
        except Exception as e: logger.warning(f"load_markets warning: {e}")

        startup_cleanup()

        while True:
            check_log_rotation()

            crash_active = crash_halt()
            cycle_decisions = []
            
                        # >>> FGI Cycle Gate (ANCHOR FGI3)
            try:
                mk_bull, fgi_v, tag = is_market_bullish()
                # Si hay extreme fear y además el régimen está apagado, pausamos ciclo (low-frequency en bears)
                if (fgi_v is not None and fgi_v <= FGI_EXTREME_FEAR) and (not market_regime_ok()):
                    logger.warning(f"⏸️  Cycle skipped due to Extreme Fear (FGI={fgi_v}, {tag}) with regime off.")
                    time.sleep(30)
                    continue
            except Exception as e:
                logger.debug(f"FGI gate error: {e}")
            
            # >>> STRATEGY ROUTER (ANCHOR SP2)
            try:
                if park_to_btc_if_needed():
                    # Estamos en BTC_PARK y régimen off → saltamos el escaneo de alts
                    time.sleep(30)
                    continue
            except Exception as e:
                logger.debug(f"strategy router error: {e}")


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
