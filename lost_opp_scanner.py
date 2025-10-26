#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lost Opportunities Scanner (standalone)
--------------------------------------
Scans exchange OHLCV data (via ccxt), detects "candidate" reversal/breakout events,
measures forward performance (1h/6h/24h), and summarizes which strict filters
(MACDh<=0, not in EMA lane, ADX gray-zone, edge check) would have blocked winners.

Usage examples:
  python lost_opp_scanner.py --symbols "SOL/USDT,BTC/USDT,ETH/USDT" --timeframe 15m --days 7
  python lost_opp_scanner.py --exchange binanceusdm --market-type swap --symbols "SOL/USDT:USDT,BTC/USDT:USDT" --timeframe 15m --days 10
  python lost_opp_scanner.py --min-move 3.0 --edge-need 0.8 --entry-edge-mult 2.0

Outputs:
  - opportunities.sqlite3  (SQLite DB with detected events and resolved outcomes)
  - opp_events.csv         (flat CSV of events + outcomes)
  - opp_summary.png        (bar chart: hit-rate by blocking reason)
  - stdout summary tables

Dependencies:
  pip install ccxt pandas numpy matplotlib
"""
import argparse
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# matplotlib only used for final chart
import matplotlib.pyplot as plt

try:
    import ccxt
except Exception as e:
    print("ERROR: ccxt is required. Install with: pip install ccxt", file=sys.stderr)
    raise

# ---------------------------
# Parameters / Defaults
# ---------------------------

DEFAULT_MIN_MOVE = 3.0           # % threshold to count as "hit" within the window
DEFAULT_WINDOWS = [60, 360, 1440]# minutes: 1h, 6h, 24h
DEFAULT_DEDUP_MIN = 30           # dedup per symbol per this many minutes
DEFAULT_DB = "opportunities.sqlite3"

# Edge / gating knobs (mimic strict filters you might be using)
DEFAULT_EDGE_NEED = 0.8          # required edge (%), e.g., fees+slippage cushion
DEFAULT_ENTRY_EDGE_MULT = 2.0    # need_edge * this must be <= projected move, else "edge block"
DEFAULT_ADX_GRAY_MIN = 17.0      # gray-zone low
DEFAULT_ADX_GRAY_MAX = 22.0      # gray-zone high
DEFAULT_LANE_EMA_FAST = 20
DEFAULT_LANE_EMA_SLOW = 50
DEFAULT_RVOL_N = 20
DEFAULT_ATR_N = 14
DEFAULT_RSI_N = 14

# ---------------------------
# Indicator helpers (pure numpy)
# ---------------------------

def ema(series: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average."""
    if n <= 1:
        return series.copy()
    alpha = 2.0 / (n + 1.0)
    out = np.empty_like(series, dtype=float)
    out[:] = np.nan
    if len(series) > 0:
        out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out

def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Wilder smoothing
    def rma(x, n):
        out = np.empty_like(x, dtype=float)
        out[:] = np.nan
        if len(x) == 0:
            return out
        out[n-1] = np.nanmean(x[:n])
        alpha = 1.0 / n
        for i in range(n, len(x)):
            prev = out[i-1] if not math.isnan(out[i-1]) else np.nanmean(x[:n])
            out[i] = (prev * (n - 1) + x[i]) / n
        return out
    avg_gain = rma(gain, n)
    avg_loss = rma(loss, n)
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:n] = np.nan
    return rsi

def macd(close: np.ndarray, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    tr = true_range(high, low, close)
    # Wilder smoothing
    out = np.empty_like(tr, dtype=float); out[:] = np.nan
    if len(tr) >= n:
        out[n-1] = np.nanmean(tr[:n])
        for i in range(n, len(tr)):
            out[i] = (out[i-1] * (n - 1) + tr[i]) / n
    return out

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)[1:]
    # Wilder smoothing
    def wilder(x, n):
        out = np.empty_like(x, dtype=float); out[:] = np.nan
        if len(x) >= n:
            out[n-1] = np.nanmean(x[:n])
            for i in range(n, len(x)):
                out[i] = (out[i-1] * (n - 1) + x[i]) / n
        return out

    tr_n = wilder(tr, n)
    plus_dm_n = wilder(plus_dm, n)
    minus_dm_n = wilder(minus_dm, n)

    plus_di = 100.0 * (plus_dm_n / tr_n)
    minus_di = 100.0 * (minus_dm_n / tr_n)
    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = wilder(dx, n)

    # Align lengths to original arrays (pad with nan at start)
    def pad(x):
        pad_len = len(close) - len(x)
        if pad_len > 0:
            return np.concatenate((np.array([np.nan] * pad_len), x))
        return x
    return pad(plus_di), pad(minus_di), pad(adx)

def sma(series: np.ndarray, n: int) -> np.ndarray:
    out = np.convolve(series, np.ones(n)/n, mode='full')
    out = out[:len(series)]
    # Proper rolling SMA: set first n-1 to nan
    out[:n-1] = np.nan
    return out

# ---------------------------
# Data / Exchange helpers
# ---------------------------

def make_exchange(name: str, market_type: str = "spot"):
    ex = getattr(ccxt, name)()
    if hasattr(ex, 'options') and isinstance(ex.options, dict):
        ex.options['defaultType'] = market_type
    ex.load_markets()
    return ex

def fetch_ohlcv_all(ex, symbol: str, timeframe: str, since_ms: int, limit: int = 1500) -> List[List[float]]:
    """Fetch OHLCV from 'since' to now in chunks."""
    out = []
    ms_per_candle = ex.parse_timeframe(timeframe) * 1000
    now_ms = int(time.time() * 1000)
    cursor = since_ms
    while cursor < now_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        out.extend(batch)
        last = batch[-1][0]
        # Avoid infinite loop
        cursor = max(cursor + ms_per_candle, last + ms_per_candle)
        # Rate limit
        time.sleep(getattr(ex, 'rateLimit', 200) / 1000.0)
        if len(batch) < limit:
            break
    return out

# ---------------------------
# Core scanner
# ---------------------------

from dataclasses import dataclass

@dataclass
class Event:
    ts: int
    iso: str
    symbol: str
    price: float
    reason_event: str  # e.g., RSI_rebound, Breakout_macd_leq0, ADX_gray_push
    # Block flags (simulate strict filters that might have prevented entry)
    block_macdh_leq0: int
    block_not_in_lane: int
    block_edge: int
    block_adx_gray: int
    # Features for analysis
    rsi: float
    macdh: float
    adx: float
    rvol: float
    atr_pct: float
    ema_fast: float
    ema_slow: float

@dataclass
class Outcome:
    win_min: int
    ret_close_pct: float
    mfe_pct: float
    mae_pct: float
    hit: int

def detect_events(df: pd.DataFrame,
                  adx_gray=(DEFAULT_ADX_GRAY_MIN, DEFAULT_ADX_GRAY_MAX),
                  ema_fast_n=DEFAULT_LANE_EMA_FAST,
                  ema_slow_n=DEFAULT_LANE_EMA_SLOW) -> List[int]:
    """
    Return indices where an "interesting" candidate event happened:
      - RSI rebound: crossed up 30
      - Breakout while MACDh<=0 and not in lane yet
      - ADX gray-zone push with positive slope
    """
    idxs = []

    # RSI rebound
    rsi_prev = df['rsi'].shift(1)
    cond_rsi = (rsi_prev <= 30) & (df['rsi'] > 30)

    # Breakout above EMA_fast while MACDh<=0 and not in lane (EMA_fast <= EMA_slow or close <= EMA_fast)
    in_lane = (df['ema_fast'] > df['ema_slow']) & (df['close'] > df['ema_fast'])
    cross_fast = (df['close'] > df['ema_fast']) & (df['close'].shift(1) <= df['ema_fast'].shift(1))
    cond_break_macd = cross_fast & (df['macdh'] <= 0) & (~in_lane)

    # ADX gray-zone with positive slope and price slope up
    adx_low, adx_high = adx_gray
    adx_slope = df['adx'] - df['adx'].shift(1)
    price_slope = df['close'] - df['close'].shift(3)
    cond_adx_gray = (df['adx'] >= adx_low) & (df['adx'] <= adx_high) & (adx_slope > 0) & (price_slope > 0)

    # Combine
    any_event = cond_rsi | cond_break_macd | cond_adx_gray

    for i, flag in enumerate(any_event):
        if bool(flag) and i > 50:  # ignore first 50 bars for indicator warmup
            idxs.append(i)
    return idxs

def projected_move_pct(atr_pct: float, k: float = 2.0) -> float:
    """Crude projection: k * ATR% as a likely move range (percent)."""
    return (atr_pct * 100.0) * k

def build_events(df: pd.DataFrame,
                 need_edge_pct: float,
                 entry_edge_mult: float,
                 adx_gray=(DEFAULT_ADX_GRAY_MIN, DEFAULT_ADX_GRAY_MAX)) -> List[Event]:
    idxs = detect_events(df, adx_gray=adx_gray)
    events: List[Event] = []
    for i in idxs:
        in_lane = (df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]) and (df['close'].iloc[i] > df['ema_fast'].iloc[i])
        proj = projected_move_pct(df['atr_pct'].iloc[i])
        need = need_edge_pct * entry_edge_mult
        e = Event(
            ts = int(df['ts'].iloc[i]),
            iso = datetime.utcfromtimestamp(df['ts'].iloc[i]/1000).isoformat() + 'Z',
            symbol = str(df['symbol'].iloc[i]),
            price = float(df['close'].iloc[i]),
            reason_event = (
                'RSI_rebound' if (df['rsi'].shift(1).iloc[i] <= 30 and df['rsi'].iloc[i] > 30) else
                ('Breakout_macd_leq0' if ((df['close'].iloc[i] > df['ema_fast'].iloc[i]) and (df['macdh'].iloc[i] <= 0) and (not in_lane)) else
                 'ADX_gray_push')
            ),
            block_macdh_leq0 = 1 if df['macdh'].iloc[i] <= 0 else 0,
            block_not_in_lane = 0 if in_lane else 1,
            block_edge = 1 if proj < need else 0,
            block_adx_gray = 1 if (df['adx'].iloc[i] >= DEFAULT_ADX_GRAY_MIN and df['adx'].iloc[i] <= DEFAULT_ADX_GRAY_MAX) else 0,
            rsi = float(df['rsi'].iloc[i]),
            macdh = float(df['macdh'].iloc[i]),
            adx = float(df['adx'].iloc[i]),
            rvol = float(df['rvol'].iloc[i]),
            atr_pct = float(df['atr_pct'].iloc[i]),
            ema_fast = float(df['ema_fast'].iloc[i]),
            ema_slow = float(df['ema_slow'].iloc[i]),
        )
        events.append(e)
    return events

def forward_outcomes(df: pd.DataFrame, idx: int, windows_min: List[int], min_move: float) -> List[Outcome]:
    outcomes: List[Outcome] = []
    ts0 = df['ts'].iloc[idx]
    price0 = df['close'].iloc[idx]
    # Precompute intraperiod MFE/MAE up to each window end
    for win in windows_min:
        # find first index whose ts >= ts0 + win*60*1000
        target = ts0 + win*60*1000
        j = idx
        while j < len(df)-1 and df['ts'].iloc[j] < target:
            j += 1
        highs = df['high'].iloc[idx:j+1]
        lows  = df['low'].iloc[idx:j+1]
        close = df['close'].iloc[j]
        mfe = float(highs.max()) if len(highs) else float(price0)
        mae = float(lows.min())  if len(lows)  else float(price0)
        ret_close = (close / price0 - 1.0) * 100.0
        mfe_pct = (mfe / price0 - 1.0) * 100.0
        mae_pct = (mae / price0 - 1.0) * 100.0
        outcomes.append(Outcome(win_min=win,
                                ret_close_pct=round(ret_close, 3),
                                mfe_pct=round(mfe_pct, 3),
                                mae_pct=round(mae_pct, 3),
                                hit=1 if mfe_pct >= min_move else 0))
    return outcomes

# ---------------------------
# SQLite storage
# ---------------------------

def sql_init(db_path: str):
    with sqlite3.connect(db_path) as cx:
        cx.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            reason_event TEXT NOT NULL,
            block_macdh_leq0 INTEGER,
            block_not_in_lane INTEGER,
            block_edge INTEGER,
            block_adx_gray INTEGER,
            rsi REAL, macdh REAL, adx REAL, rvol REAL, atr_pct REAL,
            ema_fast REAL, ema_slow REAL
        );
        """)
        cx.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            win_min INTEGER NOT NULL,
            ret_close_pct REAL,
            mfe_pct REAL,
            mae_pct REAL,
            hit INTEGER,
            FOREIGN KEY(event_id) REFERENCES events(id)
        );
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_ev_symbol_ts ON events(symbol, ts);")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_out_event ON outcomes(event_id);")
        cx.commit()

def sql_insert_event(cx, e: Event) -> int:
    cur = cx.execute("""
        INSERT INTO events(ts, symbol, price, reason_event, block_macdh_leq0, block_not_in_lane, block_edge, block_adx_gray,
                           rsi, macdh, adx, rvol, atr_pct, ema_fast, ema_slow)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (e.iso, e.symbol, e.price, e.reason_event, e.block_macdh_leq0, e.block_not_in_lane, e.block_edge, e.block_adx_gray,
          e.rsi, e.macdh, e.adx, e.rvol, e.atr_pct, e.ema_fast, e.ema_slow))
    return cur.lastrowid

def sql_insert_outcomes(cx, event_id: int, outs: List[Outcome]):
    cx.executemany("""
        INSERT INTO outcomes(event_id, win_min, ret_close_pct, mfe_pct, mae_pct, hit)
        VALUES(?,?,?,?,?,?)
    """, [(event_id, o.win_min, o.ret_close_pct, o.mfe_pct, o.mae_pct, o.hit) for o in outs])

# ---------------------------
# Chart
# ---------------------------

def chart_block_hit_rate(db_path: str, out_path: str, min_move: float):
    with sqlite3.connect(db_path) as cx:
        df = pd.read_sql_query("""
          SELECT e.id, e.reason_event,
                 e.block_macdh_leq0, e.block_not_in_lane, e.block_edge, e.block_adx_gray,
                 o.win_min, o.hit
          FROM events e JOIN outcomes o ON o.event_id = e.id
          WHERE o.win_min = 1440  -- 24h horizon
        """, cx)

    if df.empty:
        return None

    # Compute hit-rate per block type
    block_cols = ['block_macdh_leq0', 'block_not_in_lane', 'block_edge', 'block_adx_gray']
    rows = []
    for col in block_cols:
        sub = df[df[col] == 1]
        if len(sub) == 0:
            hr = 0.0
            n = 0
        else:
            hr = 100.0 * sub['hit'].mean()
            n = len(sub)
        rows.append((col, n, hr))

    chart_df = pd.DataFrame(rows, columns=['block', 'n_events', 'hit_rate_pct']).sort_values(['hit_rate_pct', 'n_events'], ascending=[False, False])
    plt.figure(figsize=(8,5))
    plt.title(f"Lost Opportunities — Hit-rate among events BLOCKED by filter (≥ +{min_move}% within 24h)")
    plt.barh(chart_df['block'][::-1], chart_df['hit_rate_pct'][::-1])
    plt.xlabel("Hit-rate (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path

# ---------------------------
# Main pipeline
# ---------------------------

def process_symbol(ex, symbol: str, timeframe: str, days: int,
                   need_edge_pct: float, entry_edge_mult: float,
                   windows_min: List[int], db_path: str, min_move: float):

    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    raw = fetch_ohlcv_all(ex, symbol, timeframe, since, limit=1500)
    if not raw or len(raw) < 200:
        print(f"[warn] insufficient data for {symbol}")
        return 0, 0

    df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
    df['symbol'] = symbol

    # Indicators
    close = df['close'].to_numpy(dtype=float)
    high  = df['high'].to_numpy(dtype=float)
    low   = df['low'].to_numpy(dtype=float)
    vol   = df['volume'].to_numpy(dtype=float)

    df['ema_fast'] = ema(close, DEFAULT_LANE_EMA_FAST)
    df['ema_slow'] = ema(close, DEFAULT_LANE_EMA_SLOW)
    df['rsi'] = rsi(close, DEFAULT_RSI_N)
    macd_line, signal_line, hist = macd(close, 12, 26, 9)
    df['macd'] = macd_line; df['macds'] = signal_line; df['macdh'] = hist
    atr_vals = atr(high, low, close, DEFAULT_ATR_N)
    df['atr'] = atr_vals
    df['atr_pct'] = atr_vals / close
    _, _, adx_vals = adx(high, low, close, DEFAULT_ATR_N)
    df['adx'] = adx_vals
    df['rvol'] = vol / sma(vol, DEFAULT_RVOL_N)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(df) < 200:
        print(f"[warn] post-indicator rows <200 for {symbol}")
        return 0, 0

    events = build_events(df, need_edge_pct, entry_edge_mult)
    if not events:
        print(f"[info] no events for {symbol}")
        return 0, 0

    with sqlite3.connect(db_path) as cx:
        inserted, hits = 0, 0
        for e in events:
            event_id = sql_insert_event(cx, e)
            outs = forward_outcomes(df, df.index[df['ts']==e.ts][0], windows_min, min_move)
            sql_insert_outcomes(cx, event_id, outs)
            inserted += 1
            # Count 24h hit
            for o in outs:
                if o.win_min == 1440 and o.hit == 1:
                    hits += 1
                    break
        cx.commit()
    print(f"[done] {symbol}: events={inserted}, 24h hits={hits}")
    return inserted, hits

def export_csv(db_path: str, out_csv: str):
    with sqlite3.connect(db_path) as cx:
        df = pd.read_sql_query("""
          SELECT e.ts, e.symbol, e.price, e.reason_event,
                 e.block_macdh_leq0, e.block_not_in_lane, e.block_edge, e.block_adx_gray,
                 e.rsi, e.macdh, e.adx, e.rvol, e.atr_pct, e.ema_fast, e.ema_slow,
                 MAX(CASE WHEN o.win_min=60  THEN o.mfe_pct END) AS mfe_1h,
                 MAX(CASE WHEN o.win_min=360 THEN o.mfe_pct END) AS mfe_6h,
                 MAX(CASE WHEN o.win_min=1440 THEN o.mfe_pct END) AS mfe_24h
          FROM events e
          JOIN outcomes o ON o.event_id = e.id
          GROUP BY e.id
          ORDER BY e.ts DESC
        """, cx)
    if not df.empty:
        df.to_csv(out_csv, index=False)
        return out_csv
    return None

def main():
    global DEFAULT_MIN_MOVE
    ap = argparse.ArgumentParser(description="Lost Opportunities Scanner (standalone)")
    ap.add_argument("--exchange", default="binance", help="ccxt exchange id (binance, binanceusdm, kucoin, etc.)")
    ap.add_argument("--market-type", default="spot", help="ccxt defaultType (spot, swap)")
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g., 'SOL/USDT,BTC/USDT')")
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument("--days", type=int, default=7, help="How many days back to scan")
    ap.add_argument("--min-move", type=float, default=DEFAULT_MIN_MOVE, help="Hit threshold (%% MFE within 24h)")
    ap.add_argument("--edge-need", type=float, default=DEFAULT_EDGE_NEED, help="Required edge %% (fees+slippage cushion)")
    ap.add_argument("--entry-edge-mult", type=float, default=DEFAULT_ENTRY_EDGE_MULT, help="need_edge * this <= projected move")
    ap.add_argument("--db", default=DEFAULT_DB, help="SQLite path")
    ap.add_argument("--csv", default="opp_events.csv", help="CSV output path")
    ap.add_argument("--chart", default="opp_summary.png", help="Chart output path")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols provided.", file=sys.stderr)
        sys.exit(1)

    sql_init(args.db)
    ex = make_exchange(args.exchange, args.market_type)

    total_events, total_hits = 0, 0
    for sym in symbols:
        ev, hit = process_symbol(ex, sym, args.timeframe, args.days,
                                 args.edge_need, args.entry_edge_mult,
                                 DEFAULT_WINDOWS, args.db, args.min_move)
        total_events += ev; total_hits += hit

    csv_path = export_csv(args.db, args.csv)
    chart_path = chart_block_hit_rate(args.db, args.chart, args.min_move)

    print("\n=== SUMMARY ===")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Total events: {total_events}")
    print(f"24h hits (MFE ≥ +{args.min_move}%): {total_hits}")
    if csv_path:
        print(f"CSV: {csv_path}")
    if chart_path:
        print(f"Chart: {chart_path}")
    print(f"DB: {args.db}")

if __name__ == "__main__":
    main()
