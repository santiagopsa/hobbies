"""
Backtesting program for trader_v21.py
Simulates buys and exits with trailing stop using historical data

NOTE: This backtest uses a simplified version of the buy logic from hybrid_decision()
since the original function requires live exchange API calls. The trailing stop logic
is more closely aligned with the original dynamic_trailing_stop() function.

For best results:
1. Ensure you have API keys set in .env (for data fetching)
2. Adjust BACKTEST_START_DATE and BACKTEST_END_DATE as needed
3. The backtest will fetch historical data and simulate trades

Usage:
    python backtest_trader_v21.py --symbols BTC/USDT ETH/USDT --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations
import os
import sys
import time
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
from dotenv import load_dotenv

# Create necessary directories before importing trader_v21
# (trader_v21 tries to create log files on import)
log_dir = os.path.expanduser("~/hobbies")
os.makedirs(log_dir, exist_ok=True)

# Import all necessary functions and config from trader_v21
# We'll import the entire module to access all functions
import trader_v21 as trader

load_dotenv()

# =========================
# Backtesting Configuration
# =========================
INITIAL_BALANCE = float(os.getenv("BACKTEST_INITIAL_BALANCE", "1000.0"))
TRANSACTION_FEE = float(os.getenv("BACKTEST_FEE", "0.001"))  # 0.1% per side
MAX_OPEN_TRADES = int(os.getenv("BACKTEST_MAX_OPEN_TRADES", "5"))
RESERVE_USDT = float(os.getenv("BACKTEST_RESERVE_USDT", "20.0"))
RISK_FRACTION = float(os.getenv("BACKTEST_RISK_FRACTION", "0.18"))

# Timeframe for backtesting (should match DECISION_TIMEFRAME from trader_v21)
BACKTEST_TIMEFRAME = trader.DECISION_TIMEFRAME  # "1h"
BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "2024-01-01")
BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "2024-12-31")

# Symbols to backtest
BACKTEST_SYMBOLS = trader.SELECTED_CRYPTOS  # ['BTC/USDT', 'ETH/USDT', ...]

# =========================
# Backtesting State
# =========================
class BacktestTrade:
    """Represents an open trade in backtesting"""
    def __init__(self, symbol: str, entry_price: float, amount: float, entry_time: datetime, 
                 atr_abs: float, trade_id: str):
        self.symbol = symbol
        self.entry_price = entry_price
        self.amount = amount
        self.entry_time = entry_time
        self.atr_abs = atr_abs
        self.trade_id = trade_id
        self.highest_price = entry_price
        self.stop_price = None
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.is_closed = False
        
        # Trailing stop state
        self.initial_stop = entry_price - (trader.INIT_STOP_ATR_MULT * atr_abs)
        if self.initial_stop >= entry_price:
            self.initial_stop = entry_price * 0.985
        self.initial_R = max(entry_price - self.initial_stop, entry_price * 0.003)
        self.stop_price = self.initial_stop
        
        # Volatility class
        prof = trader.classify_symbol(symbol) or {}
        self.klass = prof.get("class", "medium")
        if self.klass == "unstable":
            self.base_k = trader.CHAN_K_UNSTABLE
        elif self.klass == "stable":
            self.base_k = trader.CHAN_K_STABLE
        else:
            self.base_k = trader.CHAN_K_MEDIUM
        
        # Trailing state
        self.trail_armed = False
        self.rv_grace_until = None
        self.soft_tighten_until = None
        self.rebound_pending_until = None

    def update_trailing_stop(self, current_price: float, df1h: pd.DataFrame, 
                           df15m: pd.DataFrame, df4h: pd.DataFrame = None, current_time: datetime = None):
        """Update trailing stop based on current price and indicators"""
        if self.is_closed:
            return
        
        # Update highest price
        if current_price > self.highest_price:
            self.highest_price = current_price
        
        # Calculate time held (use current_time if provided, otherwise use now)
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Ensure entry_time is timezone-aware
        entry_time = self.entry_time
        if isinstance(entry_time, pd.Timestamp):
            if entry_time.tz is None:
                entry_time = entry_time.tz_localize('UTC')
            entry_time = entry_time.to_pydatetime()
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        held_seconds = (current_time - entry_time).total_seconds()
        held_long_enough = held_seconds >= max(60, trader.MIN_HOLD_SECONDS)
        
        # Arm trailing after gain threshold
        gain_pct = (current_price / self.entry_price - 1.0) * 100.0
        if not self.trail_armed and gain_pct >= trader.ARM_TRAIL_PCT:
            self.trail_armed = True
        
        # Effective k
        k_eff = self.base_k
        
        # Strong trend check
        adx1h = float(df1h['ADX'].iloc[-1]) if pd.notna(df1h['ADX'].iloc[-1]) else None
        adx4h = None
        if df4h is not None and len(df4h) > 0:
            adx4h = float(df4h['ADX'].iloc[-1]) if pd.notna(df4h['ADX'].iloc[-1]) else None
        
        strong_trend = (adx1h is not None and adx1h >= trader.STRONG_TREND_ADX_1H) and \
                      (adx4h is not None and adx4h >= trader.STRONG_TREND_ADX_4H)
        
        # RVOL spike grace
        if trader.RVOL_SPIKE_GRACE and df15m is not None and len(df15m) > 20:
            try:
                vol_closed = df15m['volume'].shift(1)
                rv_mean = vol_closed.rolling(10).mean().iloc[-1]
                rvals = []
                for i in (2, 3, 4):
                    if rv_mean and rv_mean > trader.RVOL_MEAN_MIN:
                        rvals.append(float(vol_closed.iloc[-i] / rv_mean))
                if rvals and max(rvals) >= trader.RVOL_SPIKE_THRESHOLD:
                    self.rv_grace_until = time.time() + trader.RVOL_K_BONUS_MINUTES * 60
                if self.rv_grace_until and time.time() <= self.rv_grace_until:
                    k_eff += trader.RVOL_SPIKE_K_BONUS
            except Exception:
                pass
        
        # Soft tighten (disabled in strong trend)
        if not strong_trend and df15m is not None and len(df15m) > 20:
            try:
                rsi15 = float(ta.rsi(df15m['close'], length=14).iloc[-1])
                macd15 = ta.macd(df15m['close'], fast=12, slow=26, signal=9)
                macdh15 = float(macd15['MACDh_12_26_9'].iloc[-1]) if macd15 is not None else None
                if rsi15 >= trader.NBUS_RSI15_OB and (macdh15 or 0) < 0:
                    self.soft_tighten_until = time.time() + 2 * 15 * 60
                if self.soft_tighten_until and time.time() <= self.soft_tighten_until:
                    k_eff = max(1.2, k_eff - trader.SOFT_TIGHTEN_K)
            except Exception:
                pass
        
        # Strong trend floor for k
        if strong_trend:
            if self.klass == "unstable":
                k_eff = max(k_eff, trader.STRONG_TREND_K_UNSTABLE)
            elif self.klass == "stable":
                k_eff = max(k_eff, trader.STRONG_TREND_K_STABLE)
            else:
                k_eff = max(k_eff, trader.STRONG_TREND_K_MEDIUM)
        
        # R tiers: BE and tier2 tighten
        gain = current_price - self.entry_price
        if gain >= trader.BE_R_MULT * self.initial_R:
            be_stop = self.entry_price * (1.0 + trader.required_edge_pct() / 100.0)
            self.stop_price = max(self.stop_price, be_stop)
        if gain >= trader.TIER2_R_MULT * self.initial_R:
            k_eff = max(1.2, k_eff - trader.TIER2_K_TIGHTEN)
        
        # Chandelier stop
        c_stop = trader.chandelier_stop_long(df1h, atr_len=trader.CHAN_ATR_LEN, 
                                            hh_len=trader.CHAN_LEN_HIGH, k=k_eff)
        if c_stop is not None:
            self.stop_price = max(self.stop_price, c_stop, self.initial_stop)
        
        # Early grace cap
        if (current_price - self.entry_price) < 0.6 * self.initial_R:
            grace_cap = self.entry_price - 0.2 * self.initial_R
            self.stop_price = max(self.stop_price, min(self.stop_price, grace_cap))
        
        # If trailing not armed, cap stop
        if not self.trail_armed:
            grace_cap = max(self.initial_stop, self.entry_price * (1.0 - 0.004))
            self.stop_price = max(self.stop_price, min(self.stop_price, grace_cap))


class BacktestEngine:
    """Main backtesting engine"""
    def __init__(self, initial_balance: float = INITIAL_BALANCE):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.open_trades: Dict[str, BacktestTrade] = {}  # symbol -> BacktestTrade
        self.closed_trades: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_balance
        
        # Indicator tracking for analysis
        self.entry_indicators = []  # Store indicator values at entry for each trade
        
        # Exchange for fetching data (no API keys needed for public historical data)
        # For backtesting, we use public endpoints which don't require authentication
        # Set USE_API_KEYS_IN_BACKTEST=1 in .env if you want to use API keys (not recommended)
        use_api_keys = os.getenv('USE_API_KEYS_IN_BACKTEST', '0') == '1'
        
        exchange_config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }
        
        # Only add API keys if explicitly enabled (not recommended for backtesting)
        # Public historical data doesn't require authentication
        if use_api_keys:
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_SECRET', '')
            if api_key and api_secret:
                exchange_config['apiKey'] = api_key
                exchange_config['secret'] = api_secret
        
        self.exchange = ccxt.binance(exchange_config)
    
    def get_available_balance(self) -> float:
        """Get available balance for new trades"""
        reserved = sum(trade.amount * trade.entry_price for trade in self.open_trades.values())
        return max(0, self.balance - reserved - RESERVE_USDT)
    
    def can_open_trade(self, symbol: str) -> bool:
        """Check if we can open a new trade"""
        if symbol in self.open_trades:
            return False  # Already have position
        
        # >>> DYNAMIC MAX_OPEN_TRADES: Increase to 15 if breadth > 4 (momentum in market)
        max_trades_dynamic = MAX_OPEN_TRADES
        try:
            _, breadth_count = trader.compute_breadth()
            if breadth_count > 4:  # Momentum in market
                max_trades_dynamic = 15  # More diversification in good markets
        except Exception:
            pass
        
        if len(self.open_trades) >= max_trades_dynamic:
            return False
        return True
    
    def execute_buy(self, symbol: str, price: float, atr_abs: float, 
                   entry_time: datetime) -> Optional[BacktestTrade]:
        """Execute a buy order"""
        if not self.can_open_trade(symbol):
            return None
        
        available = self.get_available_balance()
        if available < trader.MIN_NOTIONAL:
            return None
        
        # Calculate position size
        # >>> DYNAMIC RISK: Increase in fear dips (RSI4h<35, FGI>10), reduce in extreme fear (FGI<=10)
        dynamic_risk_frac = RISK_FRACTION
        try:
            # Get RSI4h for BTC and FGI
            df4h_btc = trader.fetch_and_prepare_data_hybrid("BTC/USDT", timeframe="4h", limit=50)
            rsi4h_btc = float(df4h_btc['RSI'].iloc[-1]) if df4h_btc is not None and len(df4h_btc) > 0 and pd.notna(df4h_btc['RSI'].iloc[-1]) else None
            fgi_features = trader.fetch_fgi_features()
            fgi_v = fgi_features.get("value") if fgi_features else None
            
            if rsi4h_btc is not None and fgi_v is not None:
                if rsi4h_btc < 35 and fgi_v > 10:  # Fear but not extreme
                    dynamic_risk_frac = RISK_FRACTION * 1.5  # 27% for aggressive dips
                elif fgi_v <= 10:  # Extreme bad, halve
                    dynamic_risk_frac = RISK_FRACTION * 0.5
        except Exception:
            pass  # Fallback to base RISK_FRACTION
        
        risk_amount = available * dynamic_risk_frac
        position_value = min(risk_amount, available * 0.95)  # Use 95% of available
        
        # Account for fees
        position_value_after_fee = position_value / (1 + TRANSACTION_FEE)
        amount = position_value_after_fee / price
        
        # Check minimum notional
        if amount * price < trader.MIN_NOTIONAL:
            return None
        
        # Execute trade
        cost = amount * price * (1 + TRANSACTION_FEE)
        if cost > available:
            return None
        
        self.balance -= cost
        
        # Ensure entry_time is timezone-aware datetime
        if isinstance(entry_time, pd.Timestamp):
            if entry_time.tz is None:
                entry_time = entry_time.tz_localize('UTC')
            entry_time = entry_time.to_pydatetime()
        elif entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        trade_id = f"{symbol}-{entry_time.isoformat().replace(':','-')}"
        trade = BacktestTrade(symbol, price, amount, entry_time, atr_abs, trade_id)
        self.open_trades[symbol] = trade
        self.total_trades += 1
        
        # Store indicator values at entry for later analysis
        # This will be populated when we have the dataframe
        return trade
    
    def execute_sell(self, trade: BacktestTrade, exit_price: float, 
                    exit_time: datetime, reason: str):
        """Execute a sell order"""
        if trade.is_closed:
            return
        
        # Ensure exit_time is timezone-aware
        if isinstance(exit_time, pd.Timestamp):
            if exit_time.tz is None:
                exit_time = exit_time.tz_localize('UTC')
            exit_time = exit_time.to_pydatetime()
        elif exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)
        
        trade.is_closed = True
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = reason
        
        # Calculate proceeds (after fees)
        proceeds = trade.amount * exit_price * (1 - TRANSACTION_FEE)
        self.balance += proceeds
        
        # Calculate PnL
        entry_cost = trade.amount * trade.entry_price * (1 + TRANSACTION_FEE)
        pnl = proceeds - entry_cost
        pnl_pct = (exit_price / trade.entry_price - 1.0) * 100.0
        
        # Update statistics
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.total_pnl += pnl
        
        # Ensure entry_time is timezone-aware for duration calculation
        entry_time = trade.entry_time
        if isinstance(entry_time, pd.Timestamp):
            if entry_time.tz is None:
                entry_time = entry_time.tz_localize('UTC')
            entry_time = entry_time.to_pydatetime()
        elif entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        # Store closed trade
        self.closed_trades.append({
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'amount': trade.amount,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'duration_hours': (exit_time - entry_time).total_seconds() / 3600,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'highest_price': trade.highest_price,
            'max_gain_pct': (trade.highest_price / trade.entry_price - 1.0) * 100.0
        })
        
        # Remove from open trades
        del self.open_trades[trade.symbol]
    
    def check_exit_conditions(self, trade: BacktestTrade, current_price: float,
                             current_time: datetime, df1h: pd.DataFrame,
                             df15m: pd.DataFrame, df4h: pd.DataFrame = None) -> Tuple[bool, str]:
        """Check if trade should be exited"""
        if trade.is_closed:
            return False, ""
        
        # Ensure both times are timezone-aware datetime objects
        if isinstance(current_time, pd.Timestamp):
            if current_time.tz is None:
                current_time = current_time.tz_localize('UTC')
            current_time = current_time.to_pydatetime()
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        
        entry_time = trade.entry_time
        if isinstance(entry_time, pd.Timestamp):
            if entry_time.tz is None:
                entry_time = entry_time.tz_localize('UTC')
            entry_time = entry_time.to_pydatetime()
        elif entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        
        held_seconds = (current_time - entry_time).total_seconds()
        held_long_enough = held_seconds >= max(60, trader.MIN_HOLD_SECONDS)
        
        # Initial stop (can exit immediately)
        if current_price <= trade.initial_stop:
            if not held_long_enough:
                return True, "initial_stop_early"
            return True, "initial_stop"
        
        if not held_long_enough:
            return False, ""
        
        # Update trailing stop
        trade.update_trailing_stop(current_price, df1h, df15m, df4h, current_time)
        
        # Price-triggered exit
        if current_price <= trade.stop_price:
            # Check if we need edge over fees
            if current_price >= trade.entry_price:
                edge_needed = trader.required_edge_pct() * trader.MIN_GAIN_OVER_FEES_MULT / trader.EDGE_SAFETY_MULT
                gain_pct = (current_price / trade.entry_price - 1.0) * 100.0
                if gain_pct < edge_needed:
                    # Check for structural confirmation
                    try:
                        ema20_15 = ta.ema(df15m['close'], length=20).shift(1).iloc[-1]
                        close_m1 = df15m['close'].shift(1).iloc[-1]
                        close_m2 = df15m['close'].shift(2).iloc[-1]
                        two_below_ema = (close_m1 < ema20_15) and (close_m2 < ema20_15)
                        
                        adx_series = ta.adx(df1h['high'], df1h['low'], df1h['close'], length=14)['ADX_14']
                        from scipy.stats import linregress
                        adx_slope = linregress(np.arange(6), adx_series.iloc[-6:]).slope
                        
                        if two_below_ema and adx_slope <= 0:
                            return True, "trail_structural"
                    except Exception:
                        pass
                    return False, ""  # Don't exit yet
            return True, "trail"
        
        # Volume collapse exit
        if trader.RVOL_COLLAPSE_EXIT_ENABLED and df15m is not None and len(df15m) > 20:
            try:
                vol_closed = df15m['volume'].shift(1)
                rv_mean = vol_closed.rolling(10).mean().iloc[-1]
                if rv_mean and rv_mean > trader.RVOL_MEAN_MIN:
                    rvol15_closed = float(vol_closed.iloc[-1] / rv_mean)
                    if rvol15_closed < trader.RVOL_COLLAPSE_EXIT:
                        # Check strong trend
                        adx1h = float(df1h['ADX'].iloc[-1]) if pd.notna(df1h['ADX'].iloc[-1]) else None
                        adx4h = None
                        if df4h is not None and len(df4h) > 0:
                            adx4h = float(df4h['ADX'].iloc[-1]) if pd.notna(df4h['ADX'].iloc[-1]) else None
                        strong_trend = (adx1h is not None and adx1h >= trader.STRONG_TREND_ADX_1H) and \
                                      (adx4h is not None and adx4h >= trader.STRONG_TREND_ADX_4H)
                        if not strong_trend:
                            return True, "volume_collapse"
            except Exception:
                pass
        
        # Structure exit: EMA20 1h + Donchian lower
        try:
            ema20_1h = float(df1h['EMA20'].iloc[-1]) if pd.notna(df1h['EMA20'].iloc[-1]) else None
            close1h = float(df1h['close'].iloc[-1])
            d_low = trader.donchian_lower(df1h, length=trader.DONCHIAN_LEN_EXIT)
            
            if d_low is not None and ema20_1h is not None:
                if (close1h < ema20_1h) and (close1h < d_low):
                    adx1h = float(df1h['ADX'].iloc[-1]) if pd.notna(df1h['ADX'].iloc[-1]) else None
                    adx4h = None
                    if df4h is not None and len(df4h) > 0:
                        adx4h = float(df4h['ADX'].iloc[-1]) if pd.notna(df4h['ADX'].iloc[-1]) else None
                    strong_trend = (adx1h is not None and adx1h >= trader.STRONG_TREND_ADX_1H) and \
                                  (adx4h is not None and adx4h >= trader.STRONG_TREND_ADX_4H)
                    
                    if strong_trend:
                        # Check ADX slope
                        try:
                            adx_series = ta.adx(df1h['high'], df1h['low'], df1h['close'], length=14)['ADX_14']
                            from scipy.stats import linregress
                            adx_slope = linregress(np.arange(6), adx_series.iloc[-6:]).slope
                            if adx_slope is not None and adx_slope <= 0:
                                return True, "structure"
                        except Exception:
                            return True, "structure"
                    else:
                        return True, "structure"
        except Exception:
            pass
        
        # Time-stop (3h with MACDh15 < 0 twice)
        if held_seconds >= trader.TIME_STOP_HOURS * 3600:
            try:
                macd15 = ta.macd(df15m['close'], fast=12, slow=26, signal=9)['MACDh_12_26_9']
                if float(macd15.iloc[-1]) < 0.0 and float(macd15.iloc[-2]) < 0.0:
                    return True, "time_stop_3h"
            except Exception:
                pass
        
        # Bar-based time-stop (mejorado: no matar trades que van bien)
        try:
            n_bars = trader.count_closed_bars_since(df1h, trade.entry_time)
            if n_bars is not None and n_bars >= trader.TIME_STOP_BARS_1H:
                # Calculate current gain
                gain_pct = (current_price / trade.entry_price - 1.0) * 100.0
                
                # Check if tape is improving
                try:
                    adx_series = ta.adx(df1h['high'], df1h['low'], df1h['close'], length=14)['ADX_14']
                    from scipy.stats import linregress
                    adx_slope = linregress(np.arange(6), adx_series.iloc[-6:]).slope
                    vwap1h = float(df1h['VWAP'].iloc[-1]) if pd.notna(df1h['VWAP'].iloc[-1]) else None
                    close1h = float(df1h['close'].iloc[-1])
                    vwap_ok = (not trader.TAPE_IMPROVING_VWAP_REQ) or (vwap1h is None) or (close1h > vwap1h)
                    tape_improving = (adx_slope is not None and adx_slope >= trader.TAPE_IMPROVING_ADX_SLOPE_MIN) and vwap_ok
                    
                    if not tape_improving or n_bars >= trader.TIME_STOP_EXTEND_BARS:
                        # >>> MEJORADO: Sólo matamos si el trade no despegó o es claramente perdedor
                        # Si vamos claramente en verde, no usamos time-stop, dejamos que el trailing mande
                        if gain_pct < -0.5:
                            return True, "time_stop_bars_loss"      # perdedora lenta
                        if -0.3 <= gain_pct <= 0.5:
                            return True, "time_stop_bars_scratch"    # scratch cerca de flat
                        # Si vamos ya claramente en verde (>0.5%), no usamos time-stop
                except Exception:
                    # Fallback: solo matar si claramente perdedor
                    if gain_pct < -0.5:
                        return True, "time_stop_bars_loss"
                    if n_bars >= trader.TIME_STOP_EXTEND_BARS and gain_pct <= 0.5:
                        return True, "time_stop_bars_scratch"
        except Exception:
            pass
        
        return False, ""
    
    def update_equity(self, current_time: datetime, price_map: Dict[str, float] = None):
        """
        Update equity curve
        price_map: {symbol: current_price} al timestamp actual
        """
        open_value = 0.0
        for trade in self.open_trades.values():
            if price_map and trade.symbol in price_map:
                cur_price = price_map[trade.symbol]
            else:
                # Fallback to entry price if no current price available
                cur_price = trade.entry_price
            open_value += trade.amount * cur_price
        
        equity = self.balance + open_value
        
        # Ensure current_time is timezone-aware
        if isinstance(current_time, pd.Timestamp):
            if current_time.tz is None:
                current_time = current_time.tz_localize('UTC')
            current_time = current_time.to_pydatetime()
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        
        self.equity_curve.append((current_time, equity))
        
        # Update drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100.0 if self.peak_equity > 0 else 0.0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def get_statistics(self) -> Dict:
        """Get backtesting statistics"""
        if not self.closed_trades:
            return {}
        
        df = pd.DataFrame(self.closed_trades)
        
        win_rate = self.winning_trades / self.total_trades * 100.0 if self.total_trades > 0 else 0.0
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0.0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0.0
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if avg_loss < 0 and self.losing_trades > 0 else float('inf')
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100.0
        
        # Trend accuracy metrics (NEW - for trend prediction optimization)
        # Measure if price actually went up after entry (regardless of exit timing)
        trend_correct = 0  # Trades where price went up after entry
        trend_avg_gain = 0.0  # Average price gain (even if exited early)
        trend_max_gain = 0.0  # Average maximum gain reached
        
        if len(df) > 0:
            # For each trade, check if price went up after entry
            trend_correct_trades = df[df['max_gain_pct'] > 0]  # Price went up at some point
            trend_correct = len(trend_correct_trades)
            trend_accuracy = (trend_correct / len(df) * 100.0) if len(df) > 0 else 0.0
            
            # Average gain (using max_gain_pct which shows how much price went up)
            trend_avg_gain = df['max_gain_pct'].mean() if len(df) > 0 else 0.0
            
            # Average of maximum gains (how much price typically went up)
            trend_max_gain = df['max_gain_pct'].mean() if len(df) > 0 else 0.0
            
            # Trades where price went up AND stayed up (sustained trend)
            sustained_trends = df[(df['max_gain_pct'] > 2.0) & (df['pnl_pct'] > 0)]  # Went up 2%+ and closed positive
            sustained_trend_pct = (len(sustained_trends) / len(df) * 100.0) if len(df) > 0 else 0.0
        else:
            trend_accuracy = 0.0
            sustained_trend_pct = 0.0
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return_pct': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate_pct': win_rate,
            'total_pnl': self.total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': self.max_drawdown,
            'avg_trade_duration_hours': df['duration_hours'].mean() if len(df) > 0 else 0.0,
            'best_trade_pct': df['pnl_pct'].max() if len(df) > 0 else 0.0,
            'worst_trade_pct': df['pnl_pct'].min() if len(df) > 0 else 0.0,
            # Trend accuracy metrics (NEW)
            'trend_accuracy_pct': trend_accuracy,
            'trend_correct_trades': trend_correct,
            'trend_avg_gain_pct': trend_avg_gain,
            'trend_max_gain_pct': trend_max_gain,
            'sustained_trend_pct': sustained_trend_pct,
        }
    
    def get_indicator_analysis(self) -> Dict:
        """Analyze which indicators correlate with successful trades"""
        if not self.closed_trades:
            return {}
        
        # This will be populated during backtest
        if not hasattr(self, 'entry_indicators'):
            return {}
        
        df_indicators = pd.DataFrame(self.entry_indicators)
        df_trades = pd.DataFrame(self.closed_trades)
        
        # Merge to get indicator values at entry
        analysis = {}
        
        # Analyze each indicator
        indicators_to_analyze = ['ADX', 'RSI', 'RVOL10', 'EMA20_above_EMA50', 'price_slope']
        
        for indicator in indicators_to_analyze:
            if indicator not in df_indicators.columns:
                continue
            
            # Split into winning and losing trades
            winning = df_trades[df_trades['pnl'] > 0]
            losing = df_trades[df_trades['pnl'] <= 0]
            
            if len(winning) > 0 and len(losing) > 0:
                win_indicator = df_indicators.loc[winning.index, indicator].dropna()
                loss_indicator = df_indicators.loc[losing.index, indicator].dropna()
                
                if len(win_indicator) > 0 and len(loss_indicator) > 0:
                    analysis[indicator] = {
                        'winning_avg': float(win_indicator.mean()),
                        'losing_avg': float(loss_indicator.mean()),
                        'winning_median': float(win_indicator.median()),
                        'losing_median': float(loss_indicator.median()),
                        'difference': float(win_indicator.mean() - loss_indicator.mean()),
                        'winning_std': float(win_indicator.std()),
                        'losing_std': float(loss_indicator.std()),
                    }
        
        return analysis


def get_buy_signal_from_data(symbol: str, df: pd.DataFrame, current_price: float, params: Dict = None) -> Tuple[str, int, float, str]:
    """
    Simplified buy signal extraction from prepared dataframe
    This is a simplified version of hybrid_decision that works with historical data
    """
    if df is None or len(df) < 60:
        return "hold", 50, 0.0, "insufficient_data"
    
    try:
        base = symbol.split('/')[0]
        
        # Get learned params (with fallback to defaults)
        try:
            lp = trader.get_learn_params(base)
            RSI_MIN = lp["rsi_min"]
            RSI_MAX = lp["rsi_max"]
            ADX_MIN = lp["adx_min"]
            RVOL_BASE = lp["rvol_base"]
        except Exception:
            # Use defaults if learning params not available
            RSI_MIN = trader.RSI_MIN_DEFAULT
            RSI_MAX = trader.RSI_MAX_DEFAULT
            ADX_MIN = trader.ADX_MIN_DEFAULT
            RVOL_BASE = trader.RVOL_BASE_DEFAULT
        
        # Volatility profile (with fallback)
        try:
            prof = trader.classify_symbol(symbol)
            klass = prof["class"] if prof else "medium"
        except Exception:
            klass = "medium"  # Default to medium volatility
        
        # Adjust thresholds by volatility class
        if klass == "stable":
            RSI_MIN_K, RSI_MAX_K = max(RSI_MIN, 52), min(RSI_MAX, 68)
            ADX_MIN_K = max(18, ADX_MIN - 2)
            RVOL_BASE_K = max(0.8, RVOL_BASE * 0.8)
        elif klass == "unstable":
            RSI_MIN_K, RSI_MAX_K = max(48, RSI_MIN - 2), min(66, RSI_MAX)
            ADX_MIN_K = max(23, ADX_MIN)
            RVOL_BASE_K = max(1.05, RVOL_BASE)
        else:
            RSI_MIN_K, RSI_MAX_K = RSI_MIN, RSI_MAX
            ADX_MIN_K = ADX_MIN
            RVOL_BASE_K = RVOL_BASE
        
        # >>> PROXY FGI: RSI4h<RSI_FEAR_PROXY adjustments (fear proxy = best time for entries)
        try:
            df4h = trader.fetch_and_prepare_data_hybrid(symbol, timeframe="4h", limit=50)
            rsi4h = float(df4h['RSI'].iloc[-1]) if df4h is not None and len(df4h) > 0 and pd.notna(df4h['RSI'].iloc[-1]) else None
            
            if rsi4h is not None and rsi4h < trader.RSI_FEAR_PROXY:
                # Calculate vol_slope for ADX_MIN override
                vol_slope_fear = None
                try:
                    vol_slope_fear = trader.calculate_vol_slope(df, periods=10) if df is not None and len(df) >= 10 else None
                except Exception:
                    pass
                
                # >>> RELAJA ADX_MIN EN FEAR: Lower to 15 when RSI4h<35 and vol_slope>0, but keep 20 if vol_slope<0
                if vol_slope_fear is not None and vol_slope_fear > 0:
                    if ADX_MIN_K > 15:
                        ADX_MIN_K = 15  # Allow weaker trends in dips with positive vol
                else:
                    # Keep ADX_MIN at 20 if vol_slope<0 (avoid bad markets)
                    if ADX_MIN_K < 20:
                        ADX_MIN_K = 20
                
                # >>> OVERRIDE RSI RANGE EN FEAR: Allow RSI up to 65 in fear (overbought OK in dip market)
                if RSI_MAX_K < 65:
                    RSI_MAX_K = 65  # Allow mild overbought in dips
        except Exception:
            pass  # Fallback to normal thresholds
        
        row = df.iloc[-1]
        
        # Basic checks
        adx = float(row['ADX']) if pd.notna(row['ADX']) else 0.0
        rsi = float(row['RSI']) if pd.notna(row['RSI']) else 0.0
        rvol_1h = float(row['RVOL10']) if pd.notna(row['RVOL10']) else None
        
        # Hard blocks
        if adx < 17.5:
            return "hold", 50, 0.0, "ADX<17.5"
        if rvol_1h is not None and rvol_1h < 0.75:
            return "hold", 50, 0.0, f"RVOL1h too low: {rvol_1h:.2f}"
        if rsi < RSI_MIN_K or rsi > RSI_MAX_K:
            return "hold", 50, 0.0, f"RSI out of range: {rsi:.1f}"
        if adx < ADX_MIN_K:
            return "hold", 50, 0.0, f"ADX too low: {adx:.1f}"
        
        # Lane check
        in_lane = bool(row['EMA20'] > row['EMA50'] and row['close'] > row['EMA20'])
        
        # Trend validation - clear upward trend required
        # Get trend thresholds from params (optimizable) or use defaults
        if params is not None:
            MIN_SLOPE10_PCT = params.get('MIN_SLOPE10_PCT', 0.05)
            MIN_SLOPE20_PCT = params.get('MIN_SLOPE20_PCT', 0.02)
            MAX_NEAR_HIGH_PCT = params.get('MAX_NEAR_HIGH_PCT', -1.0)
        else:
            # Try to get from trader module if available (for optimization)
            try:
                MIN_SLOPE10_PCT = getattr(trader, 'MIN_SLOPE10_PCT', 0.05)
                MIN_SLOPE20_PCT = getattr(trader, 'MIN_SLOPE20_PCT', 0.02)
                MAX_NEAR_HIGH_PCT = getattr(trader, 'MAX_NEAR_HIGH_PCT', -1.0)
            except:
                MIN_SLOPE10_PCT = 0.05
                MIN_SLOPE20_PCT = 0.02
                MAX_NEAR_HIGH_PCT = -1.0
        
        price_slope10_pct = float(row.get('PRICE_SLOPE10_PCT', 0.0) or 0.0)
        price_slope20_pct = float(row.get('PRICE_SLOPE20_PCT', 0.0) or 0.0)
        price_near_high = float(row.get('PRICE_NEAR_HIGH_PCT', -10.0) or -10.0)
        
        # Validate clear upward trend (not just a peak)
        # 1. Short-term slope must be positive and significant
        # 2. Medium-term slope should also be positive (sustained trend)
        # 3. Price should not be too close to recent high (avoid buying at peaks)
        clear_uptrend = (
            price_slope10_pct > MIN_SLOPE10_PCT and
            (pd.isna(price_slope20_pct) or price_slope20_pct > MIN_SLOPE20_PCT) and
            price_near_high < MAX_NEAR_HIGH_PCT
        )
        
        # Hard block: reject if no clear uptrend
        if not clear_uptrend:
            trend_reason = []
            if price_slope10_pct <= MIN_SLOPE10_PCT:
                trend_reason.append(f"slope10_pct={price_slope10_pct:.3f}% <= {MIN_SLOPE10_PCT:.3f}%")
            if not pd.isna(price_slope20_pct) and price_slope20_pct <= MIN_SLOPE20_PCT:
                trend_reason.append(f"slope20_pct={price_slope20_pct:.3f}% <= {MIN_SLOPE20_PCT:.3f}%")
            if price_near_high >= MAX_NEAR_HIGH_PCT:
                trend_reason.append(f"near_high={price_near_high:.2f}% >= {MAX_NEAR_HIGH_PCT:.2f}% (too close to peak)")
            return "hold", 50, 0.0, f"no_clear_uptrend: {', '.join(trend_reason)}"
        
        # Score calculation (simplified)
        score = 0.0
        notes = []
        
        if in_lane:
            score += 2.0
            notes.append("in_lane")
        
        if adx >= ADX_MIN_K:
            score += 1.5
            notes.append(f"ADX={adx:.1f}")
        
        if rvol_1h and rvol_1h >= RVOL_BASE_K:
            score += 1.5
            notes.append(f"RVOL={rvol_1h:.2f}")
        
        if rsi >= RSI_MIN_K and rsi <= RSI_MAX_K:
            score += 1.0
            notes.append(f"RSI={rsi:.1f}")
        
        # Strong trend bonus (both slopes positive and significant)
        if price_slope10_pct > 0.1 and (pd.isna(price_slope20_pct) or price_slope20_pct > 0.05):
            score += 1.5
            notes.append(f"strong_trend (slope10={price_slope10_pct:.3f}%, slope20={price_slope20_pct:.3f}%)")
        elif price_slope10_pct > 0.05:
            score += 0.8
            notes.append(f"clear_uptrend (slope10={price_slope10_pct:.3f}%)")
        
        score_gate = trader.get_score_gate()
        
        # >>> LOG OVERRIDE IN FEAR: Debug score 0.0 buys when fear override active
        fear_override_active = False
        try:
            df4h_btc_log = trader.fetch_and_prepare_data_hybrid("BTC/USDT", timeframe="4h", limit=50)
            rsi4h_btc_log = float(df4h_btc_log['RSI'].iloc[-1]) if df4h_btc_log is not None and len(df4h_btc_log) > 0 and pd.notna(df4h_btc_log['RSI'].iloc[-1]) else None
            if rsi4h_btc_log is not None and rsi4h_btc_log < 35:
                fear_override_active = True
        except Exception:
            pass
        
        # >>> LOG OVERRIDE IN FEAR: Debug low score buys when fear override active
        if score < trader.SCORE_GATE_HARD_MIN and fear_override_active:
            print(f"[FEAR-OVERRIDE] {symbol}: Fear override buy with score {score:.1f} despite gate {score_gate:.1f} (RSI4h={rsi4h_btc_log:.1f}<35)")
        
        if score >= score_gate:
            conf = int(min(50 + 10*(1 if in_lane else 0) + 10*(1 if clear_uptrend else 0), 100))
            return "buy", conf, score, ", ".join(notes)
        else:
            return "hold", 50, score, f"score {score:.1f} < gate {score_gate:.1f}"
    
    except Exception as e:
        return "hold", 50, 0.0, f"error: {e}"


def prepare_data_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe with indicators (similar to fetch_and_prepare_data_hybrid)"""
    if df is None or len(df) < 50:
        return None
    
    df = df.copy()
    df['EMA20'] = ta.ema(df['close'], length=20)
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['RSI'] = ta.rsi(df['close'], length=14)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14'] if adx is not None and not adx.empty else np.nan
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    rv_mean = df['volume'].rolling(10).mean()
    rv_ok = rv_mean.iloc[-1] and rv_mean.iloc[-1] > trader.RVOL_MEAN_MIN
    df['RVOL10'] = (df['volume'] / rv_mean) if rv_ok else np.nan
    # Calculate multiple price slopes for trend validation
    if len(df) >= 20:
        from scipy.stats import linregress
        # Short-term slope (last 10 periods) - for immediate trend
        x10 = np.arange(10)
        df.loc[df.index[-1], 'PRICE_SLOPE10'] = linregress(x10, df['close'].iloc[-10:]).slope
        
        # Medium-term slope (last 20 periods) - for sustained trend
        x20 = np.arange(20)
        df.loc[df.index[-1], 'PRICE_SLOPE20'] = linregress(x20, df['close'].iloc[-20:]).slope
        
        # Calculate slope as percentage of price (normalized)
        current_price = df['close'].iloc[-1]
        df.loc[df.index[-1], 'PRICE_SLOPE10_PCT'] = (df.loc[df.index[-1], 'PRICE_SLOPE10'] / current_price * 100) if current_price > 0 else 0
        df.loc[df.index[-1], 'PRICE_SLOPE20_PCT'] = (df.loc[df.index[-1], 'PRICE_SLOPE20'] / current_price * 100) if current_price > 0 else 0
        
        # Check if price is near recent high (to avoid buying at peaks)
        recent_high_20 = df['high'].iloc[-20:].max()
        df.loc[df.index[-1], 'PRICE_NEAR_HIGH_PCT'] = ((current_price / recent_high_20 - 1) * 100) if recent_high_20 > 0 else 0
        
        # Volume slope
        df.loc[df.index[-1], 'VOL_SLOPE10'] = linregress(x10, df['volume'].iloc[-10:]).slope
    elif len(df) >= 10:
        from scipy.stats import linregress
        x = np.arange(10)
        df.loc[df.index[-1], 'PRICE_SLOPE10'] = linregress(x, df['close'].iloc[-10:]).slope
        current_price = df['close'].iloc[-1]
        df.loc[df.index[-1], 'PRICE_SLOPE10_PCT'] = (df.loc[df.index[-1], 'PRICE_SLOPE10'] / current_price * 100) if current_price > 0 else 0
        df.loc[df.index[-1], 'PRICE_SLOPE20'] = np.nan
        df.loc[df.index[-1], 'PRICE_SLOPE20_PCT'] = np.nan
        df.loc[df.index[-1], 'PRICE_NEAR_HIGH_PCT'] = np.nan
        df.loc[df.index[-1], 'VOL_SLOPE10'] = linregress(x, df['volume'].iloc[-10:]).slope
    else:
        df['PRICE_SLOPE10'] = np.nan
        df['PRICE_SLOPE10_PCT'] = np.nan
        df['PRICE_SLOPE20'] = np.nan
        df['PRICE_SLOPE20_PCT'] = np.nan
        df['PRICE_NEAR_HIGH_PCT'] = np.nan
        df['VOL_SLOPE10'] = np.nan
    try:
        df['VWAP'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
    except Exception:
        df['VWAP'] = np.nan
    return df


def fetch_historical_data(symbol: str, timeframe: str, start_date: str, end_date: str, 
                         exchange: ccxt.Exchange) -> pd.DataFrame:
    """Fetch historical OHLCV data with retry logic"""
    max_retries = 3
    retry_delay = 2  # seconds
    
    try:
        since = exchange.parse8601(start_date + 'T00:00:00Z')
        until = exchange.parse8601(end_date + 'T23:59:59Z')
        
        all_ohlcv = []
        current = since
        
        while current < until:
            # Retry logic for each batch
            ohlcv = None
            for attempt in range(max_retries):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
                    if ohlcv:
                        break
                except ccxt.AuthenticationError as e:
                    # If it's an auth error, try without API keys (public data doesn't need them)
                    print(f"  Warning: Authentication error for {symbol}, trying public access...")
                    if hasattr(exchange, 'apiKey') and exchange.apiKey:
                        # Remove API keys and retry
                        exchange.apiKey = None
                        exchange.secret = None
                        try:
                            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
                            if ohlcv:
                                break
                        except Exception as e2:
                            print(f"  Error fetching {symbol} (attempt {attempt+1}/{max_retries}): {e2}")
                    else:
                        print(f"  Error fetching {symbol} (attempt {attempt+1}/{max_retries}): {e}")
                except Exception as e:
                    print(f"  Error fetching {symbol} (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
            
            if not ohlcv:
                print(f"  Failed to fetch data for {symbol} after {max_retries} attempts")
                break
            
            all_ohlcv.extend(ohlcv)
            current = ohlcv[-1][0] + 1
            
            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)
        
        if not all_ohlcv:
            print(f"  No data retrieved for {symbol}")
            return None
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)
        
        # Prepare with indicators
        df = prepare_data_from_ohlcv(df)
        
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_backtest(symbols: List[str] = None, start_date: str = None, end_date: str = None,
                initial_balance: float = INITIAL_BALANCE, entry_only: bool = False, 
                entry_hold_periods: int = 24) -> BacktestEngine:
    """Run backtest on specified symbols"""
    if symbols is None:
        symbols = BACKTEST_SYMBOLS
    if start_date is None:
        start_date = BACKTEST_START_DATE
    if end_date is None:
        end_date = BACKTEST_END_DATE
    
    engine = BacktestEngine(initial_balance)
    exchange = engine.exchange
    
    # Patch trader_v21's exchange to use our backtest exchange (without API keys)
    # This prevents authentication errors when trader_v21 functions are called
    original_trader_exchange = trader.exchange
    trader.exchange = exchange
    
    try:
        print(f"Starting backtest from {start_date} to {end_date}")
        print(f"Symbols: {symbols}")
        print(f"Initial balance: ${initial_balance:.2f}")
        print("-" * 80)
        
        # Fetch historical data for all symbols
        symbol_data = {}
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            df = fetch_historical_data(symbol, BACKTEST_TIMEFRAME, start_date, end_date, exchange)
            if df is not None and len(df) > 200:  # Need enough data for indicators
                symbol_data[symbol] = df
                print(f"  Loaded {len(df)} candles for {symbol}")
            else:
                print(f"  Failed to load data for {symbol}")
        
        if not symbol_data:
            print("No data loaded. Exiting.")
            trader.exchange = original_trader_exchange  # Restore before early return
            return engine
        
        # Get all unique timestamps (timestamp is the index after prepare_data_from_ohlcv)
        all_timestamps = set()
        for df in symbol_data.values():
            all_timestamps.update(df.index.tolist())
        all_timestamps = sorted(all_timestamps)
        
        print(f"\nProcessing {len(all_timestamps)} time steps...")
        print("-" * 80)
        
        # Process each timestamp
        for i, timestamp in enumerate(all_timestamps):
            if i % 100 == 0:
                print(f"Processing step {i}/{len(all_timestamps)}... Balance: ${engine.balance:.2f}, Open trades: {len(engine.open_trades)}")
            
            # Check exits for open trades (skip if entry_only mode)
            if not entry_only:
                for symbol, trade in list(engine.open_trades.items()):
                    if symbol not in symbol_data:
                        continue
                    
                    df = symbol_data[symbol]
                    # timestamp is the index, so check if it exists
                    if timestamp not in df.index:
                        continue
                    current_row = df.loc[[timestamp]]
                    
                    current_price = float(current_row['close'].iloc[0])
                    current_idx = df.index.get_loc(timestamp)
                    
                    # Get dataframes for indicators (use data up to current point)
                    df1h = df.iloc[:current_idx+1].copy() if current_idx >= 60 else None
                    if df1h is None or len(df1h) < 60:
                        continue
                    
                    # Prepare data for indicators
                    try:
                        df1h = prepare_data_from_ohlcv(df1h)
                        if df1h is None or len(df1h) < 60:
                            continue
                    except Exception:
                        continue
                    
                    # For 15m and 4h, we'll use 1h data as approximation (in full backtest, fetch separately)
                    df15m = df1h
                    df4h = None
                    
                    should_exit, reason = engine.check_exit_conditions(
                        trade, current_price, timestamp, df1h, df15m, df4h
                    )
                    
                    if should_exit:
                        engine.execute_sell(trade, current_price, timestamp, reason)
                        print(f"SELL {symbol} @ ${current_price:.4f} | Reason: {reason} | PnL: ${(current_price - trade.entry_price) * trade.amount:.2f}")
            else:
                # Entry-only mode: Close trades after fixed period (no trailing stops)
                for symbol, trade in list(engine.open_trades.items()):
                    if symbol not in symbol_data:
                        continue
                    
                    df = symbol_data[symbol]
                    if timestamp not in df.index:
                        continue
                    
                    current_row = df.loc[[timestamp]]
                    current_price = float(current_row['close'].iloc[0])
                    
                    # Calculate how many periods have passed since entry
                    try:
                        entry_idx = df.index.get_loc(trade.entry_time) if trade.entry_time in df.index else None
                        current_idx = df.index.get_loc(timestamp)
                        
                        if entry_idx is not None:
                            periods_held = current_idx - entry_idx
                            
                            # Update highest price (for trend measurement)
                            if current_price > trade.highest_price:
                                trade.highest_price = current_price
                            
                            # Close after fixed period
                            if periods_held >= entry_hold_periods:
                                engine.execute_sell(trade, current_price, timestamp, f"entry_only_hold_{entry_hold_periods}periods")
                                price_change_pct = ((current_price / trade.entry_price - 1) * 100)
                                print(f"SELL {symbol} @ ${current_price:.4f} | Entry-only: {periods_held} periods | Price change: {price_change_pct:.2f}%")
                    except (KeyError, ValueError):
                        # Entry time not in index, skip
                        continue
            
            # Check for new buy signals
            for symbol in symbols:
                if symbol not in symbol_data:
                    continue
                if not engine.can_open_trade(symbol):
                    continue
                
                df = symbol_data[symbol]
                # timestamp is the index, so check if it exists
                if timestamp not in df.index:
                    continue
                current_row = df.loc[[timestamp]]
                
                current_idx = df.index.get_loc(timestamp)
                if current_idx < 200:  # Need enough history
                    continue
                
                # Prepare data up to current point
                try:
                    df_prep = df.iloc[:current_idx+1].copy()
                    if len(df_prep) < 200:  # Need enough history
                        continue
                    
                    df_prep = prepare_data_from_ohlcv(df_prep)
                    if df_prep is None or len(df_prep) < 60:
                        continue
                    
                    # Temporarily override trader's fetch function to use our prepared data
                    # We need to mock the exchange calls in hybrid_decision
                    # For now, we'll create a simplified version that works with prepared data
                    
                    # Get buy signal using a modified approach
                    # Since hybrid_decision uses live exchange calls, we'll need to adapt
                    # For backtesting, we'll extract the core logic
                    current_price = float(current_row['close'].iloc[0])
                    # Pass trend params if available from trader module
                    trend_params = {}
                    try:
                        trend_params['MIN_SLOPE10_PCT'] = getattr(trader, 'MIN_SLOPE10_PCT', None)
                        trend_params['MIN_SLOPE20_PCT'] = getattr(trader, 'MIN_SLOPE20_PCT', None)
                        trend_params['MAX_NEAR_HIGH_PCT'] = getattr(trader, 'MAX_NEAR_HIGH_PCT', None)
                        trend_params = {k: v for k, v in trend_params.items() if v is not None}
                    except:
                        pass
                    action, conf, score, note = get_buy_signal_from_data(symbol, df_prep, current_price, params=trend_params if trend_params else None)
                    
                    if action == "buy" and score > 0:
                        atr_abs = float(df_prep['ATR'].iloc[-1]) if pd.notna(df_prep['ATR'].iloc[-1]) else None
                        
                        if atr_abs is None or atr_abs <= 0:
                            continue
                        
                        # Extract indicator values at entry
                        row = df_prep.iloc[-1]
                        # Calculate trend_score: composite metric combining slopes and distance from high
                        # Based on analysis: successful trends have better price_slope10_pct, price_slope20_pct, and price_near_high_pct
                        price_slope10_pct_val = float(row.get('PRICE_SLOPE10_PCT', 0.0) or 0.0)
                        price_slope20_pct_val = float(row.get('PRICE_SLOPE20_PCT', 0.0) or 0.0) if pd.notna(row.get('PRICE_SLOPE20_PCT')) else 0.0
                        price_near_high_pct_val = float(row.get('PRICE_NEAR_HIGH_PCT', -10.0) or -10.0)
                        
                        # trend_score: weighted combination
                        # 0.7 * slope10 (short-term momentum) + 0.3 * slope20 (medium-term trend) + 0.2 * (-near_high) (distance from peak, negative = good)
                        trend_score = (
                            0.7 * price_slope10_pct_val +
                            0.3 * price_slope20_pct_val +
                            0.2 * (-price_near_high_pct_val)  # lejos del máximo reciente suma (negative = good)
                        )
                        
                        entry_indicators = {
                            'ADX': float(row['ADX']) if pd.notna(row['ADX']) else None,
                            'RSI': float(row['RSI']) if pd.notna(row['RSI']) else None,
                            'RVOL10': float(row['RVOL10']) if pd.notna(row['RVOL10']) else None,
                            'EMA20_above_EMA50': bool(row['EMA20'] > row['EMA50']) if pd.notna(row['EMA20']) and pd.notna(row['EMA50']) else None,
                            'price_slope10_pct': price_slope10_pct_val,
                            'price_slope20_pct': price_slope20_pct_val if pd.notna(row.get('PRICE_SLOPE20_PCT')) else None,
                            'price_near_high_pct': price_near_high_pct_val,
                            'trend_score': trend_score,  # Composite trend metric
                            'close_above_EMA20': bool(row['close'] > row['EMA20']) if pd.notna(row['EMA20']) else None,
                            'ATR_pct': float((atr_abs / current_price) * 100) if atr_abs else None,
                            'score': score,
                            'confidence': conf,
                        }
                        
                        trade = engine.execute_buy(symbol, current_price, atr_abs, timestamp)
                        if trade:
                            # Store indicators with trade index
                            engine.entry_indicators.append(entry_indicators)
                            print(f"BUY {symbol} @ ${current_price:.4f} | Score: {score:.2f} | Conf: {conf}% | {note}")
                except Exception as e:
                    # Silently skip errors
                    pass
            
            # Update equity curve (con precios actuales)
            price_map = {
                sym: float(df.loc[timestamp]['close']) 
                for sym, df in symbol_data.items() 
                if timestamp in df.index
            }
            engine.update_equity(timestamp, price_map)
        
        print("\n" + "=" * 80)
        print("BACKTEST COMPLETE")
        print("=" * 80)
        
        stats = engine.get_statistics()
        print("\n--- PERFORMANCE STATISTICS ---")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Indicator Analysis
        print("\n" + "=" * 80)
        print("INDICATOR ANALYSIS - What predicts steady upward trends?")
        print("=" * 80)
        
        if engine.closed_trades and len(engine.entry_indicators) > 0:
            # Ensure we have matching entries
            min_len = min(len(engine.closed_trades), len(engine.entry_indicators))
            engine.entry_indicators = engine.entry_indicators[:min_len]
            
            df_trades = pd.DataFrame(engine.closed_trades[:min_len])
            df_indicators = pd.DataFrame(engine.entry_indicators[:min_len])
            
            # Separate winning and losing trades
            winning_trades = df_trades[df_trades['pnl'] > 0]
            losing_trades = df_trades[df_trades['pnl'] <= 0]
            
            print(f"\nTotal trades analyzed: {len(df_trades)}")
            print(f"Winning trades: {len(winning_trades)} ({len(winning_trades)/len(df_trades)*100:.1f}%)")
            print(f"Losing trades: {len(losing_trades)} ({len(losing_trades)/len(df_trades)*100:.1f}%)")
            
            # Analyze each indicator
            print("\n--- INDICATOR COMPARISON (Winning vs Losing Trades) ---")
            
            indicators_to_check = ['ADX', 'RSI', 'RVOL10', 'price_slope10_pct', 'price_slope20_pct', 'trend_score', 'ATR_pct', 'score']
            
            for indicator in indicators_to_check:
                if indicator not in df_indicators.columns:
                    continue
                
                win_vals = df_indicators.loc[winning_trades.index, indicator].dropna() if len(winning_trades) > 0 else pd.Series()
                loss_vals = df_indicators.loc[losing_trades.index, indicator].dropna() if len(losing_trades) > 0 else pd.Series()
                
                if len(win_vals) > 0 and len(loss_vals) > 0:
                    win_mean = win_vals.mean()
                    loss_mean = loss_vals.mean()
                    diff = win_mean - loss_mean
                    diff_pct = (diff / abs(loss_mean) * 100) if loss_mean != 0 else 0
                    
                    print(f"\n{indicator}:")
                    print(f"  Winning trades avg: {win_mean:.3f} (median: {win_vals.median():.3f})")
                    print(f"  Losing trades avg:  {loss_mean:.3f} (median: {loss_vals.median():.3f})")
                    print(f"  Difference: {diff:+.3f} ({diff_pct:+.1f}%)")
                    
                    # Statistical significance (simple t-test approximation)
                    if len(win_vals) > 5 and len(loss_vals) > 5:
                        win_std = win_vals.std()
                        loss_std = loss_vals.std()
                        pooled_std = np.sqrt((win_std**2 + loss_std**2) / 2)
                        if pooled_std > 0:
                            t_stat = diff / (pooled_std * np.sqrt(1/len(win_vals) + 1/len(loss_vals)))
                            print(f"  T-statistic: {t_stat:.2f} ({'***' if abs(t_stat) > 2.5 else '**' if abs(t_stat) > 2.0 else '*' if abs(t_stat) > 1.5 else ''})")
            
            # Analyze combinations
            print("\n--- BEST INDICATOR COMBINATIONS FOR WINNING TRADES ---")
            
            # Find trades with best returns
            top_trades = df_trades.nlargest(min(10, len(df_trades)), 'pnl_pct')
            if len(top_trades) > 0:
                top_indicators = df_indicators.loc[top_trades.index]
                print(f"\nTop {len(top_trades)} trades (by return %):")
                for indicator in ['ADX', 'RSI', 'RVOL10', 'score']:
                    if indicator in top_indicators.columns:
                        avg = top_indicators[indicator].mean()
                        print(f"  {indicator}: {avg:.3f} (avg)")
            
            # Analyze steady trends (trades that went up and stayed up)
            steady_winners = df_trades[
                (df_trades['pnl_pct'] > 2.0) &  # At least 2% gain
                (df_trades['max_gain_pct'] > df_trades['pnl_pct'] * 0.8)  # Didn't give back much
            ]
            
            if len(steady_winners) > 0:
                print(f"\n--- STEADY TREND INDICATORS ({len(steady_winners)} trades) ---")
                steady_indicators = df_indicators.loc[steady_winners.index]
                for indicator in ['ADX', 'RSI', 'RVOL10', 'price_slope10_pct', 'price_slope20_pct', 'trend_score', 'score']:
                    if indicator in steady_indicators.columns:
                        avg = steady_indicators[indicator].mean()
                        print(f"  {indicator}: {avg:.3f} (avg)")
        else:
            print("No indicator data available for analysis")
    
    finally:
        # Restore original exchange to avoid affecting other operations
        trader.exchange = original_trader_exchange
    
    return engine


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest trader_v21 strategy')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to backtest (e.g., BTC/USDT ETH/USDT)')
    parser.add_argument('--start', type=str, default=BACKTEST_START_DATE, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=BACKTEST_END_DATE, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=INITIAL_BALANCE, help='Initial balance')
    parser.add_argument('--output', type=str, default=None, help='Output file for results JSON')
    
    args = parser.parse_args()
    
    engine = run_backtest(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        initial_balance=args.balance
    )
    
    if args.output:
        results = {
            'statistics': engine.get_statistics(),
            'trades': engine.closed_trades,
            'equity_curve': [(ts.isoformat(), eq) for ts, eq in engine.equity_curve]
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

