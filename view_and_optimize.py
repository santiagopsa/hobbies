import sqlite3
import json
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import statsmodels.api as sm
import time
import optuna
import joblib
import concurrent.futures
import logging
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_NAME = "trading_real.db"

# Create optimized_weights table if missing
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS optimized_weights (
        symbol TEXT PRIMARY KEY,
        weights TEXT NOT NULL,
        last_optimized TEXT NOT NULL
    )
''')
conn.commit()

# Coin-specific weights
COIN_WEIGHTS = {
    'BTC': {
        'category': 'stable',
        'MIN_ADX': 15,
        'MIN_RELATIVE_VOLUME': 0.05,
        'MAX_SUPPORT_DISTANCE': 0.05,
        'VOLUME_SPIKE_FACTOR': 1.1,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 4,
            'short_vol_trend': 3,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 2,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 2,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'ETH': {
        'category': 'growth',
        'MIN_ADX': 20,
        'MIN_RELATIVE_VOLUME': 0.06,
        'MAX_SUPPORT_DISTANCE': 0.03,
        'VOLUME_SPIKE_FACTOR': 1.3,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 3,
            'short_vol_trend': 5,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'BNB': {
        'category': 'growth',
        'MIN_ADX': 18,
        'MIN_RELATIVE_VOLUME': 0.06,
        'MAX_SUPPORT_DISTANCE': 0.04,
        'VOLUME_SPIKE_FACTOR': 1.2,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 3,
            'short_vol_trend': 5,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'SOL': {
        'category': 'high_vol',
        'MIN_ADX': 20,
        'MIN_RELATIVE_VOLUME': 0.3,
        'MAX_SUPPORT_DISTANCE': 0.02,
        'VOLUME_SPIKE_FACTOR': 1.5,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 3,
            'short_vol_trend': 5,
            'price_trend': 1,
            'support_dist': 3,
            'adx_bonus': 2,
            'rsi_penalty': -2,
            'oversold': 3,
            'vol_spike': 2,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'XRP': {
        'category': 'growth',
        'MIN_ADX': 20,
        'MIN_RELATIVE_VOLUME': 0.08,
        'MAX_SUPPORT_DISTANCE': 0.03,
        'VOLUME_SPIKE_FACTOR': 1.4,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 3,
            'short_vol_trend': 5,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'DOGE': {
        'category': 'high_vol',
        'MIN_ADX': 20,
        'MIN_RELATIVE_VOLUME': 0.4,
        'MAX_SUPPORT_DISTANCE': 0.02,
        'VOLUME_SPIKE_FACTOR': 1.6,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 4,
            'short_vol_trend': 6,
            'price_trend': 1,
            'support_dist': 2,
            'adx_bonus': 2,
            'rsi_penalty': -2,
            'oversold': 3,
            'vol_spike': 3,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'TON': {
        'category': 'growth',
        'MIN_ADX': 22,
        'MIN_RELATIVE_VOLUME': 0.07,
        'MAX_SUPPORT_DISTANCE': 0.035,
        'VOLUME_SPIKE_FACTOR': 1.3,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 3.5,
            'short_vol_trend': 5,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'ADA': {
        'category': 'growth',
        'MIN_ADX': 20,
        'MIN_RELATIVE_VOLUME': 0.06,
        'MAX_SUPPORT_DISTANCE': 0.03,
        'VOLUME_SPIKE_FACTOR': 1.3,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 3,
            'short_vol_trend': 5,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'TRX': {
        'category': 'growth',
        'MIN_ADX': 19,
        'MIN_RELATIVE_VOLUME': 0.07,
        'MAX_SUPPORT_DISTANCE': 0.04,
        'VOLUME_SPIKE_FACTOR': 1.25,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 3,
            'short_vol_trend': 5,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    },
    'AVAX': {
        'category': 'high_vol',
        'MIN_ADX': 20,
        'MIN_RELATIVE_VOLUME': 0.2,
        'MAX_SUPPORT_DISTANCE': 0.025,
        'VOLUME_SPIKE_FACTOR': 1.45,
        'OVERSOLD_THRESHOLD': 0.95,
        'score_weights': {
            'rel_vol_bonus': 3,
            'short_vol_trend': 5,
            'price_trend': 1.5,
            'support_dist': 3,
            'adx_bonus': 2,
            'rsi_penalty': -1.5,
            'oversold': 2.5,
            'vol_spike': 2,
            'daily_vol_bonus': 1,
            'ml_prob': 3
        }
    }
}

# Initial weights
INITIAL_WEIGHTS = {
    "BTC/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "ETH/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "BNB/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "SOL/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "XRP/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "DOGE/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "TON/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "ADA/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "TRX/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    },
    "AVAX/USDT": {
        "MIN_ADX": 15,
        "MIN_RELATIVE_VOLUME": 0.5,
        "MAX_SUPPORT_DISTANCE": 0.05,
        "VOLUME_SPIKE_FACTOR": 1.0,
        "MAX_SPREAD": 0.01,
        "MIN_DAILY_VOL": 50000
    }
}

# Merge weights
def get_weights(symbol):
    base_coin = symbol.split('/')[0]
    coin_weight = COIN_WEIGHTS.get(base_coin, {})
    initial_weight = INITIAL_WEIGHTS.get(symbol, {})
    merged = {**initial_weight, **coin_weight}
    return merged

# SELECTED_CRYPTOS (default to single symbol for testing)
SELECTED_CRYPTOS = ['BTC/USDT']  # Set to list(INITIAL_WEIGHTS.keys()) for all symbols

# Manual RSI with epsilon
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period if len(seed[seed >= 0]) > 0 else 0
    down = -seed[seed < 0].sum() / period if len(seed[seed < 0]) > 0 else 0
    rs = up / (down + 1e-10) if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / (down + 1e-10) if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return pd.Series(rsi, index=prices.index)

# Manual ADX
def calculate_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_dm = high - high.shift()
    minus_dm = low.shift() - low
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    return adx

# Dynamic parameter ranges by category
def get_param_ranges(category):
    if category == 'high_vol':
        return {
            'min_adx': (10, 50),
            'min_rel_vol': (0.1, 2.0),
            'max_support': (0.01, 0.4),
            'vol_spike': (0.8, 2.5)
        }
    elif category == 'stable':
        return {
            'min_adx': (5, 30),
            'min_rel_vol': (0.05, 1.0),
            'max_support': (0.02, 0.2),
            'vol_spike': (0.8, 1.5)
        }
    else:  # growth
        return {
            'min_adx': (10, 40),
            'min_rel_vol': (0.08, 1.5),
            'max_support': (0.015, 0.3),
            'vol_spike': (0.8, 2.0)
        }

# Dummy market sentiment
def get_market_sentiment():
    return 70, "Greed"

# Calculate trading strategy
def calculate_established_strategy(indicators, data=None, symbol=None, ml_model=None, ml_prob=0.5, best_winrate=50):
    base_coin = symbol.split('/')[0]
    weights = get_weights(symbol)

    rsi = indicators.get('rsi', None)
    relative_volume = indicators.get('relative_volume', None)
    price_trend = indicators.get('price_trend', 'insufficient_data')
    short_volume_trend = indicators.get('short_volume_trend', 'insufficient_data')
    current_price = indicators.get('current_price', 0)
    support_level = indicators.get('support_level', None)
    adx = indicators.get('adx', None)
    spread = indicators.get('spread', 0)
    daily_vol = indicators.get('daily_vol', 0)
    macd_hist = indicators.get('macd_hist', 0)
    cointegration_resid = indicators.get('cointegration_resid', 0)

    sentiment_score, _ = get_market_sentiment()
    min_adx_adjusted = weights.get('MIN_ADX', 15) * 0.85 if sentiment_score > 70 else weights.get('MIN_ADX', 15)
    min_rel_vol_adjusted = weights.get('MIN_RELATIVE_VOLUME', 0.5) * 0.85 if sentiment_score > 70 else weights.get('MIN_RELATIVE_VOLUME', 0.5)

    # Dynamic volume adjustment
    if data and '1h' in data:
        recent_volatility = data['1h']['close'].pct_change().tail(24).std()
        min_rel_vol_adjusted *= (1 + recent_volatility)

    support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else 0.5

    if adx is None or adx < min_adx_adjusted:
        return "mantener", 50, "Tendencia débil"
    if relative_volume is None or relative_volume < min_rel_vol_adjusted:
        return "mantener", 50, "Volumen relativo bajo"
    if support_distance > weights.get('MAX_SUPPORT_DISTANCE', 0.05):
        return "mantener", 50, "Lejos del soporte"
    if rsi is None or rsi < 10 or rsi > 90:
        return "mantener", 50, "RSI fuera de rango"
    if spread > weights.get('MAX_SPREAD', 0.01):
        return "mantener", 50, "Spread alto"

    volume_spike = False
    if data and '1h' in data:
        df = data['1h']
        if len(df) >= 10:
            avg_volume_10 = df['volume'].rolling(window=10).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_spike = current_volume > avg_volume_10 * weights.get('VOLUME_SPIKE_FACTOR', 1.0)

    oversold = False
    if data and '1h' in data:
        df = data['1h']
        if len(df) >= 7:
            ma7 = df['close'].rolling(window=7).mean().iloc[-1]
            oversold = current_price < ma7 * weights.get('OVERSOLD_THRESHOLD', 0.95)

    score_weights = weights.get('score_weights', {})
    vol_trend_boost = score_weights.get('short_vol_trend', 2) if short_volume_trend == "increasing" else 0
    daily_vol_boost = score_weights.get('daily_vol_bonus', 1) if daily_vol >= weights.get('MIN_DAILY_VOL', 50000) else 0
    ml_prob_threshold = 0.7 if best_winrate > 75 else 0.65

    weighted_signals = [
        score_weights.get('rel_vol_bonus', 3) * (relative_volume > 1.0 if relative_volume else False),
        vol_trend_boost,
        score_weights.get('price_trend', 1) * (price_trend == "increasing"),
        score_weights.get('support_dist', 3) * (support_distance <= weights.get('MAX_SUPPORT_DISTANCE', 0.05)),
        score_weights.get('adx_bonus', 1) * (adx > 30 if adx else False),
        score_weights.get('rsi_penalty', -1) * (rsi > 90 or rsi < 10 if rsi else False),
        score_weights.get('oversold', 2) * oversold,
        score_weights.get('vol_spike', 1) * volume_spike,
        score_weights.get('ml_prob', 3) * (ml_prob > ml_prob_threshold),
        daily_vol_boost
    ]
    signals_score = sum(weighted_signals)

    if abs(cointegration_resid) < 0.01:
        signals_score += 1
    if macd_hist > 0:
        signals_score += 1

    threshold = 2 if weights.get('category') == 'high_vol' else 3
    if signals_score >= threshold:
        action = "comprar"
        confidence = 80 if signals_score < 5 else 90
        explanation = f"Compra fuerte: Volumen={relative_volume:.2f}, ADX={adx:.1f}, ML Prob={ml_prob:.2f}, MACD Hist={macd_hist:.2f}, Cointegration Resid={cointegration_resid:.4f}"
    else:
        action = "mantener"
        confidence = 60
        explanation = f"Insuficiente score: {signals_score}"

    return action, confidence, explanation

# Fetch historical data with pagination
def fetch_ohlcv_paginated(exchange, symbol, timeframe='1h', years=2):
    start_time = time.time()
    limit = 1000
    since = int((datetime.now() - timedelta(days=365*years)).timestamp() * 1000)
    all_ohlcv = []
    max_retries = 5
    retry_delay = 15
    logger.info(f"Starting OHLCV fetch for {symbol}")
    with tqdm(total=None, desc=f"Fetching {symbol} OHLCV", leave=False) as pbar:
        while True:
            for attempt in range(max_retries):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    pbar.update(len(ohlcv))
                    time.sleep(0.1)
                    break
                except ccxt.RateLimitExceeded as e:
                    logger.error(f"Rate limit exceeded for {symbol}: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                except ccxt.NetworkError as e:
                    logger.error(f"Network error for {symbol}: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                except Exception as e:
                    logger.error(f"Unexpected error for {symbol}: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to fetch OHLCV for {symbol} after {max_retries} attempts.")
                        return all_ohlcv
            else:
                break
    logger.info(f"Completed OHLCV fetch for {symbol} in {time.time() - start_time:.2f}s")
    return all_ohlcv

# Simulate backtest
def simulate_backtest(df, grid, temp_weights, ml_probs, symbol):
    start_time = time.time()
    trades = []
    positions = []
    data = {'1h': df}
    num = 0
    with tqdm(total=len(df) - 5, desc=f"Backtesting {symbol}", leave=False) as pbar:
        for i in range(len(df) - 5):
            if df['spread'].iloc[i] > temp_weights['MAX_SPREAD']:
                pbar.update(1)
                continue
            indicators_i = {
                'rsi': df['RSI'].iloc[i],
                'adx': df['ADX'].iloc[i],
                'relative_volume': df['rel_vol'].iloc[i],
                'price_trend': 'increasing' if df['close'].iloc[i] > df['close'].iloc[max(0, i-1)] else 'decreasing',
                'short_volume_trend': 'increasing' if df['volume'].iloc[i] > df['volume'].iloc[max(0, i-1)] else 'decreasing',
                'support_level': df['support_level'].iloc[i],
                'current_price': df['close'].iloc[i],
                'imbalance': df['imbalance'].iloc[i],
                'spread': df['spread'].iloc[i],
                'daily_vol': df['daily_vol'].iloc[i],
                'macd_hist': df['macd_hist'].iloc[i],
                'cointegration_resid': df['cointegration_resid'].iloc[i]
            }
            action, _, _ = calculate_established_strategy(indicators_i, data, symbol, grid, ml_probs[min(i, len(ml_probs)-1)] if len(ml_probs) > 0 else 0.5)
            if action == "comprar":
                pl = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
                pl = min(max(pl, -2), 5)  # SL -2%, TP 5%
                pl -= 0.2  # Fees
                trades.append(pl > 0)
                positions.append(pl)
                num += 1
            pbar.update(1)

    num_trades = len(trades)
    if num_trades == 0:
        logger.info(f"Backtest for {symbol} completed in {time.time() - start_time:.2f}s: No trades")
        return 0, 0, 0, 0

    winrate = sum(trades) / num_trades * 100
    profits = [p for p in positions if p > 0]
    losses = [abs(p) for p in positions if p <= 0]
    profit_factor = sum(profits) / sum(losses) if losses else float('inf')
    equity_curve = np.cumprod(1 + np.array(positions)/100)
    drawdown = (np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
    max_drawdown = drawdown.max() * 100
    logger.info(f"Backtest for {symbol} completed in {time.time() - start_time:.2f}s")

    return winrate, profit_factor, max_drawdown, num_trades

# Objective for Optuna
def objective(trial, df, grid, ml_probs, symbol, trial_data):
    start_time = time.time()
    category = COIN_WEIGHTS.get(symbol.split('/')[0], {}).get('category', 'growth')
    ranges = get_param_ranges(category)
    min_adx = trial.suggest_int('min_adx', ranges['min_adx'][0], ranges['min_adx'][1], step=5)
    min_rel_vol = trial.suggest_float('min_rel_vol', ranges['min_rel_vol'][0], ranges['min_rel_vol'][1], step=0.05)
    max_support = trial.suggest_float('max_support', ranges['max_support'][0], ranges['max_support'][1], step=0.01)
    vol_spike = trial.suggest_float('vol_spike', ranges['vol_spike'][0], ranges['vol_spike'][1], step=0.1)
    max_spread = trial.suggest_float('max_spread', 0.001, 0.05, step=0.005)
    min_daily_vol = trial.suggest_int('min_daily_vol', 5000, 200000, step=5000)

    temp_weights = {
        'MIN_ADX': min_adx,
        'MIN_RELATIVE_VOLUME': min_rel_vol,
        'MAX_SUPPORT_DISTANCE': max_support,
        'VOLUME_SPIKE_FACTOR': vol_spike,
        'OVERSOLD_THRESHOLD': 0.95,
        'MAX_SPREAD': max_spread,
        'MIN_DAILY_VOL': min_daily_vol,
        'score_weights': COIN_WEIGHTS.get(symbol.split('/')[0], {}).get('score_weights', {})
    }

    winrate, profit_factor, max_drawdown, num_trades = simulate_backtest(df, grid, temp_weights, ml_probs, symbol)
    
    # Log trial details
    score = (winrate * 0.6 + (num_trades / len(df)) * 0.3) / (max_drawdown / 100 + 0.1) * (profit_factor if profit_factor < 10 else 10)
    logger.info(f"Trial {trial.number}: Winrate={winrate:.1f}%, Trades={num_trades}, Drawdown={max_drawdown:.1f}%, Factor={profit_factor:.2f}, "
                f"Score={score:.4f}, Params={{min_adx={min_adx}, min_rel_vol={min_rel_vol:.2f}, max_support={max_support:.2f}, vol_spike={vol_spike:.2f}, "
                f"max_spread={max_spread:.3f}, min_daily_vol={min_daily_vol}}}, Time={time.time() - start_time:.2f}s")
    
    # Store trial data for CSV
    trial_data.append({
        'trial_number': trial.number,
        'min_adx': min_adx,
        'min_rel_vol': min_rel_vol,
        'max_support': max_support,
        'vol_spike': vol_spike,
        'max_spread': max_spread,
        'min_daily_vol': min_daily_vol,
        'score': score,
        'winrate': winrate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'num_trades': num_trades,
        'trial_time': time.time() - start_time
    })

    if num_trades == 0:
        return -1

    score *= 0.9 if profit_factor < 1.5 else 1.0
    score *= 0.8 if max_drawdown > 20 else 1.0
    score *= 0.7 if num_trades < 50 else 1.0
    return score

# Backtest function
def backtest_strategy(symbol, eth_df, exchange):
    start_time = time.time()
    logger.info(f"Starting backtest for {symbol}")
    
    # Fetch data
    ohlcv = fetch_ohlcv_paginated(exchange, symbol)
    if not ohlcv:
        logger.error(f"No data fetched for {symbol}. Skipping.")
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    logger.info(f"Fetched {len(df)} rows for {symbol} in {time.time() - start_time:.2f}s")

    # Fetch real-time spread and imbalance
    try:
        ticker = exchange.fetch_ticker(symbol)
        df['spread'] = (ticker['ask'] - ticker['bid']) / ticker['bid']
        order_book = exchange.fetch_order_book(symbol, limit=10)
        bid_vol = sum([order[1] for order in order_book['bids']])
        ask_vol = sum([order[1] for order in order_book['asks']])
        df['imbalance'] = bid_vol / (ask_vol + 1e-10)
        logger.info(f"Fetched spread and imbalance for {symbol} in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.warning(f"Failed to fetch spread/imbalance for {symbol}: {e}. Using defaults.")
        df['spread'] = 0.001
        df['imbalance'] = 1.1

    # Indicators
    with tqdm(total=6, desc=f"Computing indicators for {symbol}", leave=False) as pbar:
        df['RSI'] = ta.rsi(df['close'], length=14)
        pbar.update(1)
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        pbar.update(1)
        df['rel_vol'] = df['volume'] / df['volume'].rolling(10).mean()
        pbar.update(1)
        df['support_level'] = df['low'].rolling(15).min()
        df['support_dist'] = (df['close'] - df['support_level']) / df['support_level']
        pbar.update(1)
        df['vol_trend'] = np.where(df['volume'].pct_change() > 0.01, 1, 0)
        df['daily_vol'] = df['volume'].rolling(24).sum()
        pbar.update(1)
        macd = ta.macd(df['close'])
        df['macd_hist'] = macd['MACDh_12_26_9']
        pbar.update(1)
    logger.info(f"Computed indicators for {symbol} in {time.time() - start_time:.2f}s")

    # Cointegration
    df = df.join(eth_df['close'], rsuffix='_eth')
    df = df.dropna(subset=['close', 'close_eth'])
    y = np.log(df['close'])
    x = np.log(df['close_eth'])
    x = sm.add_constant(x)
    ols = sm.OLS(y, x).fit()
    df['cointegration_resid'] = ols.resid
    logger.info(f"Computed cointegration for {symbol} in {time.time() - start_time:.2f}s")

    # Features and labels
    features = df[['RSI', 'ADX', 'rel_vol', 'support_dist', 'vol_trend', 'imbalance', 'macd_hist', 'cointegration_resid']].shift(1).dropna()
    volatility = df['close'].pct_change().std() * np.sqrt(24)
    labels = (df['close'].shift(-5) / df['close'] > volatility).astype(int)
    labels = labels.loc[features.index].dropna()
    features = features.loc[labels.index]

    if len(features) < 20:
        logger.warning(f"Insufficient data for {symbol}: {len(features)} samples")
        return None

    if len(np.unique(labels)) < 2:
        logger.warning(f"Only one class in labels for {symbol}")
        return None

    # ML model
    ml_start_time = time.time()
    logger.info(f"Starting ML training for {symbol}")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(device='cuda')  # GPU acceleration
    param_grid = {
        'n_estimators': [50],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(model, param_grid, cv=tscv)
    with tqdm(total=len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['min_child_weight']) * len(param_grid['gamma']) * tscv.get_n_splits(), desc=f"Training ML for {symbol}", leave=False) as pbar:
        grid.fit(X_train, y_train)
        pbar.update(len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['min_child_weight']) * len(param_grid['gamma']) * tscv.get_n_splits())

    # ML metrics
    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"ML metrics for {symbol}: Accuracy={accuracy*100:.1f}%, Precision={precision*100:.1f}%, "
                f"Recall={recall*100:.1f}%, F1-Score={f1*100:.1f}%, Confusion Matrix={cm.tolist()}, "
                f"Training Time={time.time() - ml_start_time:.2f}s")
    logger.info(f"Feature Importance: {dict(zip(features.columns, grid.best_estimator_.feature_importances_))}")

    model_path = f"{symbol.replace('/', '_')}_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    joblib.dump(grid, model_path)
    logger.info(f"Saved ML model for {symbol} to {model_path}")

    ml_probs = grid.predict_proba(features)[:,1]

    # Optuna with trial data collection
    trial_data = []
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    best_value = -float('inf')
    best_trial_number = 0
    no_improvement_count = 0
    max_no_improvement = 50
    max_trials = 50  # Reduced for faster runtime

    logger.info(f"Starting Optuna optimization for {symbol}")
    for trial_num in tqdm(range(max_trials), desc=f"Optimizing {symbol}", leave=False):
        trial_start_time = time.time()
        trial = study.ask()
        score = objective(trial, df, grid, ml_probs, symbol, trial_data)
        study.tell(trial, score)
        if score > best_value:
            best_value = score
            best_trial_number = trial.number
            no_improvement_count = 0
            logger.info(f"New best score for {symbol}: {best_value:.4f} at trial {best_trial_number}, "
                        f"Time={time.time() - trial_start_time:.2f}s")
        else:
            no_improvement_count += 1

        # Heartbeat logging every 10 trials
        if (trial_num + 1) % 10 == 0:
            logger.info(f"Heartbeat: Completed {trial_num + 1}/{max_trials} trials for {symbol}, "
                        f"Current Best Score={best_value:.4f}")

        if no_improvement_count >= max_no_improvement:
            logger.info(f"Early stopping for {symbol} after {max_no_improvement} trials with no improvement")
            break

    best_params = study.best_params
    best_weights = {
        'MIN_ADX': best_params['min_adx'],
        'MIN_RELATIVE_VOLUME': best_params['min_rel_vol'],
        'MAX_SUPPORT_DISTANCE': best_params['max_support'],
        'VOLUME_SPIKE_FACTOR': best_params['vol_spike'],
        'OVERSOLD_THRESHOLD': 0.95,
        'MAX_SPREAD': best_params['max_spread'],
        'MIN_DAILY_VOL': best_params['min_daily_vol'],
        'score_weights': COIN_WEIGHTS.get(symbol.split('/')[0], {}).get('score_weights', {})
    }

    # Re-run for final metrics
    final_start_time = time.time()
    best_winrate, best_profit_factor, max_drawdown, num_trades = simulate_backtest(df, grid, best_weights, ml_probs, symbol)
    logger.info(f"Best for {symbol}: Winrate={best_winrate:.1f}%, Factor={best_profit_factor:.2f}, "
                f"Drawdown={max_drawdown:.1f}%, Trades={num_trades}, Trial={best_trial_number}, "
                f"Final Backtest Time={time.time() - final_start_time:.2f}s")

    # Save trial data to CSV
    trial_df = pd.DataFrame(trial_data)
    trial_df['symbol'] = symbol
    trial_df['ml_accuracy'] = accuracy
    trial_df['ml_precision'] = precision
    trial_df['ml_recall'] = recall
    trial_df['ml_f1'] = f1
    trial_df['confusion_matrix'] = [cm.tolist()] * len(trial_df)
    trial_df['best_score'] = best_value
    trial_df['best_trial'] = best_trial_number
    output_path = f"{symbol.replace('/', '_')}_optimization_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    trial_df.to_csv(output_path, index=False)
    logger.info(f"Saved optimization results for {symbol} to {output_path}")

    # Summary statistics
    if not trial_df.empty:
        mean_score = trial_df['score'].mean()
        std_score = trial_df['score'].std()
        logger.info(f"Summary for {symbol}: Mean Score={mean_score:.4f}, Std Score={std_score:.4f}, "
                    f"Trials={len(trial_df)}, Best Score={best_value:.4f} at Trial {best_trial_number}")

    # Save best weights
    cursor.execute('''
        INSERT OR REPLACE INTO optimized_weights (symbol, weights, last_optimized)
        VALUES (?, ?, ?)
    ''', (symbol, json.dumps(best_weights), datetime.now().isoformat()))
    conn.commit()

    if best_value <= 0:
        logger.warning(f"No converging params for {symbol} - falling back to defaults")
        best_weights = {
            'MIN_ADX': 10,
            'MIN_RELATIVE_VOLUME': 0.3,
            'MAX_SUPPORT_DISTANCE': 0.1,
            'VOLUME_SPIKE_FACTOR': 1.0,
            'OVERSOLD_THRESHOLD': 0.95,
            'MAX_SPREAD': 0.01,
            'MIN_DAILY_VOL': 10000,
            'score_weights': COIN_WEIGHTS.get(symbol.split('/')[0], {}).get('score_weights', {})
        }
        cursor.execute('''
            INSERT OR REPLACE INTO optimized_weights (symbol, weights, last_optimized)
            VALUES (?, ?, ?)
        ''', (symbol, json.dumps(best_weights), datetime.now().isoformat()))
        conn.commit()

    total_time = time.time() - start_time
    logger.info(f"Completed backtest for {symbol} in {total_time:.2f}s")
    return best_winrate, total_time

# Main function
def main():
    start_time = time.time()
    logger.info("Starting main execution")
    exchange = ccxt.binance({
        'apiKey': BINANCE_API_KEY,
        'secret': BINANCE_SECRET_KEY,
        'enableRateLimit': True,
        'rateLimit': 100,
    })
    try:
        exchange_info = exchange.load_markets()
        logger.info("Successfully loaded exchange info")
    except Exception as e:
        logger.error(f"Failed to load exchange info: {e}")
        return

    eth_ohlcv = fetch_ohlcv_paginated(exchange, 'ETH/USDT')
    if not eth_ohlcv:
        logger.error("No data fetched for ETH/USDT. Exiting.")
        return
    eth_df = pd.DataFrame(eth_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
    eth_df.set_index('timestamp', inplace=True)
    logger.info(f"Fetched ETH/USDT data ({len(eth_df)} rows) in {time.time() - start_time:.2f}s")

    symbol_times = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(backtest_strategy, symbol, eth_df, exchange) for symbol in SELECTED_CRYPTOS]
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(SELECTED_CRYPTOS), desc="Processing Symbols")):
            result = future.result()
            if result:
                best_winrate, symbol_time = result
                symbol_times.append(symbol_time)
                remaining_symbols = len(SELECTED_CRYPTOS) - (i + 1)
                avg_time = sum(symbol_times) / len(symbol_times) if symbol_times else 0
                logger.info(f"Estimated time remaining for {remaining_symbols} symbols: {remaining_symbols * avg_time:.2f}s")

    total_time = time.time() - start_time
    logger.info(f"Total runtime: {total_time:.2f}s")

    cursor.execute("SELECT symbol, weights, last_optimized FROM optimized_weights ORDER BY last_optimized DESC")
    rows = cursor.fetchall()
    print("Optimized Weights (Latest First):")
    if not rows:
        print("No entries yet—run backtest_strategy to populate.")
    for row in rows:
        symbol, weights_json, timestamp = row
        weights = json.loads(weights_json)
        print(f"\nSymbol: {symbol}")
        print(f"Last Optimized: {timestamp}")
        print("Weights:")
        print(json.dumps(weights, indent=2))

    conn.close()

if __name__ == "__main__":
    main()