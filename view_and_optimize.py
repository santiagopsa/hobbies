import sqlite3
import json
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import linregress
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import time
import optuna
import joblib
import concurrent.futures
import logging
import matplotlib.pyplot as plt
import os

DB_NAME = "trading_real.db"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create optimized_weights if missing
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS optimized_weights (
        symbol TEXT PRIMARY KEY,
        weights TEXT NOT NULL,
        last_optimized TEXT NOT NULL
    )
''')
conn.commit()

# Coin-specific weights (full for standalone, added daily_vol_bonus)
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
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
            'daily_vol_bonus': 1
        }
    }
}

# Initial weights from user/forums (loosened for volume: ADX15 for trends, rel_vol0.5 for spikes, support0.05, spike1.0, spread0.01, vol50k per TradingView/Reddit for ~75% win start)
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

# Merge function to DRY weights (fall back to INITIAL_WEIGHTS if key missing in COIN_WEIGHTS)
def get_weights(symbol):
    base_coin = symbol.split('/')[0]
    coin_weight = COIN_WEIGHTS.get(base_coin, {})
    initial_weight = INITIAL_WEIGHTS.get(symbol, {})
    merged = {**initial_weight, **coin_weight}
    return merged

# SELECTED_CRYPTOS
SELECTED_CRYPTOS = list(INITIAL_WEIGHTS.keys())

# Manual RSI (numpy, fixed)
def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period if len(seed[seed >= 0]) > 0 else 0
    down = -seed[seed < 0].sum() / period if len(seed[seed < 0]) > 0 else 0
    rs = up / down if down != 0 else 0
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
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return pd.Series(rsi, index=prices.index)

# Manual ADX (fixed, with +DI/-DI)
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

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx

# Dummy get_market_sentiment for standalone
def get_market_sentiment():
    return 70, "Greed"  # Assume greed for test

# calculate_established_strategy (full, with imbalance/ML prob/MACD/cointegration)
def calculate_established_strategy(indicators, data=None, symbol=None, ml_model=None, ml_prob=0.5):
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
    cointegration_p = indicators.get('cointegration_p', 1.0)  # Default no cointegration

    sentiment_score, _ = get_market_sentiment()
    min_adx_adjusted = weights.get('MIN_ADX', 15) * 0.85 if sentiment_score > 70 else weights.get('MIN_ADX', 15)
    min_rel_vol_adjusted = weights.get('MIN_RELATIVE_VOLUME', 0.5) * 0.85 if sentiment_score > 70 else weights.get('MIN_RELATIVE_VOLUME', 0.5)

    support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else 0.5

    if adx is None or adx < min_adx_adjusted:
        return "mantener", 50, "Tendencia débil"
    if relative_volume is None or relative_volume < min_rel_vol_adjusted:
        return "mantener", 50, "Volumen relativo bajo"
    if support_distance > weights.get('MAX_SUPPORT_DISTANCE', 0.05):
        return "mantener", 50, "Lejos del soporte"
    if rsi is None or rsi < 10 or rsi > 90:  # Loosened further to 10-90 for more trades
        return "mantener", 50, "RSI fuera de rango"
    if spread > weights.get('MAX_SPREAD', 0.01):
        return "mantener", 50, "Spread alto"
    # Moved daily_vol to score boost

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
    daily_vol_boost = score_weights.get('daily_vol_bonus', 1) if daily_vol >= weights.get('MIN_DAILY_VOL', 50000) else 0  # Added boost for daily_vol
    weighted_signals = [
        score_weights.get('rel_vol_bonus', 3) * (relative_volume > 1.0 if relative_volume else False),
        vol_trend_boost,
        score_weights.get('price_trend', 1) * (price_trend == "increasing"),
        score_weights.get('support_dist', 3) * (support_distance <= weights.get('MAX_SUPPORT_DISTANCE', 0.05)),
        score_weights.get('adx_bonus', 1) * (adx > 30 if adx else False),
        score_weights.get('rsi_penalty', -1) * (rsi > 90 or rsi < 10 if rsi else False),
        score_weights.get('oversold', 2) * oversold,
        score_weights.get('vol_spike', 1) * volume_spike,
        daily_vol_boost
    ]
    signals_score = sum(weighted_signals)

    imbalance = indicators.get('imbalance', 1.0)
    if imbalance > 1.1:
        signals_score += 1

    # Integrate ML prob + MACD hist + cointegration
    if ml_prob > 0.7:
        signals_score += 2
    if macd_hist > 0:
        signals_score += 1
    if cointegration_p < 0.05:
        signals_score += 1  # Pairs edge

    if signals_score >= 3:  # Lowered to 3 for more trades
        action = "comprar"
        confidence = 80 if signals_score < 5 else 90  # Adjusted for more entries
        explanation = f"Compra fuerte: Volumen={relative_volume}, ADX={adx}, Imbalance={imbalance}, ML Prob={ml_prob}, MACD Hist={macd_hist}, Cointegration p={cointegration_p}"
    else:
        action = "mantener"
        confidence = 60
        explanation = "Insuficiente score"

    return action, confidence, explanation

# Fetch historical data with pagination for 5y+ (with retry for errors)
def fetch_ohlcv_paginated(exchange, symbol, timeframe='1h', years=5):
    limit = 1000  # Max per call
    since = int((datetime.now() - timedelta(days=365*years)).timestamp() * 1000)
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Next start
            time.sleep(1)  # Rate limit
        except Exception as e:
            logger.error(f"API error for {symbol}: {e}. Retrying in 10s...")
            time.sleep(10)  # Retry delay
    return all_ohlcv

# Simulate backtest (with SL/TP for realistic drawdown, fees 0.1%)
def simulate_backtest(df, grid, temp_weights, ml_probs, symbol):
    trades = []
    positions = []
    data = {'1h': df}
    num = 0
    for i in range(len(df) - 5):
        if df['spread'].iloc[i] > temp_weights['MAX_SPREAD']:
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
            'cointegration_p': df['cointegration_p'].iloc[i]
        }
        action, _, _ = calculate_established_strategy(indicators_i, data, symbol, grid, ml_probs[min(i, len(ml_probs)-1)] if len(ml_probs) > 0 else 0.5)
        if action == "comprar":
            pl = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
            pl = min(max(pl, -2), 5)  # SL -2%, TP 5% for <15% DD
            pl -= 0.2  # 0.1% entry + 0.1% exit fees for realism
            trades.append(pl > 0)
            positions.append(pl)
            num += 1

    num_trades = len(trades)
    if num_trades == 0:
        return 0, 0, 0, 0

    winrate = sum(trades) / num_trades * 100
    profits = [p for p in positions if p > 0]
    losses = [abs(p) for p in positions if p <= 0]
    profit_factor = sum(profits) / sum(losses) if losses else float('inf')
    equity_curve = np.cumprod(1 + np.array(positions)/100)
    drawdown = (np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
    max_drawdown = drawdown.max() * 100

    # Plot equity curve for visualization (optional, comment if not needed)
    # plt.figure(figsize=(10, 5))
    # plt.plot(equity_curve)
    # plt.title(f"Equity Curve for {symbol}")
    # plt.savefig(f"{symbol.replace('/', '_')}_equity_curve.png")
    # plt.close()

    return winrate, profit_factor, max_drawdown, num_trades

# Objective for Optuna (maximize win * trades / drawdown for balance)
def objective(trial, df, grid, ml_probs, symbol):
    min_adx = trial.suggest_int('min_adx', 5, 30, step=5)
    min_rel_vol = trial.suggest_float('min_rel_vol', 0.1, 1.0, step=0.1)
    max_support = trial.suggest_float('max_support', 0.05, 0.2, step=0.05)
    vol_spike = trial.suggest_float('vol_spike', 1.0, 1.4, step=0.2)
    max_spread = trial.suggest_float('max_spread', 0.001, 0.1, step=0.02)
    min_daily_vol = trial.suggest_int('min_daily_vol', 10000, 100000, step=10000)

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
    if num_trades == 0:
        return 0

    # Metric for highest returns: high win * volume, low drawdown, factor>2 check
    score = winrate * (num_trades / len(df)) / (max_drawdown + 1) * profit_factor
    return score if profit_factor > 2 and max_drawdown < 15 and winrate > 80 and num_trades > 100 else 0  # Enforce goals

# Backtest function (Optuna for efficiency/convergence)
def backtest_strategy(symbol, eth_df):
    exchange = ccxt.binance()
    ohlcv = fetch_ohlcv_paginated(exchange, symbol)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Vectorized indicators (no loops, but shift to prevent lookahead)
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    df['rel_vol'] = df['volume'] / df['volume'].rolling(10).mean()
    df['support_level'] = df['low'].rolling(15).min()
    df['support_dist'] = (df['close'] - df['support_level']) / df['support_level']
    df['vol_trend'] = np.where(df['volume'].pct_change() > 0.01, 1, 0)
    df['imbalance'] = 1.1  # Dummy - for real, estimate from order book if available
    df['spread'] = 0.001  # Dummy - in live, fetch from exchange.ticker()
    df['daily_vol'] = df['volume'].rolling(24).sum()
    macd = ta.macd(df['close'])
    df['macd_hist'] = macd['MACDh_12_26_9']

    # Cointegration
    df = df.join(eth_df['close'], rsuffix='_eth')
    df = df.dropna(subset=['close', 'close_eth'])
    y = np.log(df['close'])
    x = np.log(df['close_eth'])
    x = sm.add_constant(x)
    ols = sm.OLS(y, x).fit()
    resid = ols.resid
    df['cointegration_p'] = ols.pvalues.iloc[1]
    df['cointegration_resid'] = resid

    features = df[['RSI', 'ADX', 'rel_vol', 'support_dist', 'vol_trend', 'imbalance', 'macd_hist', 'cointegration_resid']].dropna()
    labels = (df['close'].shift(-5) / df['close'] > 0.01).astype(int)  # Improved label
    labels = labels.loc[features.index].dropna()
    features = features.loc[labels.index]

    if len(features) < 20:
        print(f"Insufficient data for {symbol}: {len(features)} samples")
        return None

    if len(features) != len(labels):
        logger.error(f"Misalignment in features ({len(features)}) and labels ({len(labels)}) for {symbol}")
        return None

    # Check label diversity to avoid all 0 or 1
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"Labels all same ({unique_labels}) for {symbol}, skipping ML")
        return None

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    model = xgb.XGBClassifier()
    param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    unique_y_train = len(np.unique(y_train))
    if unique_y_train < 2:
        print(f"Only one class in y_train for {symbol}, using default XGB")
        grid = model
        grid.fit(X_train, y_train)
    else:
        tscv = TimeSeriesSplit(n_splits=max(2, unique_y_train - 1))
        grid = GridSearchCV(model, param_grid, cv=tscv)
        grid.fit(X_train, y_train)

    acc = accuracy_score(y_test, grid.predict(X_test))
    print(f"ML accuracy for {symbol}: {acc*100:.1f}%")

    # Save model with timestamp to avoid overwrites
    model_path = f"{symbol.replace('/', '_')}_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    joblib.dump(grid, model_path)

    ml_probs = grid.predict_proba(features)[:,1]

    # Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, df, grid, ml_probs, symbol), n_trials=100)

    best_params = study.best_params
    best_value = study.best_value

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

    # Re-run for metrics
    best_winrate, best_profit_factor, best_max_drawdown, best_num_trades = simulate_backtest(df, grid, best_weights, ml_probs, symbol)

    print(f"Best for {symbol}: Winrate {best_winrate:.1f}%, Factor {best_profit_factor:.2f}, Drawdown {best_max_drawdown:.1f}%, Trades {best_num_trades}")

    # Fallback if no good
    if best_value == 0:
        print(f"No converging params for {symbol} - falling back to loose defaults")
        best_weights = {
            'MIN_ADX': 10,
            'MIN_RELATIVE_VOLUME': 0.3,
            'MAX_SUPPORT_DISTANCE': 0.1,
            'VOLUME_SPIKE_FACTOR': 1.0,
            'OVERSOLD_THRESHOLD': 0.95,
            'MAX_SPREAD': 0.01,
            'MIN_DAILY_VOL': 10000
        }

    # Save
    cursor.execute('''
        INSERT OR REPLACE INTO optimized_weights (symbol, weights, last_optimized)
        VALUES (?, ?, ?)
    ''', (symbol, json.dumps(best_weights), datetime.now().isoformat()))
    conn.commit()

    return best_winrate

# Run for all symbols (parallel)
def main():
    exchange = ccxt.binance()
    eth_ohlcv = fetch_ohlcv_paginated(exchange, 'ETH/USDT')
    eth_df = pd.DataFrame(eth_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
    eth_df.set_index('timestamp', inplace=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(backtest_strategy, symbol, eth_df) for symbol in SELECTED_CRYPTOS]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # View contents
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

main()