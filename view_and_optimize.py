import sqlite3
import json
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier  # For higher prob
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

DB_NAME = "trading_real.db"

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

# Coin-specific weights (full dict for standalone)
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
            'vol_spike': 2
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
            'short_vol_trend': 2 + 3,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1
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
            'short_vol_trend': 2 + 3,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1
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
            'short_vol_trend': 2 + 3,
            'price_trend': 1,
            'support_dist': 3,
            'adx_bonus': 2,
            'rsi_penalty': -2,
            'oversold': 3,
            'vol_spike': 2
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
            'short_vol_trend': 2 + 3,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1
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
            'short_vol_trend': 3 + 3,
            'price_trend': 1,
            'support_dist': 2,
            'adx_bonus': 2,
            'rsi_penalty': -2,
            'oversold': 3,
            'vol_spike': 3
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
            'short_vol_trend': 2 + 3,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1
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
            'short_vol_trend': 2 + 3,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1
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
            'short_vol_trend': 2 + 3,
            'price_trend': 2,
            'support_dist': 3,
            'adx_bonus': 1,
            'rsi_penalty': -1,
            'oversold': 2,
            'vol_spike': 1
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
            'short_vol_trend': 2 + 3,
            'price_trend': 1.5,
            'support_dist': 3,
            'adx_bonus': 2,
            'rsi_penalty': -1.5,
            'oversold': 2.5,
            'vol_spike': 2
        }
    }
}

# Initial weights from user/forums (high winrate start: ADX>30 for strong trends, rel_vol>1.2 for spikes, support<0.02 for rebounds)
INITIAL_WEIGHTS = {
    "BTC/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.5
    },
    "ETH/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.3
    },
    "BNB/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.2
    },
    "SOL/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.5
    },
    "XRP/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.4
    },
    "DOGE/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.6
    },
    "TON/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.3
    },
    "ADA/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.3
    },
    "TRX/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.25
    },
    "AVAX/USDT": {
      "MIN_ADX": 30,
      "MIN_RELATIVE_VOLUME": 1.2,
      "MAX_SUPPORT_DISTANCE": 0.02,
      "VOLUME_SPIKE_FACTOR": 1.45
    }
}

# Dummy get_market_sentiment for standalone
def get_market_sentiment():
    return 70, "Greed"  # Assume greed for test

# calculate_established_strategy (full, with imbalance for prob)
def calculate_established_strategy(indicators, data=None, symbol=None):
    base_coin = symbol.split('/')[0]
    weights = INITIAL_WEIGHTS.get(symbol, COIN_WEIGHTS.get(base_coin, INITIAL_WEIGHTS.get(symbol, {})))  # Prioritize initial/user

    rsi = indicators.get('rsi', None)
    relative_volume = indicators.get('relative_volume', None)
    price_trend = indicators.get('price_trend', 'insufficient_data')
    short_volume_trend = indicators.get('short_volume_trend', 'insufficient_data')
    current_price = indicators.get('current_price', 0)
    support_level = indicators.get('support_level', None)
    adx = indicators.get('adx', None)

    sentiment_score, _ = get_market_sentiment()
    min_adx_adjusted = weights.get('MIN_ADX', 30) * 0.85 if sentiment_score > 70 else weights.get('MIN_ADX', 30)
    min_rel_vol_adjusted = weights.get('MIN_RELATIVE_VOLUME', 1.2) * 0.85 if sentiment_score > 70 else weights.get('MIN_RELATIVE_VOLUME', 1.2)

    support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else 0.5

    if adx is None or adx < min_adx_adjusted:
        return "mantener", 50, "Tendencia débil"
    if relative_volume is None or relative_volume < min_rel_vol_adjusted:
        return "mantener", 50, "Volumen relativo bajo"
    if short_volume_trend != "increasing":
        return "mantener", 50, "Volumen no favorable"
    if support_distance > weights.get('MAX_SUPPORT_DISTANCE', 0.02):
        return "mantener", 50, "Lejos del soporte"
    if rsi is None or rsi < 30:
        return "mantener", 50, "RSI bajo"

    volume_spike = False
    if data and '1h' in data:
        df = data['1h']
        if len(df) >= 10:
            avg_volume_10 = df['volume'].rolling(window=10).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_spike = current_volume > avg_volume_10 * weights.get('VOLUME_SPIKE_FACTOR', 1.5)

    oversold = False
    if data and '1h' in data:
        df = data['1h']
        if len(df) >= 7:
            ma7 = df['close'].rolling(window=7).mean().iloc[-1]
            oversold = current_price < ma7 * weights.get('OVERSOLD_THRESHOLD', 0.95)

    score_weights = weights.get('score_weights', COIN_WEIGHTS.get(base_coin, {}).get('score_weights', {}))
    vol_trend_boost = 3 if short_volume_trend == "increasing" else 0
    weighted_signals = [
        score_weights.get('rel_vol_bonus', 3) * (relative_volume > 1.0 if relative_volume else False),
        score_weights.get('short_vol_trend', 2) * (short_volume_trend == "increasing") + vol_trend_boost,
        score_weights.get('price_trend', 1) * (price_trend == "increasing"),
        score_weights.get('support_dist', 3) * (support_distance <= weights.get('MAX_SUPPORT_DISTANCE', 0.02)),
        score_weights.get('adx_bonus', 1) * (adx > 30 if adx else False),
        score_weights.get('rsi_penalty', -1) * (rsi > 70 if rsi else False),
        score_weights.get('oversold', 2) * oversold,
        score_weights.get('vol_spike', 1) * volume_spike
    ]
    signals_score = sum(weighted_signals)

    imbalance = indicators.get('imbalance', 1.0)
    if imbalance > 1.1:
        signals_score += 1  # Boost for demand, high prob

    if signals_score >= 5:
        action = "comprar"
        confidence = 80 if signals_score < 7 else 90
        explanation = f"Compra fuerte: Volumen={relative_volume}, ADX={adx}, Imbalance={imbalance}"
    else:
        action = "mantener"
        confidence = 60
        explanation = "Insuficiente score"

    return action, confidence, explanation

# Backtest function (iterative optimization loop)
def backtest_strategy(symbol, historical_data=None):
    if historical_data is None:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
    else:
        df = historical_data

    df['RSI'] = ta.rsi(df['close'])
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    df['rel_vol'] = df['volume'] / df['volume'].rolling(10).mean()
    df['support_level'] = df['close'].rolling(15).min()
    df['support_dist'] = (df['close'] - df['support_level']) / df['support_level']
    df['vol_trend'] = np.where(df['volume'].pct_change() > 0.01, 1, 0)  # Add for prob
    df['imbalance'] = 1.1  # Dummy; integrate in bot

    features = df[['RSI', 'ADX', 'rel_vol', 'support_dist', 'vol_trend', 'imbalance']].dropna()
    labels = (df['close'].shift(-1) > df['close']).astype(int)
    labels = labels.loc[features.index].dropna()

    features = features.loc[labels.index]

    if len(features) < 20:
        print(f"Insufficient data for {symbol}: {len(features)} samples")
        return None

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    model = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)

    acc = accuracy_score(y_test, grid.predict(X_test))
    print(f"ML accuracy for {symbol}: {acc*100:.1f}%")

    # Iterative optimization: Start with initial, test variations, save best (high winrate + max trades for volume)
    best_winrate = 0
    best_profit_factor = 0
    best_max_drawdown = float('inf')
    best_num_trades = 0
    best_weights = INITIAL_WEIGHTS.get(symbol, {})

    # Ranges to iterate (from forums: rel_vol 1.0-1.5, ADX 25-35, etc for high win)
    rel_vol_ranges = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    adx_ranges = [25, 30, 35]
    support_dist_ranges = [0.015, 0.02, 0.025]
    vol_spike_ranges = [1.2, 1.3, 1.4, 1.5, 1.6]

    for min_rel_vol in rel_vol_ranges:
        for min_adx in adx_ranges:
            for max_support in support_dist_ranges:
                for vol_spike in vol_spike_ranges:
                    temp_weights = {
                        'MIN_ADX': min_adx,
                        'MIN_RELATIVE_VOLUME': min_rel_vol,
                        'MAX_SUPPORT_DISTANCE': max_support,
                        'VOLUME_SPIKE_FACTOR': vol_spike,
                        'OVERSOLD_THRESHOLD': 0.95,  # Fixed from forums
                        'score_weights': best_weights.get('score_weights', {})
                    }

                    # Simulate with temp_weights
                    trades = []
                    positions = []
                    data = {'1h': df}
                    for i in range(len(df) - 5):
                        indicators_i = {
                            'rsi': df['RSI'].iloc[i],
                            'adx': df['ADX'].iloc[i],
                            'relative_volume': df['rel_vol'].iloc[i],
                            'price_trend': 'increasing' if df['close'].iloc[i] > df['close'].iloc[max(0, i-1)] else 'decreasing',
                            'short_volume_trend': 'increasing' if df['volume'].iloc[i] > df['volume'].iloc[max(0, i-1)] else 'decreasing',
                            'support_level': df['support_level'].iloc[i],
                            'current_price': df['close'].iloc[i],
                            'imbalance': df['imbalance'].iloc[i]
                        }
                        action, _, _ = calculate_established_strategy(indicators_i, data, symbol)
                        if action == "comprar":
                            pl = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
                            trades.append(pl > 0)
                            positions.append(pl)

                    num_trades = len(trades)
                    if num_trades == 0:
                        continue

                    winrate = sum(trades) / num_trades * 100
                    profits = [p for p in positions if p > 0]
                    losses = [abs(p) for p in positions if p <= 0]
                    profit_factor = sum(profits) / sum(losses) if losses else float('inf')
                    equity_curve = np.cumprod(1 + np.array(positions)/100)
                    drawdown = (np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
                    max_drawdown = drawdown.max() * 100

                    # Save if better (high winrate + max trades for volume, factor>2, drawdown<15)
                    if winrate > best_winrate and winrate > 80 and profit_factor > 2 and max_drawdown < 15 and num_trades > best_num_trades:
                        best_winrate = winrate
                        best_profit_factor = profit_factor
                        best_max_drawdown = max_drawdown
                        best_num_trades = num_trades
                        best_weights = temp_weights
                        print(f"Improved iteration for {symbol}: Winrate {winrate:.1f}%, Factor {profit_factor:.2f}, Drawdown {max_drawdown:.1f}%, Trades {num_trades}")

    print(f"Best for {symbol}: Winrate {best_winrate:.1f}%, Factor {best_profit_factor:.2f}, Drawdown {best_max_drawdown:.1f}%, Trades {best_num_trades}")

    # Save best
    cursor.execute('''
        INSERT OR REPLACE INTO optimized_weights (symbol, weights, last_optimized)
        VALUES (?, ?, ?)
    ''', (symbol, json.dumps(best_weights), datetime.now().isoformat()))
    conn.commit()

    return best_winrate

# Run backtest to populate
backtest_strategy('BTC/USDT')

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