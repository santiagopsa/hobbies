import sqlite3
import json
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier  # Better for prob
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

# Backtest function (improved for high prob/volume)
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
    df['imbalance'] = 1.1  # Dummy; in bot, from fetch_order_book_data

    features = df[['RSI', 'ADX', 'rel_vol', 'support_dist', 'vol_trend', 'imbalance']].dropna()
    labels = (df['close'].shift(-1) > df['close']).astype(int).loc[features.index[:-1]]  # Align, drop last

    features = features.iloc[:-1]  # Match labels len

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

    # Simulate trades with full strategy for metrics
    trades = []
    positions = []
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
        action, _, _ = calculate_established_strategy(indicators_i, {'1h': df.iloc[:i+1]}, symbol)
        if action == "comprar":
            pl = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
            trades.append(pl > 0)
            positions.append(pl)

    winrate = sum(trades) / len(trades) * 100 if trades else 0
    profits = [p for p in positions if p > 0]
    losses = [abs(p) for p in positions if p <= 0]
    profit_factor = sum(profits) / sum(losses) if losses else float('inf')
    equity_curve = np.cumprod(1 + np.array(positions)/100)
    drawdown = (np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
    max_drawdown = drawdown.max() * 100

    print(f"Backtest for {symbol}: Winrate {winrate:.1f}%, Profit Factor {profit_factor:.2f}, Max Drawdown {max_drawdown:.1f}%")

    # Tune/save weights if improved (example; customize thresholds)
    optimized_weights = {"MIN_RELATIVE_VOLUME": 0.05 if winrate > 80 else 0.1, "score_weights": {"rel_vol_bonus": 4 if profit_factor > 2 else 3}}
    cursor.execute('''
        INSERT OR REPLACE INTO optimized_weights (symbol, weights, last_optimized)
        VALUES (?, ?, ?)
    ''', (symbol, json.dumps(optimized_weights), datetime.now().isoformat()))
    conn.commit()

    return winrate

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