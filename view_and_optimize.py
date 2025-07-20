import sqlite3
import json
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

DB_NAME = "trading_real.db"

# Create optimized_weights if missing
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS optimized_weights (
        symbol TEXT PRIMARY KEY,
        weights TEXT NOT NULL,  # JSON dict
        last_optimized TEXT NOT NULL
    )
''')
conn.commit()

# Backtest function (integrated for tuning)
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

    features = df[['RSI', 'ADX', 'rel_vol', 'support_dist']].dropna()
    labels = (df['close'].shift(-1) > df['close']).astype(int).dropna()

    if len(features) < 20:
        print(f"Insufficient data for {symbol}")
        return None

    X_train, X_test, y_train, y_test = train_test_split(features[:-1], labels[:-1], test_size=0.2)
    model = LogisticRegression()
    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)

    acc = accuracy_score(y_test, grid.predict(X_test))
    print(f"ML accuracy for {symbol}: {acc*100:.1f}%")

    # Simulate trades for metrics (example; expand as needed)
    trades = []
    for i in range(len(df) - 5):
        if df['rel_vol'].iloc[i] > 1.1:  # Simple signal for demo
            pl = (df['close'].iloc[i+5] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
            trades.append(pl > 0)

    winrate = sum(trades) / len(trades) * 100 if trades else 0
    print(f"Backtest winrate for {symbol}: {winrate:.1f}%")

    # Save optimized (dummy example; replace with real tuned dict)
    optimized_weights = {"MIN_RELATIVE_VOLUME": 0.05, "score_weights": {"rel_vol_bonus": 4}}
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
for row in rows:
    symbol, weights_json, timestamp = row
    weights = json.loads(weights_json)
    print(f"\nSymbol: {symbol}")
    print(f"Last Optimized: {timestamp}")
    print("Weights:")
    print(json.dumps(weights, indent=2))

conn.close()