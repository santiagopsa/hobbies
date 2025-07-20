import sqlite3
import json
from datetime import datetime

db_name = 'trading_real.db'
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Query and view
cursor.execute("SELECT symbol, weights, last_optimized FROM optimized_weights ORDER BY last_optimized DESC")
rows = cursor.fetchall()

print("Optimized Weights Database Contents (Latest First):")
if not rows:
    print("No entries yet—run backtest_strategy to populate.")
for row in rows:
    symbol, weights_json, timestamp = row
    weights = json.loads(weights_json)
    print(f"\nSymbol: {symbol}")
    print(f"Last Optimized: {timestamp}")
    print("Weights/Variables:")
    print(json.dumps(weights, indent=2))

conn.close()