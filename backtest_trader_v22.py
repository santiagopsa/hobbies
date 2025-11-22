"""
Backtesting program for trader_v22.py
Simulates buys and exits with trailing stop using historical data

NOTE: This backtest uses a simplified version of the buy logic from hybrid_decision()
since the original function requires live exchange API calls. The trailing stop logic
is more closely aligned with the original dynamic_trailing_stop() function.

For best results:
1. Ensure you have API keys set in .env (for data fetching)
2. Adjust BACKTEST_START_DATE and BACKTEST_END_DATE as needed
3. The backtest will fetch historical data and simulate trades

Usage:
    python backtest_trader_v22.py --symbols BTC/USDT ETH/USDT --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations
import os
import sys

# Create necessary directories before importing trader_v22
# (trader_v22 tries to create log files on import)
log_dir = os.path.expanduser("~/hobbies")
os.makedirs(log_dir, exist_ok=True)

# Import backtest_trader_v21 and modify to use trader_v22
# This avoids code duplication
import backtest_trader_v21

# Replace trader_v21 with trader_v22
import trader_v22 as trader
backtest_trader_v21.trader = trader

# Update references to use v22
BACKTEST_TIMEFRAME = trader.DECISION_TIMEFRAME  # Will use v22's timeframe
BACKTEST_START_DATE = backtest_trader_v21.BACKTEST_START_DATE
BACKTEST_END_DATE = backtest_trader_v21.BACKTEST_END_DATE
BACKTEST_SYMBOLS = trader.SELECTED_CRYPTOS

# Re-export everything from backtest_trader_v21 so this module works as a drop-in replacement
from backtest_trader_v21 import *

# Override description and any v21-specific references
__doc__ = __doc__.replace('trader_v21', 'trader_v22')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest trader_v22 strategy')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to backtest (e.g., BTC/USDT ETH/USDT)')
    parser.add_argument('--start', type=str, default=BACKTEST_START_DATE, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=BACKTEST_END_DATE, help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=backtest_trader_v21.INITIAL_BALANCE, help='Initial balance')
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
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
