"""
Parameter optimization script for backtest_trader_v21.py
Finds the best combination of parameters to maximize PnL
"""

import os
import sys
import json
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import backtest_trader_v21 as backtest

# =========================
# Optimization Configuration
# =========================

# Parameters to optimize with their ranges
OPTIMIZATION_PARAMS = {
    # Entry parameters
    'RISK_FRACTION': (0.10, 0.30),  # Position sizing (10% to 30%)
    'ADX_MIN': (18, 28),            # Minimum ADX for entry
    'RSI_MIN': (45, 55),            # Minimum RSI for entry
    'RSI_MAX': (65, 75),            # Maximum RSI for entry
    'RVOL_BASE': (1.2, 2.0),        # Base relative volume threshold
    'SCORE_GATE': (4.5, 6.0),       # Score gate threshold
    
    # Exit/Trailing parameters
    'CHAN_K_MEDIUM': (2.0, 3.5),   # Chandelier K for medium volatility
    'INIT_STOP_ATR_MULT': (1.0, 2.0),  # Initial stop ATR multiplier
    'ARM_TRAIL_PCT': (0.5, 1.5),   # Arm trailing after % gain
    'BE_R_MULT': (1.0, 2.0),       # Break-even at X R
}

# Fixed parameters (not optimized)
FIXED_PARAMS = {
    'INITIAL_BALANCE': 1000.0,
    'MAX_OPEN_TRADES': 5,
    'RESERVE_USDT': 20.0,
    'TRANSACTION_FEE': 0.001,
}

# Optimization method: 'grid', 'random', 'differential_evolution', or 'bayesian'
OPTIMIZATION_METHOD = os.getenv('OPT_METHOD', 'differential_evolution')
MAX_ITERATIONS = int(os.getenv('MAX_ITERATIONS', '50'))  # For random/evolution methods
GRID_RESOLUTION = int(os.getenv('GRID_RESOLUTION', '3'))  # Points per dimension for grid search


class ParameterOptimizer:
    """Optimizes backtest parameters"""
    
    def __init__(self, symbols: List[str] = None, start_date: str = None, end_date: str = None):
        self.symbols = symbols or backtest.BACKTEST_SYMBOLS[:3]  # Limit to 3 for faster optimization
        self.start_date = start_date or backtest.BACKTEST_START_DATE
        self.end_date = end_date or backtest.BACKTEST_END_DATE
        self.best_params = None
        self.best_result = None
        self.results_history = []
    
    def apply_params(self, params: Dict) -> None:
        """Apply parameters to backtest module"""
        # Update backtest module parameters
        backtest.RISK_FRACTION = params.get('RISK_FRACTION', backtest.RISK_FRACTION)
        
        # Update trader module parameters (these affect buy/exit decisions)
        import trader_v21 as trader
        
        # Entry parameters
        if 'ADX_MIN' in params:
            trader.ADX_MIN_DEFAULT = params['ADX_MIN']
        if 'RSI_MIN' in params:
            trader.RSI_MIN_DEFAULT = params['RSI_MIN']
        if 'RSI_MAX' in params:
            trader.RSI_MAX_DEFAULT = params['RSI_MAX']
        if 'RVOL_BASE' in params:
            trader.RVOL_BASE_DEFAULT = params['RVOL_BASE']
        if 'SCORE_GATE' in params:
            trader.set_score_gate(params['SCORE_GATE'])
        
        # Exit parameters
        if 'CHAN_K_MEDIUM' in params:
            trader.CHAN_K_MEDIUM = params['CHAN_K_MEDIUM']
        if 'INIT_STOP_ATR_MULT' in params:
            trader.INIT_STOP_ATR_MULT = params['INIT_STOP_ATR_MULT']
        if 'ARM_TRAIL_PCT' in params:
            trader.ARM_TRAIL_PCT = params['ARM_TRAIL_PCT']
        if 'BE_R_MULT' in params:
            trader.BE_R_MULT = params['BE_R_MULT']
    
    def evaluate_params(self, params: Dict) -> float:
        """Run backtest with given parameters and return negative total return (for minimization)"""
        try:
            # Apply parameters
            self.apply_params(params)
            
            # Run backtest
            engine = backtest.run_backtest(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_balance=FIXED_PARAMS['INITIAL_BALANCE']
            )
            
            # Get statistics
            stats = engine.get_statistics()
            
            if not stats:
                return 1e6  # Very bad score if no trades
            
            # Objective: maximize total return (so minimize negative return)
            total_return = stats.get('total_return_pct', -100.0)
            
            # Penalize for very low win rate or high drawdown
            win_rate = stats.get('win_rate_pct', 0.0)
            max_dd = stats.get('max_drawdown_pct', 100.0)
            
            # Composite score: prioritize return, but penalize low win rate and high drawdown
            score = -total_return  # Negative because we're minimizing
            
            # Penalties
            if win_rate < 30:
                score += 50  # Penalty for very low win rate
            if max_dd > 50:
                score += 30  # Penalty for very high drawdown
            if stats.get('total_trades', 0) < 10:
                score += 20  # Penalty for too few trades
            
            # Store result
            result = {
                'params': params.copy(),
                'stats': stats,
                'score': -score,  # Positive score for reporting
                'total_return': total_return,
                'win_rate': win_rate,
                'max_drawdown': max_dd,
                'total_trades': stats.get('total_trades', 0)
            }
            self.results_history.append(result)
            
            print(f"  Params: {params} | Return: {total_return:.2f}% | Win Rate: {win_rate:.1f}% | Trades: {stats.get('total_trades', 0)}")
            
            return score
            
        except Exception as e:
            print(f"  Error evaluating params {params}: {e}")
            return 1e6  # Very bad score on error
    
    def optimize_grid_search(self, use_parallel: bool = True) -> Dict:
        """Grid search optimization with optional parallel processing"""
        print("Starting Grid Search Optimization...")
        print(f"Testing {len(OPTIMIZATION_PARAMS)} parameters with {GRID_RESOLUTION} points each")
        
        # Create parameter grids
        param_names = list(OPTIMIZATION_PARAMS.keys())
        param_ranges = [OPTIMIZATION_PARAMS[name] for name in param_names]
        
        # Generate grid points
        grid_points = []
        for ranges in param_ranges:
            grid_points.append(np.linspace(ranges[0], ranges[1], GRID_RESOLUTION))
        
        # Generate all combinations
        all_combinations = list(itertools.product(*grid_points))
        param_combinations = [dict(zip(param_names, combo)) for combo in all_combinations]
        
        best_score = float('inf')
        best_params = None
        total_combinations = len(param_combinations)
        
        print(f"Total combinations to test: {total_combinations}")
        print("This may take a while...\n")
        
        if use_parallel and total_combinations > 1:
            # Use multiprocessing for grid search
            num_workers = min(cpu_count(), total_combinations, 8)  # Limit to 8 workers
            print(f"Using {num_workers} parallel workers...")
            
            args_list = [(params, self.symbols, self.start_date, self.end_date) 
                        for params in param_combinations]
            
            with Pool(processes=num_workers) as pool:
                scores = pool.map(self._evaluate_params_wrapper, args_list)
            
            # Find best result
            for i, (params, score) in enumerate(zip(param_combinations, scores)):
                if (i + 1) % 10 == 0 or score < best_score:
                    print(f"[{i+1}/{total_combinations}] Result: Score={-score:.2f}%")
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"  *** NEW BEST: Score={-score:.2f}% ***")
        else:
            # Sequential processing
            for count, params in enumerate(param_combinations, 1):
                print(f"[{count}/{total_combinations}] Testing combination...")
                score = self.evaluate_params(params)
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"  *** NEW BEST: Score={-score:.2f}% ***")
        
        return best_params
    
    def _evaluate_params_wrapper(self, args):
        """Wrapper for multiprocessing"""
        params, symbols, start_date, end_date = args
        try:
            # Create a temporary optimizer instance for this evaluation
            temp_optimizer = ParameterOptimizer(symbols, start_date, end_date)
            return temp_optimizer.evaluate_params(params)
        except Exception as e:
            print(f"  Error in parallel evaluation: {e}")
            return 1e6
    
    def optimize_random_search(self, n_iterations: int = MAX_ITERATIONS, use_parallel: bool = True) -> Dict:
        """Random search optimization with optional parallel processing"""
        print(f"Starting Random Search Optimization ({n_iterations} iterations)...")
        
        # Generate all parameter combinations first
        param_combinations = []
        for i in range(n_iterations):
            params = {}
            for name, (min_val, max_val) in OPTIMIZATION_PARAMS.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[name] = np.random.uniform(min_val, max_val)
            param_combinations.append(params)
        
        best_score = float('inf')
        best_params = None
        
        if use_parallel and len(param_combinations) > 1:
            # Use multiprocessing
            num_workers = min(cpu_count(), len(param_combinations), 8)  # Limit to 8 workers
            print(f"Using {num_workers} parallel workers...")
            
            args_list = [(params, self.symbols, self.start_date, self.end_date) 
                        for params in param_combinations]
            
            with Pool(processes=num_workers) as pool:
                scores = pool.map(self._evaluate_params_wrapper, args_list)
            
            # Find best result
            for i, (params, score) in enumerate(zip(param_combinations, scores)):
                print(f"[{i+1}/{n_iterations}] Result: Score={-score:.2f}%")
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"  *** NEW BEST: Score={-score:.2f}% ***")
        else:
            # Sequential processing
            for i, params in enumerate(param_combinations):
                print(f"[{i+1}/{n_iterations}] Testing random combination...")
                score = self.evaluate_params(params)
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"  *** NEW BEST: Score={-score:.2f}% ***")
        
        return best_params
    
    def optimize_differential_evolution(self, max_iterations: int = None) -> Dict:
        """Differential evolution optimization (scipy)"""
        print("Starting Differential Evolution Optimization...")
        
        if max_iterations is None:
            max_iterations = MAX_ITERATIONS
        
        param_names = list(OPTIMIZATION_PARAMS.keys())
        bounds = [OPTIMIZATION_PARAMS[name] for name in param_names]
        
        def objective(x):
            params = dict(zip(param_names, x))
            return self.evaluate_params(params)
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            popsize=15,
            seed=42,
            polish=True
        )
        
        best_params = dict(zip(param_names, result.x))
        return best_params
    
    def optimize(self, method: str = None) -> Dict:
        """Run optimization with specified method"""
        method = method or OPTIMIZATION_METHOD
        
        print("=" * 80)
        print("PARAMETER OPTIMIZATION FOR BACKTEST")
        print("=" * 80)
        print(f"Symbols: {self.symbols}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Method: {method}")
        print(f"Parameters to optimize: {list(OPTIMIZATION_PARAMS.keys())}")
        print("=" * 80)
        print()
        
        if method == 'grid':
            best_params = self.optimize_grid_search(use_parallel=True)
        elif method == 'random':
            best_params = self.optimize_random_search(use_parallel=True)
        elif method == 'differential_evolution':
            best_params = self.optimize_differential_evolution()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Final evaluation with best params
        print("\n" + "=" * 80)
        print("FINAL EVALUATION WITH BEST PARAMETERS")
        print("=" * 80)
        self.apply_params(best_params)
        engine = backtest.run_backtest(
            symbols=self.symbols,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_balance=FIXED_PARAMS['INITIAL_BALANCE']
        )
        final_stats = engine.get_statistics()
        
        self.best_params = best_params
        self.best_result = final_stats
        
        return best_params
    
    def save_results(self, filename: str = "optimization_results.json"):
        """Save optimization results to file"""
        results = {
            'best_parameters': self.best_params,
            'best_statistics': self.best_result,
            'optimization_history': self.results_history[-50:],  # Last 50 results
            'all_results_count': len(self.results_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {filename}")
    
    def print_summary(self):
        """Print optimization summary"""
        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)
        
        if self.best_params:
            print("\nBest Parameters:")
            for key, value in sorted(self.best_params.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        if self.best_result:
            print("\nBest Results:")
            for key, value in sorted(self.best_result.items()):
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
        
        # Show top 5 results
        if self.results_history:
            print("\nTop 5 Results:")
            sorted_results = sorted(self.results_history, key=lambda x: x['total_return'], reverse=True)[:5]
            for i, result in enumerate(sorted_results, 1):
                print(f"\n  {i}. Return: {result['total_return']:.2f}% | "
                      f"Win Rate: {result['win_rate']:.1f}% | "
                      f"Trades: {result['total_trades']}")
                print(f"     Params: {result['params']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize backtest parameters')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to optimize (e.g., BTC/USDT ETH/USDT)')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--method', type=str, default=OPTIMIZATION_METHOD, 
                       choices=['grid', 'random', 'differential_evolution'],
                       help='Optimization method')
    parser.add_argument('--iterations', type=int, default=MAX_ITERATIONS,
                       help='Number of iterations (for random/evolution methods)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing (use single core)')
    parser.add_argument('--output', type=str, default='optimization_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    optimizer = ParameterOptimizer(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end
    )
    
    # Store parallel flag
    use_parallel = not args.no_parallel
    
    if use_parallel:
        print(f"Parallel processing enabled: Using up to {min(cpu_count(), 8)} CPU cores")
    else:
        print("Parallel processing disabled: Using single core")
    
    # Run optimization with parallel flag
    if args.method == 'grid':
        best_params = optimizer.optimize_grid_search(use_parallel=use_parallel)
    elif args.method == 'random':
        best_params = optimizer.optimize_random_search(n_iterations=args.iterations, use_parallel=use_parallel)
    elif args.method == 'differential_evolution':
        best_params = optimizer.optimize_differential_evolution(max_iterations=args.iterations)
    else:
        best_params = optimizer.optimize(method=args.method)
    optimizer.print_summary()
    optimizer.save_results(args.output)
    
    print(f"\nOptimization complete! Best parameters saved to {args.output}")


if __name__ == "__main__":
    main()

