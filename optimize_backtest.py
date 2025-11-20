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

# Entry parameters (for buy signals)
ENTRY_PARAMS = {
    'RISK_FRACTION': (0.10, 0.30),  # Position sizing (10% to 30%)
    'ADX_MIN': (18, 28),            # Minimum ADX for entry
    'RSI_MIN': (45, 55),            # Minimum RSI for entry
    'RSI_MAX': (65, 75),            # Maximum RSI for entry
    'RVOL_BASE': (1.2, 2.0),        # Base relative volume threshold
    'SCORE_GATE': (4.5, 6.0),       # Score gate threshold
    # Trend validation parameters (NEW - for predicting future growth)
    'MIN_SLOPE10_PCT': (0.02, 0.15),  # Minimum short-term price slope % (0.02% to 0.15% per period)
    'MIN_SLOPE20_PCT': (0.01, 0.10),  # Minimum medium-term price slope % (0.01% to 0.10% per period)
    'MAX_NEAR_HIGH_PCT': (-3.0, -0.5),  # Maximum distance from recent high % (avoid peaks: -3% to -0.5%)
}

# Exit/Trailing parameters (for sell signals and stop loss)
EXIT_PARAMS = {
    'CHAN_K_MEDIUM': (2.0, 3.5),   # Chandelier K for medium volatility
    'INIT_STOP_ATR_MULT': (1.0, 2.0),  # Initial stop ATR multiplier
    'ARM_TRAIL_PCT': (0.5, 1.5),   # Arm trailing after % gain
    'BE_R_MULT': (1.0, 2.0),       # Break-even at X R
}

# All parameters (for full optimization if needed)
OPTIMIZATION_PARAMS = {**ENTRY_PARAMS, **EXIT_PARAMS}

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
        # Store fixed params from previous phase
        self.fixed_entry_params = {}  # Used when optimizing exit params
        self.fixed_exit_params = {}   # Used when optimizing entry params
    
    def apply_params(self, params: Dict, phase: str = 'full') -> None:
        """
        Apply parameters to backtest module
        phase: 'entry', 'exit', or 'full'
        """
        # Merge with fixed params from previous phase
        if phase == 'entry':
            # When optimizing entry, use fixed exit params
            all_params = {**self.fixed_exit_params, **params}
        elif phase == 'exit':
            # When optimizing exit, use fixed entry params
            all_params = {**self.fixed_entry_params, **params}
        else:
            # Full optimization - use all params
            all_params = params
        
        # Update backtest module parameters
        backtest.RISK_FRACTION = all_params.get('RISK_FRACTION', backtest.RISK_FRACTION)
        
        # Update trader module parameters (these affect buy/exit decisions)
        import trader_v21 as trader
        
        # Entry parameters
        if 'ADX_MIN' in all_params:
            trader.ADX_MIN_DEFAULT = all_params['ADX_MIN']
        if 'RSI_MIN' in all_params:
            trader.RSI_MIN_DEFAULT = all_params['RSI_MIN']
        if 'RSI_MAX' in all_params:
            trader.RSI_MAX_DEFAULT = all_params['RSI_MAX']
        if 'RVOL_BASE' in all_params:
            trader.RVOL_BASE_DEFAULT = all_params['RVOL_BASE']
        if 'SCORE_GATE' in all_params:
            trader.set_score_gate(all_params['SCORE_GATE'])
        
        # Exit parameters
        if 'CHAN_K_MEDIUM' in all_params:
            trader.CHAN_K_MEDIUM = all_params['CHAN_K_MEDIUM']
        if 'INIT_STOP_ATR_MULT' in all_params:
            trader.INIT_STOP_ATR_MULT = all_params['INIT_STOP_ATR_MULT']
        if 'ARM_TRAIL_PCT' in all_params:
            trader.ARM_TRAIL_PCT = all_params['ARM_TRAIL_PCT']
        if 'BE_R_MULT' in all_params:
            trader.BE_R_MULT = all_params['BE_R_MULT']
        
        # Trend validation parameters (for predicting future growth)
        if 'MIN_SLOPE10_PCT' in all_params:
            trader.MIN_SLOPE10_PCT = all_params['MIN_SLOPE10_PCT']
        if 'MIN_SLOPE20_PCT' in all_params:
            trader.MIN_SLOPE20_PCT = all_params['MIN_SLOPE20_PCT']
        if 'MAX_NEAR_HIGH_PCT' in all_params:
            trader.MAX_NEAR_HIGH_PCT = all_params['MAX_NEAR_HIGH_PCT']
    
    def evaluate_params(self, params: Dict, phase: str = 'full', objective: str = 'pnl') -> float:
        """
        Run backtest with given parameters and return score (for minimization)
        phase: 'entry', 'exit', or 'full'
        objective: 'pnl' (maximize profit) or 'trend' (maximize trend accuracy)
        """
        try:
            # Apply parameters
            self.apply_params(params, phase=phase)
            
            # Run backtest
            # If optimizing entry phase with trend objective, use entry-only mode
            entry_only = (phase == 'entry' and getattr(self, 'objective', 'trend') == 'trend')
            engine = backtest.run_backtest(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                initial_balance=FIXED_PARAMS['INITIAL_BALANCE'],
                entry_only=entry_only,
                entry_hold_periods=24  # Hold for 24 periods (24 hours if 1h timeframe)
            )
            
            # Get statistics
            stats = engine.get_statistics()
            
            if not stats:
                return 1e6  # Very bad score if no trades
            
            total_trades = stats.get('total_trades', 0)
            if total_trades < 10:
                return 1e6  # Very bad score if too few trades
            
            if objective == 'trend':
                # Objective: maximize trend accuracy (minimize negative trend accuracy)
                trend_accuracy = stats.get('trend_accuracy_pct', 0.0)  # % of trades where price went up
                trend_avg_gain = stats.get('trend_avg_gain_pct', 0.0)  # Average price gain
                sustained_trend = stats.get('sustained_trend_pct', 0.0)  # % of sustained trends
                
                # Composite score: prioritize trend accuracy and average gain
                # We want to minimize the negative of (accuracy + avg_gain + sustained)
                # Higher is better, so we minimize the negative
                score = -(trend_accuracy * 0.5 + trend_avg_gain * 0.3 + sustained_trend * 0.2)
                
                # Store result
                result = {
                    'params': params.copy(),
                    'stats': stats,
                    'score': -score,  # Positive score for reporting
                    'trend_accuracy': trend_accuracy,
                    'trend_avg_gain': trend_avg_gain,
                    'sustained_trend': sustained_trend,
                    'total_trades': total_trades
                }
                self.results_history.append(result)
                
                print(f"  Params: {params} | Trend Acc: {trend_accuracy:.1f}% | Avg Gain: {trend_avg_gain:.2f}% | Sustained: {sustained_trend:.1f}% | Trades: {total_trades}")
                
            else:  # objective == 'pnl'
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
                
                # Store result
                result = {
                    'params': params.copy(),
                    'stats': stats,
                    'score': -score,  # Positive score for reporting
                    'total_return': total_return,
                    'win_rate': win_rate,
                    'max_drawdown': max_dd,
                    'total_trades': total_trades
                }
                self.results_history.append(result)
                
                print(f"  Params: {params} | Return: {total_return:.2f}% | Win Rate: {win_rate:.1f}% | Trades: {total_trades}")
            
            return score
            
        except Exception as e:
            print(f"  Error evaluating params {params}: {e}")
            return 1e6  # Very bad score on error
    
    def optimize_grid_search(self, use_parallel: bool = True, phase: str = 'full') -> Dict:
        """Grid search optimization with optional parallel processing"""
        phase_name = {'entry': 'ENTRY', 'exit': 'EXIT', 'full': 'FULL'}.get(phase, 'FULL')
        print(f"Starting Grid Search Optimization ({phase_name} parameters)...")
        
        # Select which params to optimize based on phase
        if phase == 'entry':
            params_to_optimize = ENTRY_PARAMS
        elif phase == 'exit':
            params_to_optimize = EXIT_PARAMS
        else:
            params_to_optimize = OPTIMIZATION_PARAMS
        
        print(f"Testing {len(params_to_optimize)} parameters with {GRID_RESOLUTION} points each")
        
        # Create parameter grids
        param_names = list(params_to_optimize.keys())
        param_ranges = [params_to_optimize[name] for name in param_names]
        
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
            
            args_list = [(params, self.symbols, self.start_date, self.end_date, phase) 
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
                score = self.evaluate_params(params, phase=phase, objective=getattr(self, 'objective', 'trend'))
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"  *** NEW BEST: Score={-score:.2f}% ***")
        
        return best_params
    
    def _evaluate_params_wrapper(self, args):
        """Wrapper for multiprocessing"""
        if len(args) == 5:
            params, symbols, start_date, end_date, phase = args
        else:
            params, symbols, start_date, end_date = args
            phase = 'full'
        try:
            # Create a temporary optimizer instance for this evaluation
            temp_optimizer = ParameterOptimizer(symbols, start_date, end_date)
            # Copy fixed params from parent
            temp_optimizer.fixed_entry_params = self.fixed_entry_params.copy()
            temp_optimizer.fixed_exit_params = self.fixed_exit_params.copy()
            return temp_optimizer.evaluate_params(params, phase=phase)
        except Exception as e:
            print(f"  Error in parallel evaluation: {e}")
            return 1e6
    
    def optimize_random_search(self, n_iterations: int = MAX_ITERATIONS, use_parallel: bool = True, phase: str = 'full') -> Dict:
        """Random search optimization with optional parallel processing"""
        phase_name = {'entry': 'ENTRY', 'exit': 'EXIT', 'full': 'FULL'}.get(phase, 'FULL')
        print(f"Starting Random Search Optimization ({phase_name} parameters, {n_iterations} iterations)...")
        
        # Select which params to optimize based on phase
        if phase == 'entry':
            params_to_optimize = ENTRY_PARAMS
        elif phase == 'exit':
            params_to_optimize = EXIT_PARAMS
        else:
            params_to_optimize = OPTIMIZATION_PARAMS
        
        # Generate all parameter combinations first
        param_combinations = []
        for i in range(n_iterations):
            params = {}
            for name, (min_val, max_val) in params_to_optimize.items():
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
            
            args_list = [(params, self.symbols, self.start_date, self.end_date, phase) 
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
                score = self.evaluate_params(params, phase=phase, objective=getattr(self, 'objective', 'trend'))
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"  *** NEW BEST: Score={-score:.2f}% ***")
        
        return best_params
    
    def optimize_differential_evolution(self, max_iterations: int = None, phase: str = 'full') -> Dict:
        """
        Differential evolution optimization (scipy)
        phase: 'entry', 'exit', or 'full'
        """
        phase_name = {'entry': 'ENTRY', 'exit': 'EXIT', 'full': 'FULL'}.get(phase, 'FULL')
        print(f"Starting Differential Evolution Optimization ({phase_name} parameters)...")
        
        if max_iterations is None:
            max_iterations = MAX_ITERATIONS
        
        # Select which params to optimize based on phase
        if phase == 'entry':
            params_to_optimize = ENTRY_PARAMS
        elif phase == 'exit':
            params_to_optimize = EXIT_PARAMS
        else:
            params_to_optimize = OPTIMIZATION_PARAMS
        
        param_names = list(params_to_optimize.keys())
        bounds = [params_to_optimize[name] for name in param_names]
        
        def objective(x):
            params = dict(zip(param_names, x))
            return self.evaluate_params(params, phase=phase, objective=getattr(self, 'objective', 'trend'))
        
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
            # Sort by appropriate metric based on objective
            objective = getattr(self, 'objective', 'trend')
            if objective == 'trend':
                sorted_results = sorted(self.results_history, key=lambda x: x.get('trend_accuracy', 0), reverse=True)[:5]
                for i, result in enumerate(sorted_results, 1):
                    print(f"\n  {i}. Trend Acc: {result.get('trend_accuracy', 0):.1f}% | "
                          f"Avg Gain: {result.get('trend_avg_gain', 0):.2f}% | "
                          f"Sustained: {result.get('sustained_trend', 0):.1f}% | "
                          f"Trades: {result['total_trades']}")
                    print(f"     Params: {result['params']}")
            else:
                sorted_results = sorted(self.results_history, key=lambda x: x.get('total_return', -100), reverse=True)[:5]
                for i, result in enumerate(sorted_results, 1):
                    print(f"\n  {i}. Return: {result.get('total_return', 0):.2f}% | "
                          f"Win Rate: {result.get('win_rate', 0):.1f}% | "
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
    parser.add_argument('--phase', type=str, default='full',
                       choices=['entry', 'exit', 'full', 'both'],
                       help='Optimization phase: entry (buy params), exit (sell/trailing params), full (all), both (entry then exit)')
    parser.add_argument('--entry-params-file', type=str, default=None,
                       help='JSON file with optimized entry params (for exit phase optimization)')
    parser.add_argument('--objective', type=str, default='trend',
                       choices=['pnl', 'trend'],
                       help='Optimization objective: pnl (maximize profit) or trend (maximize trend accuracy)')
    
    args = parser.parse_args()
    
    optimizer = ParameterOptimizer(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end
    )
    
    # Load entry params if provided (for exit phase optimization)
    if args.entry_params_file and os.path.exists(args.entry_params_file):
        with open(args.entry_params_file, 'r') as f:
            entry_results = json.load(f)
            optimizer.fixed_entry_params = entry_results.get('best_parameters', {})
            print(f"Loaded entry parameters from {args.entry_params_file}")
            print(f"Fixed entry params: {optimizer.fixed_entry_params}")
    
    # Store parallel flag
    use_parallel = not args.no_parallel
    
    if use_parallel:
        print(f"Parallel processing enabled: Using up to {min(cpu_count(), 8)} CPU cores")
    else:
        print("Parallel processing disabled: Using single core")
    
    # Determine which phase(s) to run
    phases_to_run = []
    if args.phase == 'both':
        phases_to_run = ['entry', 'exit']
    else:
        phases_to_run = [args.phase]
    
    # Show objective
    print(f"\nOptimization Objective: {args.objective.upper()}")
    if args.objective == 'trend':
        print("  → Maximizing trend accuracy (identifying when price will go up)")
        print("  → Metrics: Trend Accuracy %, Average Gain %, Sustained Trends %")
    else:
        print("  → Maximizing profit (PnL)")
        print("  → Metrics: Total Return %, Win Rate %, Max Drawdown %")
    
    all_best_params = {}
    
    # Run optimization for each phase
    for phase in phases_to_run:
        print("\n" + "=" * 80)
        print(f"PHASE: {phase.upper()} PARAMETERS")
        print("=" * 80)
        
        if phase == 'entry':
            print("Optimizing BUY signal parameters...")
            print("Parameters: RISK_FRACTION, ADX_MIN, RSI_MIN, RSI_MAX, RVOL_BASE, SCORE_GATE")
        elif phase == 'exit':
            print("Optimizing SELL/TRAILING STOP parameters...")
            print("Parameters: CHAN_K_MEDIUM, INIT_STOP_ATR_MULT, ARM_TRAIL_PCT, BE_R_MULT")
            if optimizer.fixed_entry_params:
                print(f"Using fixed entry params: {optimizer.fixed_entry_params}")
            else:
                print("WARNING: No fixed entry params set. Using default entry params.")
        else:
            print("Optimizing ALL parameters...")
        
        # Run optimization
        if args.method == 'grid':
            best_params = optimizer.optimize_grid_search(
                use_parallel=use_parallel,
                phase=phase
            )
        elif args.method == 'random':
            best_params = optimizer.optimize_random_search(
                n_iterations=args.iterations,
                use_parallel=use_parallel,
                phase=phase
            )
        elif args.method == 'differential_evolution':
            best_params = optimizer.optimize_differential_evolution(
                max_iterations=args.iterations, 
                phase=phase
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")
        
        # Store results
        if phase == 'entry':
            optimizer.fixed_entry_params = best_params
            all_best_params.update(best_params)
        elif phase == 'exit':
            optimizer.fixed_exit_params = best_params
            all_best_params.update(best_params)
        else:
            all_best_params = best_params
        
        # Final evaluation with best params for this phase
        print(f"\n--- FINAL EVALUATION ({phase.upper()} PHASE) ---")
        optimizer.apply_params(best_params, phase=phase)
        engine = backtest.run_backtest(
            symbols=optimizer.symbols,
            start_date=optimizer.start_date,
            end_date=optimizer.end_date,
            initial_balance=FIXED_PARAMS['INITIAL_BALANCE']
        )
        final_stats = engine.get_statistics()
        
        optimizer.best_params = all_best_params
        optimizer.best_result = final_stats
        
        # Save intermediate results
        phase_output = args.output.replace('.json', f'_{phase}.json')
        optimizer.save_results(phase_output)
        optimizer.print_summary()
        
        print(f"\n{phase.upper()} phase complete! Results saved to {phase_output}")
    
    # Save final combined results
    if len(phases_to_run) > 1:
        print("\n" + "=" * 80)
        print("FINAL COMBINED RESULTS")
        print("=" * 80)
        optimizer.best_params = all_best_params
        optimizer.save_results(args.output)
        optimizer.print_summary()
        print(f"\nAll phases complete! Final results saved to {args.output}")


if __name__ == "__main__":
    main()

