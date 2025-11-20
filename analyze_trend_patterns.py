"""
Analyze successful trends and find common patterns in indicators/parameters
that preceded them. This helps identify what conditions predict future growth.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.stats import linregress
import backtest_trader_v21 as backtest

# Configuration
MIN_TREND_GAIN_PCT = 3.0  # Minimum price gain to consider a "successful trend"
MIN_TREND_DURATION_PERIODS = 12  # Minimum periods the trend lasted
ANALYSIS_LOOKBACK_PERIODS = 20  # How many periods before trend to analyze
MAX_MAE_PCT = -3.0  # Maximum adverse excursion (drawdown) allowed before considering trend "good"

def compute_excursions(entry_price: float, future_prices: pd.Series) -> Tuple[float, float]:
    """
    Compute MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion)
    
    Returns:
        mfe: Maximum % in favor (positive)
        mae: Maximum % against (negative)
    """
    if len(future_prices) == 0 or entry_price <= 0:
        return 0.0, 0.0
    
    rel_changes = (future_prices / entry_price - 1.0) * 100.0
    mfe = float(rel_changes.max())  # máximo a favor
    mae = float(rel_changes.min())  # máximo en contra (negativo)
    return mfe, mae

def best_range_for_indicator(values: np.ndarray, labels: np.ndarray, min_width_q: float = 0.3) -> Optional[Dict]:
    """
    Find optimal range for an indicator using grid search over quantiles.
    Now includes baseline comparison to show improvement over random.
    
    Args:
        values: Array of indicator values
        labels: Boolean array (True=success, False=failure)
        min_width_q: Minimum width of range as quantile fraction (default 0.3 = 30%)
    
    Returns:
        Dict with 'low', 'high', 'precision', 'coverage', 'score', 'baseline', 'improvement' or None
    """
    # Filter out NaN values
    valid_mask = ~np.isnan(values)
    values_clean = values[valid_mask]
    labels_clean = labels[valid_mask]
    
    if len(values_clean) < 50:
        return None
    
    # Calculate baseline (overall success rate)
    total_success = labels_clean.sum()
    total_samples = len(labels_clean)
    baseline_precision = total_success / total_samples if total_samples > 0 else 0.0
    
    if total_success == 0:
        return None
    
    qs = np.linspace(0.05, 0.95, 19)  # 5%, 10%, ..., 95%
    v_sorted = np.sort(values_clean)
    
    best = None
    
    for i, q1 in enumerate(qs):
        for q2 in qs[i+1:]:
            if q2 - q1 < min_width_q:
                continue
            
            low = np.quantile(v_sorted, q1)
            high = np.quantile(v_sorted, q2)
            in_range = (values_clean >= low) & (values_clean <= high)
            n_range = in_range.sum()
            
            if n_range < 50:
                continue
            
            succ_in_range = (labels_clean & in_range).sum()
            if succ_in_range == 0:
                continue
            
            precision = succ_in_range / n_range
            coverage = succ_in_range / total_success
            # Score: balance precision and coverage (penalize tiny ranges)
            score = precision * (coverage ** 0.5)
            
            # Improvement over baseline (multiplicative factor)
            improvement_factor = precision / baseline_precision if baseline_precision > 0 else 0.0
            
            if best is None or score > best["score"]:
                best = {
                    "low": float(low),
                    "high": float(high),
                    "precision": float(precision),
                    "coverage": float(coverage),
                    "score": float(score),
                    "baseline": float(baseline_precision),
                    "improvement_factor": float(improvement_factor),
                    "n_samples": int(n_range),
                    "n_success": int(succ_in_range),
                }
    
    return best

def find_successful_trends(symbol_data: Dict[str, pd.DataFrame], 
                           min_gain_pct: float = MIN_TREND_GAIN_PCT,
                           min_duration: int = MIN_TREND_DURATION_PERIODS,
                           max_mae_pct: float = MAX_MAE_PCT) -> List[Dict]:
    """
    Find all successful upward trends in the data
    Returns list of trend events with entry conditions
    """
    successful_trends = []
    
    for symbol, df in symbol_data.items():
        if len(df) < 100:
            continue
        
        # Use while loop to skip overlapping windows (jump to peak after finding a trend)
        i = 100
        while i < len(df) - min_duration:
            entry_price = df['close'].iloc[i]
            
            # Check price movement over next periods
            future_prices = df['close'].iloc[i+1:i+min_duration+1]
            if len(future_prices) < min_duration:
                i += 1
                continue
            
            max_future_price = future_prices.max()
            gain_pct = ((max_future_price / entry_price) - 1) * 100
            
            # Compute MFE and MAE (risk-aware analysis)
            mfe, mae = compute_excursions(entry_price, future_prices)
            
            # Check if this is a successful trend (with risk constraint)
            # Only consider "good" trends that respect max drawdown (compatible with stops)
            if mfe >= min_gain_pct and mae >= max_mae_pct:
                # Find when the peak occurred
                peak_idx = future_prices.idxmax()
                peak_price = max_future_price
                
                # Get entry conditions (indicators at entry point)
                entry_row = df.iloc[i]
                entry_time = df.index[i]
                
                # Calculate price slopes directly if not available
                price_slope10_pct = None
                price_slope20_pct = None
                price_near_high_pct = None
                
                # Calculate volume metrics
                volume_10_avg = None
                volume_slope10 = None
                volume_ratio = None  # Current volume / 20-period average
                
                if i >= 10:
                    try:
                        # Average volume over last 10 periods
                        volumes_10 = df['volume'].iloc[i-9:i+1].values
                        if len(volumes_10) == 10 and not np.any(np.isnan(volumes_10)):
                            volume_10_avg = float(np.mean(volumes_10))
                            
                            # Volume slope (trend in volume)
                            x10 = np.arange(10)
                            volume_slope10 = linregress(x10, volumes_10).slope
                            if np.isnan(volume_slope10):
                                volume_slope10 = None
                            else:
                                volume_slope10 = float(volume_slope10)
                    except Exception:
                        pass
                
                if i >= 20:
                    try:
                        # Volume ratio: current volume vs 20-period average
                        volumes_20 = df['volume'].iloc[i-19:i+1].values
                        current_volume = df['volume'].iloc[i]
                        if not np.isnan(current_volume) and len(volumes_20) == 20:
                            volume_20_avg = np.mean(volumes_20[~np.isnan(volumes_20)])
                            if volume_20_avg > 0:
                                volume_ratio = float(current_volume / volume_20_avg)
                    except Exception:
                        pass
                
                # Calculate 10-period slope (last 10 periods including current)
                if i >= 10:
                    try:
                        x10 = np.arange(10)
                        prices_10 = df['close'].iloc[i-9:i+1].values
                        if len(prices_10) == 10:
                            # Check for NaN values
                            valid_prices = prices_10[~np.isnan(prices_10)]
                            if len(valid_prices) == 10:
                                slope10 = linregress(x10, prices_10).slope
                                if not np.isnan(slope10) and entry_price > 0:
                                    price_slope10_pct = float((slope10 / entry_price) * 100)
                    except Exception as e:
                        # Keep as None if calculation fails
                        pass
                
                # Calculate 20-period slope (last 20 periods including current)
                if i >= 20:
                    try:
                        x20 = np.arange(20)
                        prices_20 = df['close'].iloc[i-19:i+1].values
                        if len(prices_20) == 20:
                            # Check for NaN values
                            valid_prices = prices_20[~np.isnan(prices_20)]
                            if len(valid_prices) == 20:
                                slope20 = linregress(x20, prices_20).slope
                                if not np.isnan(slope20) and entry_price > 0:
                                    price_slope20_pct = float((slope20 / entry_price) * 100)
                        
                        # Calculate distance from recent high
                        recent_high_20 = df['high'].iloc[i-19:i+1].max()
                        if not np.isnan(recent_high_20) and recent_high_20 > 0:
                            price_near_high_pct = float(((entry_price / recent_high_20) - 1) * 100)
                    except Exception as e:
                        # Keep as None if calculation fails
                        pass
                
                # Get indicators from 20 periods before entry (to see what preceded it)
                if i >= ANALYSIS_LOOKBACK_PERIODS:
                    lookback_start = i - ANALYSIS_LOOKBACK_PERIODS
                    lookback_data = df.iloc[lookback_start:i+1]
                    
                    trend_info = {
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'entry_price': float(entry_price),
                        'peak_time': peak_idx,
                        'peak_price': float(peak_price),
                        'gain_pct': gain_pct,
                        'mfe_pct': mfe,  # Max Favorable Excursion
                        'mae_pct': mae,  # Max Adverse Excursion (negative)
                        'duration_hours': (peak_idx - entry_time).total_seconds() / 3600 if hasattr(peak_idx - entry_time, 'total_seconds') else min_duration,  # Duration in hours, not periods
                        'entry_indicators': {
                            'ADX': float(entry_row['ADX']) if pd.notna(entry_row.get('ADX')) else None,
                            'RSI': float(entry_row['RSI']) if pd.notna(entry_row.get('RSI')) else None,
                            'RVOL10': float(entry_row['RVOL10']) if pd.notna(entry_row.get('RVOL10')) else None,
                            'EMA20_above_EMA50': bool(entry_row['EMA20'] > entry_row['EMA50']) if pd.notna(entry_row.get('EMA20')) and pd.notna(entry_row.get('EMA50')) else None,
                            'close_above_EMA20': bool(entry_row['close'] > entry_row['EMA20']) if pd.notna(entry_row.get('EMA20')) else None,
                            'price_slope10_pct': price_slope10_pct,
                            'price_slope20_pct': price_slope20_pct,
                            'price_near_high_pct': price_near_high_pct,
                            'ATR_pct': float((entry_row.get('ATR', 0) / entry_price * 100)) if pd.notna(entry_row.get('ATR')) else None,
                            # Volume metrics
                            'volume_10_avg': volume_10_avg,
                            'volume_slope10': volume_slope10,
                            'volume_ratio': volume_ratio,  # Current volume / 20-period avg
                            'current_volume': float(df['volume'].iloc[i]) if pd.notna(df['volume'].iloc[i]) else None,
                        },
                        'lookback_indicators': {
                            # Average indicators over lookback period
                            'avg_ADX': float(lookback_data['ADX'].mean()) if 'ADX' in lookback_data.columns else None,
                            'avg_RSI': float(lookback_data['RSI'].mean()) if 'RSI' in lookback_data.columns else None,
                            'avg_RVOL10': float(lookback_data['RVOL10'].mean()) if 'RVOL10' in lookback_data.columns else None,
                            'trending_up_flag': int((lookback_data['close'].iloc[-1] > lookback_data['close'].iloc[0])),  # 0/1 flag, not "days"
                            'price_slope_lookback': float((lookback_data['close'].iloc[-1] / lookback_data['close'].iloc[0] - 1) * 100),
                        }
                    }
                    
                    successful_trends.append(trend_info)
                    
                    # Skip to peak to avoid overlapping windows (more realistic for a bot that enters once per movement)
                    try:
                        peak_position = df.index.get_loc(peak_idx)
                        i = peak_position + 1  # Start checking from after the peak
                        continue
                    except (KeyError, ValueError):
                        # If peak_idx not found in index, just increment normally
                        i += 1
                        continue
            else:
                # No successful trend, move to next candle
                i += 1
    
    return successful_trends

def analyze_common_patterns(trends: List[Dict]) -> Dict:
    """
    Analyze what indicators/parameters are common across successful trends
    """
    if not trends:
        return {}
    
    # Collect all indicator values (skip boolean values for percentile calculations)
    indicator_values = defaultdict(list)
    lookback_values = defaultdict(list)
    boolean_indicators = defaultdict(list)  # Track boolean indicators separately
    
    for trend in trends:
        for key, value in trend['entry_indicators'].items():
            if value is not None:
                # Handle boolean values separately
                if isinstance(value, bool):
                    boolean_indicators[key].append(value)
                else:
                    indicator_values[key].append(value)
        
        for key, value in trend['lookback_indicators'].items():
            if value is not None:
                if isinstance(value, bool):
                    boolean_indicators[key].append(value)
                else:
                    lookback_values[key].append(value)
    
    # Calculate statistics
    patterns = {
        'total_trends': len(trends),
        'entry_indicators': {},
        'lookback_indicators': {},
        'gain_statistics': {
            'avg_gain_pct': np.mean([t['gain_pct'] for t in trends]),
            'median_gain_pct': np.median([t['gain_pct'] for t in trends]),
            'min_gain_pct': np.min([t['gain_pct'] for t in trends]),
            'max_gain_pct': np.max([t['gain_pct'] for t in trends]),
        },
        'risk_statistics': {
            'avg_mfe_pct': np.mean([t.get('mfe_pct', 0) for t in trends]),
            'median_mfe_pct': np.median([t.get('mfe_pct', 0) for t in trends]),
            'avg_mae_pct': np.mean([t.get('mae_pct', 0) for t in trends]),
            'median_mae_pct': np.median([t.get('mae_pct', 0) for t in trends]),
            'worst_mae_pct': np.min([t.get('mae_pct', 0) for t in trends]),
        }
    }
    
    # Analyze entry indicators (numeric)
    for indicator, values in indicator_values.items():
        if values:
            # Convert to numpy array and ensure numeric
            values_array = np.array(values, dtype=float)
            patterns['entry_indicators'][indicator] = {
                'mean': float(np.mean(values_array)),
                'median': float(np.median(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'q25': float(np.percentile(values_array, 25)),
                'q75': float(np.percentile(values_array, 75)),
            }
    
    # Analyze boolean indicators separately
    for indicator, values in boolean_indicators.items():
        if values:
            true_count = sum(values)
            false_count = len(values) - true_count
            patterns['entry_indicators'][indicator] = {
                'true_percentage': float(true_count / len(values) * 100),
                'false_percentage': float(false_count / len(values) * 100),
                'true_count': true_count,
                'false_count': false_count,
                'total': len(values),
            }
    
    # Analyze lookback indicators (numeric)
    for indicator, values in lookback_values.items():
        if values:
            # Convert to numpy array and ensure numeric
            values_array = np.array(values, dtype=float)
            patterns['lookback_indicators'][indicator] = {
                'mean': float(np.mean(values_array)),
                'median': float(np.median(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
            }
    
    return patterns

def find_optimal_thresholds(trends: List[Dict]) -> Dict:
    """
    Find optimal threshold values by analyzing what ranges appear most often
    in successful trends
    """
    if not trends:
        return {}
    
    thresholds = {}
    
    # For each indicator, find the range where most successful trends occur
    indicators_to_analyze = [
        'ADX', 'RSI', 'RVOL10', 
        'price_slope10_pct', 'price_slope20_pct', 'price_near_high_pct',
        'volume_10_avg', 'volume_slope10', 'volume_ratio'
    ]
    
    for indicator in indicators_to_analyze:
        values = []
        for trend in trends:
            val = trend['entry_indicators'].get(indicator)
            if val is not None:
                values.append(val)
        
        if values:
            # Find range that contains 80% of successful trends
            sorted_vals = sorted(values)
            lower_idx = int(len(sorted_vals) * 0.1)  # 10th percentile
            upper_idx = int(len(sorted_vals) * 0.9)  # 90th percentile
            
            thresholds[indicator] = {
                'min': float(sorted_vals[lower_idx]),
                'max': float(sorted_vals[upper_idx]),
                'median': float(np.median(values)),
                'mean': float(np.mean(values)),
            }
    
    return thresholds

def find_optimal_ranges_grid_search(successful_trends: List[Dict], 
                                     failed_trends: List[Dict],
                                     indicators: List[str]) -> Dict:
    """
    Find optimal ranges for indicators using grid search over quantiles.
    More sophisticated than simple percentiles - maximizes precision * coverage.
    """
    optimal_ranges = {}
    
    # Combine all trends into arrays
    all_trends = successful_trends + failed_trends
    labels = np.array([True] * len(successful_trends) + [False] * len(failed_trends))
    
    for indicator in indicators:
        values = []
        valid_labels = []
        
        for i, trend in enumerate(all_trends):
            val = trend['entry_indicators'].get(indicator)
            if val is not None:
                values.append(val)
                valid_labels.append(labels[i])
        
        if len(values) < 50:
            continue
        
        values_array = np.array(values)
        labels_array = np.array(valid_labels)
        
        best_range = best_range_for_indicator(values_array, labels_array, min_width_q=0.3)
        if best_range:
            optimal_ranges[indicator] = best_range
    
    return optimal_ranges

def evaluate_ranges_on_dataset(successful_trends: List[Dict], 
                                failed_trends: List[Dict],
                                optimal_ranges: Dict) -> Dict:
    """
    Evaluate optimal ranges (learned on training data) on a validation dataset.
    Returns validation metrics for each range.
    """
    validation_results = {}
    
    # Combine all trends
    all_trends = successful_trends + failed_trends
    labels = np.array([True] * len(successful_trends) + [False] * len(failed_trends))
    
    # Calculate baseline for validation set
    total_val = len(all_trends)
    success_val = len(successful_trends)
    baseline_val = success_val / total_val if total_val > 0 else 0.0
    
    for indicator, range_info in optimal_ranges.items():
        low = range_info['low']
        high = range_info['high']
        
        # Get values and labels for this indicator
        values = []
        valid_labels = []
        
        for i, trend in enumerate(all_trends):
            val = trend['entry_indicators'].get(indicator)
            if val is not None:
                values.append(val)
                valid_labels.append(labels[i])
        
        if len(values) < 10:
            continue
        
        values_array = np.array(values)
        labels_array = np.array(valid_labels)
        
        # Count samples in range
        in_range = (values_array >= low) & (values_array <= high)
        n_range = in_range.sum()
        
        if n_range < 10:
            continue
        
        # Calculate precision in validation
        succ_in_range = (labels_array & in_range).sum()
        precision_val = succ_in_range / n_range if n_range > 0 else 0.0
        coverage_val = succ_in_range / success_val if success_val > 0 else 0.0
        improvement_factor_val = precision_val / baseline_val if baseline_val > 0 else 0.0
        
        validation_results[indicator] = {
            'range': {'low': float(low), 'high': float(high)},
            'baseline_val': float(baseline_val),
            'precision_val': float(precision_val),
            'coverage_val': float(coverage_val),
            'improvement_factor_val': float(improvement_factor_val),
            'n_samples_val': int(n_range),
            'n_success_val': int(succ_in_range),
            # Compare with training
            'precision_train': range_info.get('precision', 0),
            'improvement_factor_train': range_info.get('improvement_factor', 0),
            'generalization_ok': abs(improvement_factor_val - range_info.get('improvement_factor', 0)) < 2.0,  # Rough check
        }
    
    return {
        'baseline': float(baseline_val),
        'total_trades': total_val,
        'successful_trades': success_val,
        'indicator_results': validation_results,
    }

def compare_with_failed_trends(symbol_data: Dict[str, pd.DataFrame],
                               successful_trends: List[Dict],
                               min_gain_pct: float = MIN_TREND_GAIN_PCT,
                               max_mae_pct: float = MAX_MAE_PCT) -> Tuple[Dict, List[Dict]]:
    """
    Compare indicators from successful trends vs failed trends.
    Uses same logic as successful trends (jump to end of window) to avoid bias.
    """
    # Find failed trends using same windowing logic as successful trends
    # This avoids over-representing failures
    failed_trends = []
    
    for symbol, df in symbol_data.items():
        if len(df) < 100:
            continue
        
        # Use same while loop logic as successful trends to avoid bias
        i = 100
        while i < len(df) - MIN_TREND_DURATION_PERIODS:
            entry_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+MIN_TREND_DURATION_PERIODS+1]
            
            if len(future_prices) < MIN_TREND_DURATION_PERIODS:
                i += 1  # Fix: Always increment to avoid infinite loop
                continue
            
            max_future_price = future_prices.max()
            gain_pct = ((max_future_price / entry_price) - 1) * 100
            
            # Compute MFE and MAE
            mfe, mae = compute_excursions(entry_price, future_prices)
            
            # Failed trend: price didn't go up much OR exceeded max MAE
            # Use same criteria as successful trends but inverted
            is_failed = (mfe < min_gain_pct) or (mae < max_mae_pct)
            
            if is_failed:
                entry_row = df.iloc[i]
                entry_time = df.index[i]
                
                # Calculate price slopes directly
                price_slope10_pct = None
                price_slope20_pct = None
                price_near_high_pct = None
                
                # Calculate volume metrics
                volume_10_avg = None
                volume_slope10 = None
                volume_ratio = None
                
                if i >= 10:
                    try:
                        volumes_10 = df['volume'].iloc[i-9:i+1].values
                        if len(volumes_10) == 10 and not np.any(np.isnan(volumes_10)):
                            volume_10_avg = float(np.mean(volumes_10))
                            x10 = np.arange(10)
                            volume_slope10 = linregress(x10, volumes_10).slope
                            if np.isnan(volume_slope10):
                                volume_slope10 = None
                            else:
                                volume_slope10 = float(volume_slope10)
                    except Exception:
                        pass
                
                if i >= 20:
                    try:
                        volumes_20 = df['volume'].iloc[i-19:i+1].values
                        current_volume = df['volume'].iloc[i]
                        if not np.isnan(current_volume) and len(volumes_20) == 20:
                            volume_20_avg = np.mean(volumes_20[~np.isnan(volumes_20)])
                            if volume_20_avg > 0:
                                volume_ratio = float(current_volume / volume_20_avg)
                    except Exception:
                        pass
                
                try:
                    if i >= 10:
                        x10 = np.arange(10)
                        prices_10 = df['close'].iloc[i-9:i+1].values
                        if len(prices_10) == 10 and not np.any(np.isnan(prices_10)):
                            slope10 = linregress(x10, prices_10).slope
                            if not np.isnan(slope10) and entry_price > 0:
                                price_slope10_pct = float((slope10 / entry_price) * 100)
                except Exception:
                    pass
                
                try:
                    if i >= 20:
                        x20 = np.arange(20)
                        prices_20 = df['close'].iloc[i-19:i+1].values
                        if len(prices_20) == 20 and not np.any(np.isnan(prices_20)):
                            slope20 = linregress(x20, prices_20).slope
                            if not np.isnan(slope20) and entry_price > 0:
                                price_slope20_pct = float((slope20 / entry_price) * 100)
                        
                        recent_high_20 = df['high'].iloc[i-19:i+1].max()
                        if not np.isnan(recent_high_20) and recent_high_20 > 0:
                            price_near_high_pct = float(((entry_price / recent_high_20) - 1) * 100)
                except Exception:
                    pass
                
                if i >= ANALYSIS_LOOKBACK_PERIODS:
                    # Find end of window (similar to peak for successful trends)
                    # Use the last price in the window as the "exit" point
                    window_end_idx = i + MIN_TREND_DURATION_PERIODS
                    if window_end_idx < len(df):
                        window_end_time = df.index[window_end_idx]
                    else:
                        window_end_time = df.index[-1]
                    
                    failed_trends.append({
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': window_end_time,
                        'gain_pct': gain_pct,
                        'mfe_pct': mfe,
                        'mae_pct': mae,
                        'entry_indicators': {
                            'ADX': float(entry_row['ADX']) if pd.notna(entry_row.get('ADX')) else None,
                            'RSI': float(entry_row['RSI']) if pd.notna(entry_row.get('RSI')) else None,
                            'RVOL10': float(entry_row['RVOL10']) if pd.notna(entry_row.get('RVOL10')) else None,
                            'price_slope10_pct': price_slope10_pct,
                            'price_slope20_pct': price_slope20_pct,
                            'price_near_high_pct': price_near_high_pct,
                            # Volume metrics
                            'volume_10_avg': volume_10_avg,
                            'volume_slope10': volume_slope10,
                            'volume_ratio': volume_ratio,
                        }
                    })
                    
                    # Skip to end of window (same logic as successful trends)
                    i = window_end_idx + 1
                    continue
                else:
                    # Fix: No hay suficiente lookback para registrar el fallo, pero avanzamos una vela
                    i += 1
                    continue
            else:
                # Not a failed trend either (edge case), move forward
                i += 1
    
    # Compare successful vs failed
    comparison = {
        'successful_count': len(successful_trends),
        'failed_count': len(failed_trends),
        'indicators_comparison': {}
    }
    
    indicators_to_compare = [
        'ADX', 'RSI', 'RVOL10', 
        'price_slope10_pct', 'price_slope20_pct', 'price_near_high_pct',
        'volume_10_avg', 'volume_slope10', 'volume_ratio'
    ]
    
    for indicator in indicators_to_compare:
        success_vals = [t['entry_indicators'].get(indicator) for t in successful_trends 
                       if t['entry_indicators'].get(indicator) is not None]
        failed_vals = [t['entry_indicators'].get(indicator) for t in failed_trends 
                      if t['entry_indicators'].get(indicator) is not None]
        
        if success_vals and failed_vals:
            comparison['indicators_comparison'][indicator] = {
                'successful_mean': float(np.mean(success_vals)),
                'failed_mean': float(np.mean(failed_vals)),
                'difference': float(np.mean(success_vals) - np.mean(failed_vals)),
                'difference_pct': float((np.mean(success_vals) - np.mean(failed_vals)) / abs(np.mean(failed_vals)) * 100) if np.mean(failed_vals) != 0 else 0,
            }
    
    return comparison, failed_trends

def analyze_by_segment(trends: List[Dict], segment_name: str, segment_mask: np.ndarray) -> Dict:
    """
    Analyze trends for a specific segment (e.g., by symbol or regime)
    """
    segment_trends = [trends[i] for i in range(len(trends)) if segment_mask[i]]
    
    if len(segment_trends) < 10:
        return None
    
    patterns = analyze_common_patterns(segment_trends)
    thresholds = find_optimal_thresholds(segment_trends)
    
    return {
        'segment_name': segment_name,
        'count': len(segment_trends),
        'patterns': patterns,
        'thresholds': thresholds,
    }

def calculate_expected_return_realistic(successful_trends: List[Dict], failed_trends: List[Dict],
                                       take_profit_pct: float = 5.0, stop_loss_pct: float = -3.0) -> Dict:
    """
    Calculate expected return using realistic exit logic (TP/SL first-touch).
    More realistic than assuming we always capture full MFE/MAE.
    
    For each trend (both successful and failed):
    - If price first touches +take_profit_pct → win of +take_profit_pct
    - If price first touches stop_loss_pct → loss of stop_loss_pct
    - Otherwise → close at end of window
    
    Returns:
        Dict with p (success probability), g_win, g_loss, expected_return
    """
    # Combine all trends and determine outcome for each
    all_trades = []
    
    for trend in successful_trends + failed_trends:
        mfe = trend.get('mfe_pct', 0)
        mae = trend.get('mae_pct', 0)
        
        # Determine which threshold was hit first
        hit_tp_first = (mfe >= take_profit_pct) and (mae > stop_loss_pct or mae == 0)
        hit_sl_first = (mae <= stop_loss_pct) and (mfe < take_profit_pct or mfe == 0)
        
        if hit_tp_first:
            # TP hit first
            result = take_profit_pct
            hit_tp = True
        elif hit_sl_first:
            # SL hit first
            result = stop_loss_pct
            hit_tp = False
        else:
            # Neither hit, use final result
            final_gain = trend.get('gain_pct', 0)
            if final_gain >= take_profit_pct:
                result = take_profit_pct
                hit_tp = True
            elif final_gain <= stop_loss_pct:
                result = stop_loss_pct
                hit_tp = False
            else:
                # Close at end of window
                result = final_gain
                hit_tp = (result > 0)
        
        all_trades.append({
            'result': result,
            'hit_tp_first': hit_tp,
        })
    
    if len(all_trades) == 0:
        return None
    
    total_trades = len(all_trades)
    
    # Calculate statistics
    winning_trades = [t for t in all_trades if t['hit_tp_first']]
    losing_trades = [t for t in all_trades if not t['hit_tp_first']]
    
    p = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    if winning_trades:
        g_win = float(np.mean([t['result'] for t in winning_trades]))
    else:
        g_win = 0.0
    
    if losing_trades:
        g_loss = float(np.mean([t['result'] for t in losing_trades]))
    else:
        g_loss = 0.0
    
    expected_return = p * g_win + (1 - p) * g_loss
    
    return {
        'success_probability': float(p),
        'avg_win_pct': float(g_win),
        'avg_loss_pct': float(g_loss),
        'expected_return_pct': float(expected_return),
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'take_profit_pct': take_profit_pct,
        'stop_loss_pct': stop_loss_pct,
        'note': 'Uses realistic first-touch TP/SL logic, not full MFE/MAE',
    }

def calculate_expected_return(successful_trends: List[Dict], failed_trends: List[Dict]) -> Dict:
    """
    Calculate expected return metric (optimistic version using full MFE/MAE).
    For comparison with realistic version.
    """
    total_trades = len(successful_trends) + len(failed_trends)
    if total_trades == 0:
        return None
    
    p = len(successful_trends) / total_trades
    
    # Average gain of successful trends (using MFE as the favorable outcome)
    if successful_trends:
        g_win = np.mean([t.get('mfe_pct', t.get('gain_pct', 0)) for t in successful_trends])
    else:
        g_win = 0.0
    
    # Average loss of failed trends (using MAE as the adverse outcome)
    if failed_trends:
        g_loss = np.mean([t.get('mae_pct', 0) for t in failed_trends])
    else:
        g_loss = 0.0
    
    expected_return = p * g_win + (1 - p) * g_loss
    
    return {
        'success_probability': float(p),
        'avg_win_pct': float(g_win),
        'avg_loss_pct': float(g_loss),
        'expected_return_pct': float(expected_return),
        'total_trades': total_trades,
        'winning_trades': len(successful_trends),
        'losing_trades': len(failed_trends),
        'note': 'Optimistic: assumes full MFE/MAE capture',
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze successful trends and find common patterns')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to analyze (e.g., BTC/USDT ETH/USDT)')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--train-start', type=str, default=None, help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, default=None, help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--validate-start', type=str, default=None, help='Validation start date (YYYY-MM-DD)')
    parser.add_argument('--validate-end', type=str, default=None, help='Validation end date (YYYY-MM-DD)')
    parser.add_argument('--min-gain', type=float, default=MIN_TREND_GAIN_PCT,
                       help=f'Minimum price gain to consider successful (default: {MIN_TREND_GAIN_PCT}%%)')
    parser.add_argument('--max-mae', type=float, default=MAX_MAE_PCT,
                       help=f'Maximum adverse excursion allowed (default: {MAX_MAE_PCT}%%)')
    parser.add_argument('--output', type=str, default='trend_patterns_analysis.json',
                       help='Output file for analysis results')
    parser.add_argument('--segment-by', choices=['symbol', 'regime', 'both', 'none'], default='both',
                       help='Segment analysis by symbol, regime, both, or none')
    
    args = parser.parse_args()
    
    # Get symbols and date range
    symbols = args.symbols or backtest.BACKTEST_SYMBOLS[:3]
    start_date = args.start or backtest.BACKTEST_START_DATE
    end_date = args.end or backtest.BACKTEST_END_DATE
    
    # Train/validate split
    train_start = args.train_start or start_date
    train_end = args.train_end or (args.validate_start if args.validate_start else end_date)
    validate_start = args.validate_start
    validate_end = args.validate_end or end_date
    
    print("=" * 80)
    print("TREND PATTERN ANALYSIS (Enhanced)")
    print("=" * 80)
    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Minimum gain: {args.min_gain}%")
    print(f"Maximum MAE: {args.max_mae}%")
    if validate_start:
        print(f"Train: {train_start} to {train_end}")
        print(f"Validate: {validate_start} to {validate_end}")
    print("=" * 80)
    print()
    
    # Fetch historical data
    print("Fetching historical data...")
    symbol_data = {}
    engine = backtest.BacktestEngine(1000.0)
    exchange = engine.exchange
    
    # Patch trader exchange
    import trader_v21 as trader
    original_trader_exchange = trader.exchange
    trader.exchange = exchange
    
    try:
        for symbol in symbols:
            print(f"  Loading {symbol}...")
            df = backtest.fetch_historical_data(symbol, backtest.BACKTEST_TIMEFRAME, start_date, end_date, exchange)
            if df is not None and len(df) > 200:
                print(f"    Preparing indicators for {len(df)} candles...")
                df = backtest.prepare_data_from_ohlcv(df)
                symbol_data[symbol] = df
                print(f"    Loaded {len(df)} candles with indicators")
            else:
                print(f"    Failed to load data")
        
        if not symbol_data:
            print("No data loaded. Exiting.")
            return
        
        # Split data for train/validate if specified
        train_data = {}
        validate_data = {}
        
        if validate_start:
            for symbol, df in symbol_data.items():
                train_mask = (df.index >= train_start) & (df.index < train_end)
                validate_mask = (df.index >= validate_start) & (df.index <= validate_end)
                train_data[symbol] = df[train_mask].copy()
                validate_data[symbol] = df[validate_mask].copy()
        else:
            train_data = symbol_data
            validate_data = {}
        
        print(f"\nAnalyzing trends in training data...")
        print(f"Looking for trends with gain >= {args.min_gain}% and MAE >= {args.max_mae}%")
        
        # Find successful trends
        successful_trends = find_successful_trends(train_data, min_gain_pct=args.min_gain, max_mae_pct=args.max_mae)
        
        print(f"\nFound {len(successful_trends)} successful trends (risk-aware)")
        
        if len(successful_trends) == 0:
            print("No successful trends found. Try lowering --min-gain or --max-mae or using different date range.")
            return
        
        # Analyze patterns
        print("\nAnalyzing common patterns...")
        patterns = analyze_common_patterns(successful_trends)
        
        # Find optimal thresholds (simple percentiles)
        print("Finding optimal thresholds (percentile method)...")
        thresholds = find_optimal_thresholds(successful_trends)
        
        # Compare with failed trends
        print("Comparing with failed trends...")
        comparison, failed_trends = compare_with_failed_trends(train_data, successful_trends, 
                                                               min_gain_pct=args.min_gain, max_mae_pct=args.max_mae)
        
        # Grid search for optimal ranges
        print("Finding optimal ranges (grid search method)...")
        indicators_for_grid = ['ADX', 'RSI', 'RVOL10', 'price_slope10_pct', 'volume_ratio']
        optimal_ranges = find_optimal_ranges_grid_search(successful_trends, failed_trends, indicators_for_grid)
        
        # Calculate expected return (both optimistic and realistic)
        print("Calculating expected return...")
        expected_return_optimistic = calculate_expected_return(successful_trends, failed_trends)
        expected_return_realistic = calculate_expected_return_realistic(successful_trends, failed_trends,
                                                                        take_profit_pct=args.min_gain,
                                                                        stop_loss_pct=args.max_mae)
        
        # Validation evaluation (if validation data available)
        validation_results = None
        if validate_data and len(validate_data) > 0:
            print("\nEvaluating on validation data...")
            successful_trends_val = find_successful_trends(validate_data, min_gain_pct=args.min_gain, max_mae_pct=args.max_mae)
            failed_trends_val = []
            _, failed_trends_val = compare_with_failed_trends(validate_data, successful_trends_val,
                                                               min_gain_pct=args.min_gain, max_mae_pct=args.max_mae)
            
            print(f"  Validation trends: {len(successful_trends_val)} successful, {len(failed_trends_val)} failed")
            
            # Evaluate optimal ranges on validation
            validation_results = evaluate_ranges_on_dataset(successful_trends_val, failed_trends_val, optimal_ranges)
            
            # Calculate expected return on validation
            expected_return_val_optimistic = calculate_expected_return(successful_trends_val, failed_trends_val)
            expected_return_val_realistic = calculate_expected_return_realistic(successful_trends_val, failed_trends_val,
                                                                               take_profit_pct=args.min_gain,
                                                                               stop_loss_pct=args.max_mae)
            validation_results['expected_return_optimistic'] = expected_return_val_optimistic
            validation_results['expected_return_realistic'] = expected_return_val_realistic
        
        # Segment analysis
        segment_analysis = {}
        if args.segment_by in ['symbol', 'both']:
            print("\nAnalyzing by symbol...")
            for symbol in symbols:
                symbol_mask = np.array([t['symbol'] == symbol for t in successful_trends])
                segment_result = analyze_by_segment(successful_trends, f"symbol_{symbol}", symbol_mask)
                if segment_result:
                    segment_analysis[f"symbol_{symbol}"] = segment_result
        
        if args.segment_by in ['regime', 'both']:
            print("Analyzing by regime (trending vs non-trending)...")
            trending_mask = np.array([t['lookback_indicators'].get('trending_up_flag', 0) == 1 for t in successful_trends])
            non_trending_mask = ~trending_mask
            
            segment_result = analyze_by_segment(successful_trends, "regime_trending", trending_mask)
            if segment_result:
                segment_analysis["regime_trending"] = segment_result
            
            segment_result = analyze_by_segment(successful_trends, "regime_non_trending", non_trending_mask)
            if segment_result:
                segment_analysis["regime_non_trending"] = segment_result
        
        # Print results
        print("\n" + "=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        
        print(f"\nTotal Successful Trends: {patterns['total_trends']}")
        print(f"Average Gain (MFE): {patterns['gain_statistics']['avg_gain_pct']:.2f}%")
        print(f"Median Gain: {patterns['gain_statistics']['median_gain_pct']:.2f}%")
        print(f"Gain Range: {patterns['gain_statistics']['min_gain_pct']:.2f}% to {patterns['gain_statistics']['max_gain_pct']:.2f}%")
        
        print("\n" + "-" * 80)
        print("RISK STATISTICS (MAE/MFE)")
        print("-" * 80)
        print(f"Average MFE: {patterns['risk_statistics']['avg_mfe_pct']:.2f}%")
        print(f"Average MAE: {patterns['risk_statistics']['avg_mae_pct']:.2f}%")
        print(f"Worst MAE: {patterns['risk_statistics']['worst_mae_pct']:.2f}%")
        print(f"→ Your stops should be set around {abs(patterns['risk_statistics']['worst_mae_pct']):.2f}% to avoid being stopped out")
        
        print("\n" + "-" * 80)
        print("EXPECTED RETURN")
        print("-" * 80)
        print("OPTIMISTIC (assumes full MFE/MAE capture):")
        if expected_return_optimistic:
            print(f"  Success Probability: {expected_return_optimistic['success_probability']*100:.1f}%")
            print(f"  Average Win: {expected_return_optimistic['avg_win_pct']:.2f}%")
            print(f"  Average Loss: {expected_return_optimistic['avg_loss_pct']:.2f}%")
            print(f"  Expected Return: {expected_return_optimistic['expected_return_pct']:.2f}% per trade")
            print(f"  Total Trades: {expected_return_optimistic['total_trades']} (Wins: {expected_return_optimistic['winning_trades']}, Losses: {expected_return_optimistic['losing_trades']})")
        
        print("\nREALISTIC (first-touch TP/SL logic):")
        if expected_return_realistic:
            print(f"  Success Probability: {expected_return_realistic['success_probability']*100:.1f}%")
            print(f"  Average Win: {expected_return_realistic['avg_win_pct']:.2f}%")
            print(f"  Average Loss: {expected_return_realistic['avg_loss_pct']:.2f}%")
            print(f"  Expected Return: {expected_return_realistic['expected_return_pct']:.2f}% per trade")
            print(f"  Total Trades: {expected_return_realistic['total_trades']} (Wins: {expected_return_realistic['winning_trades']}, Losses: {expected_return_realistic['losing_trades']})")
            print(f"  TP: {expected_return_realistic['take_profit_pct']:.1f}%, SL: {expected_return_realistic['stop_loss_pct']:.1f}%")
        
        print("\n" + "-" * 80)
        print("OPTIMAL RANGES (Grid Search Method)")
        print("-" * 80)
        print("These ranges maximize precision * coverage:")
        print("⚠️  Note: Precision shows improvement over baseline, not absolute success rate")
        for indicator, range_info in sorted(optimal_ranges.items()):
            baseline = range_info.get('baseline', 0)
            improvement = range_info.get('improvement_factor', 1.0)
            print(f"\n{indicator}:")
            print(f"  Range: {range_info['low']:.3f} to {range_info['high']:.3f}")
            print(f"  Precision: {range_info['precision']*100:.3f}% (success rate in this range)")
            print(f"  Baseline: {baseline*100:.3f}% (overall success rate)")
            print(f"  Improvement: {improvement:.2f}x over baseline")
            print(f"  Coverage: {range_info['coverage']*100:.1f}% (of all successful trends)")
            print(f"  Score: {range_info['score']:.3f}")
            print(f"  Samples: {range_info['n_samples']} (Success: {range_info['n_success']})")
        
        print("\n" + "-" * 80)
        print("SUCCESSFUL vs FAILED TRENDS COMPARISON")
        print("-" * 80)
        print(f"Successful: {comparison['successful_count']} trends")
        print(f"Failed: {comparison['failed_count']} trends")
        
        for indicator, comp in sorted(comparison['indicators_comparison'].items()):
            print(f"\n{indicator}:")
            print(f"  Successful mean: {comp['successful_mean']:.3f}")
            print(f"  Failed mean: {comp['failed_mean']:.3f}")
            print(f"  Difference: {comp['difference']:+.3f} ({comp['difference_pct']:+.1f}%)")
            if comp['difference'] > 0:
                print(f"  → Successful trends had HIGHER {indicator}")
            else:
                print(f"  → Successful trends had LOWER {indicator}")
        
        # Segment analysis results
        if segment_analysis:
            print("\n" + "=" * 80)
            print("SEGMENT ANALYSIS")
            print("=" * 80)
            for segment_name, segment_data in segment_analysis.items():
                print(f"\n{segment_name}:")
                print(f"  Count: {segment_data['count']} trends")
                if 'thresholds' in segment_data and segment_data['thresholds']:
                    print("  Optimal Thresholds:")
                    for ind, thresh in list(segment_data['thresholds'].items())[:5]:
                        print(f"    {ind}: {thresh.get('min', 0):.3f} to {thresh.get('max', 0):.3f}")
        
        # Validation results
        if validation_results:
            print("\n" + "=" * 80)
            print("VALIDATION RESULTS")
            print("=" * 80)
            print(f"Baseline (validation): {validation_results['baseline']*100:.3f}%")
            print(f"Total trades (validation): {validation_results['total_trades']}")
            print(f"Successful trades: {validation_results['successful_trades']}")
            
            print("\n" + "-" * 80)
            print("RANGE PERFORMANCE ON VALIDATION")
            print("-" * 80)
            print("⚠️  Compare improvement_factor_val with improvement_factor_train")
            print("   If they're similar, the range generalizes well")
            print("   If validation is much worse, possible overfitting")
            
            for indicator, val_result in sorted(validation_results['indicator_results'].items()):
                print(f"\n{indicator}:")
                print(f"  Range: {val_result['range']['low']:.3f} to {val_result['range']['high']:.3f}")
                print(f"  Precision (validation): {val_result['precision_val']*100:.3f}%")
                print(f"  Baseline (validation): {val_result['baseline_val']*100:.3f}%")
                print(f"  Improvement (validation): {val_result['improvement_factor_val']:.2f}x")
                print(f"  Improvement (training): {val_result['improvement_factor_train']:.2f}x")
                print(f"  Generalization OK: {val_result['generalization_ok']}")
                print(f"  Samples: {val_result['n_samples_val']} (Success: {val_result['n_success_val']})")
            
            print("\n" + "-" * 80)
            print("EXPECTED RETURN ON VALIDATION")
            print("-" * 80)
            if validation_results.get('expected_return_realistic'):
                er_val = validation_results['expected_return_realistic']
                print(f"  Success Probability: {er_val['success_probability']*100:.1f}%")
                print(f"  Expected Return (realistic): {er_val['expected_return_pct']:.2f}% per trade")
                print(f"  Total Trades: {er_val['total_trades']} (Wins: {er_val['winning_trades']}, Losses: {er_val['losing_trades']})")
                
                # Compare with training
                if expected_return_realistic:
                    print(f"\n  Comparison with Training:")
                    print(f"    Training ER: {expected_return_realistic['expected_return_pct']:.2f}%")
                    print(f"    Validation ER: {er_val['expected_return_pct']:.2f}%")
                    er_diff = er_val['expected_return_pct'] - expected_return_realistic['expected_return_pct']
                    print(f"    Difference: {er_diff:+.2f}%")
                    if abs(er_diff) < 1.0:
                        print(f"    → Generalization looks good (similar ER)")
                    else:
                        print(f"    → Warning: Significant difference, possible overfitting")
        
        # Save results
        results = {
            'analysis_date': datetime.now().isoformat(),
            'config': {
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date,
                'train_start': train_start,
                'train_end': train_end,
                'validate_start': validate_start,
                'validate_end': validate_end,
                'min_gain_pct': args.min_gain,
                'max_mae_pct': args.max_mae,
            },
            'successful_trends_count': len(successful_trends),
            'patterns': patterns,
            'optimal_thresholds': thresholds,
            'optimal_ranges': optimal_ranges,
            'comparison': comparison,
            'expected_return_optimistic': expected_return_optimistic,
            'expected_return_realistic': expected_return_realistic,
            'segment_analysis': segment_analysis,
            'validation_results': validation_results,
            'sample_trends': successful_trends[:10],  # Save first 10 as examples
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n\nResults saved to {args.output}")
        print("=" * 80)
        
    finally:
        # Restore original exchange
        trader.exchange = original_trader_exchange

if __name__ == "__main__":
    main()
