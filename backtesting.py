import ccxt
import os
import pandas as pd
import time
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import csv
from itertools import product
import tensorflow as tf

# Configurar TensorFlow para usar la GPU AMD
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU AMD detectada y configurada correctamente.")
else:
    print("No se detectó ninguna GPU.")

def fetch_historical_data(exchange, symbol, timeframe='15m', limit=96):
    """
    Fetch historical OHLCV data for a given symbol and timeframe.
    Returns a DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        data = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()  # Retorna un DataFrame vacío en caso de error

def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for the given data.
    """
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['rsi'] = rsi
    return data

def calculate_factors(data):
    """
    Calculate factors like distance_factor, volume_growth, price_change, trend_factor, and RSI.
    Adds these as new columns to the DataFrame.
    """
    data['distance_factor'] = data['close'] / data['low']
    data['volume_growth'] = data['volume'].pct_change().fillna(0)
    data['price_change'] = data['close'].pct_change().fillna(0)

    short_ma = data['close'].rolling(window=7).mean()
    long_ma = data['close'].rolling(window=25).mean()
    trend_condition = (short_ma > long_ma)
    data['trend_factor'] = trend_condition.astype(float)

    # Add RSI calculation
    data = calculate_rsi(data)

    # Replace NaN RSI values with 50 (neutral)
    data['rsi'] = data['rsi'].fillna(50)

    return data

def backtest_strategy(exchange, symbol, timeframe='15m', limit=96, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Perform backtesting on a single symbol with specified parameters.
    """
    print(f"\nFetching historical data for {symbol}...")
    data = fetch_historical_data(exchange, symbol, timeframe, limit)

    if data.empty:
        print(f"No data fetched for {symbol}. Skipping backtest.")
        return pd.DataFrame()

    print("Calculating factors...")
    data = calculate_factors(data)

    # Scoring and backtesting
    print("Backtesting...")
    initial_capital = 1000
    capital = initial_capital
    portfolio = 0
    history = []

    for index, row in data.iterrows():
        # Calculate score
        score = (
            weights[0] * row['distance_factor'] +
            weights[1] * row['volume_growth'] +
            weights[2] * row['price_change'] +
            weights[3] * row['trend_factor']
        )

        # Buy signal
        if score > 0.8 and capital > 0:
            portfolio += capital / row['close']
            capital = 0
            action = 'BUY'
        # Sell signal
        elif score < 0.2 and portfolio > 0:
            if row['rsi'] > 70:
                capital += portfolio * row['close']
                portfolio = 0
                action = 'SELL'
            else:
                action = 'HOLD'
        else:
            action = 'HOLD'

        history.append({
            "timestamp": row['timestamp'],
            "capital": capital,
            "portfolio": portfolio * row['close'],
            "total": capital + portfolio * row['close'],
            "score": score,
            "action": action
        })

    results = pd.DataFrame(history)
    results['return'] = results['total'].pct_change().fillna(0)
    total_return = (results['total'].iloc[-1] / initial_capital - 1) * 100

    print(f"Final capital for {symbol}: {results['total'].iloc[-1]:.2f} USDT")
    print(f"Total return for {symbol}: {total_return:.2f}%")

    return results

def evaluate_weights(exchange, symbol, timeframe, limit, weights):
    """
    Evaluate a specific set of weights by running a backtest.
    """
    results = backtest_strategy(exchange, symbol, timeframe, limit, weights)
    if results.empty:
        return weights, -np.inf  # Penalize if backtest failed
    final_capital = results['total'].iloc[-1]
    total_return = (final_capital / 1000 - 1) * 100
    return weights, total_return

def optimize_weights_parallel(exchange, symbol, timeframe='15m', limit=96, n_jobs=-1):
    """
    Optimize the weights for the scoring function using grid search with parallel processing.
    """
    weight_ranges = np.arange(0.0, 1.1, 0.1)
    combinations = [
        weights for weights in product(weight_ranges, repeat=4)
        if round(sum(weights), 1) == 1.0
    ]

    print(f"Testing {len(combinations)} combinations of weights for {symbol}...")

    # Parallel evaluation of weights
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_weights)(exchange, symbol, timeframe, limit, weights)
        for weights in combinations
    )

    # Filter out failed backtests
    results = [res for res in results if res[1] != -np.inf]

    if not results:
        print(f"No successful backtests para {symbol}.")
        return None, None, []

    # Find the best weights
    best_weights, best_return = max(results, key=lambda x: x[1])

    print(f"Mejores pesos para {symbol}: {best_weights}, Retorno Total: {best_return:.2f}%")

    # Return the best weights, best return, and all results
    return best_weights, best_return, results

def plot_results(results, symbol):
    """
    Plot returns for different weight combinations.
    """
    if not results:
        print(f"No results to plot para {symbol}.")
        return

    weights, returns = zip(*results)
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(returns)), returns, color='skyblue', alpha=0.6)
    plt.xlabel('Combinación de Pesos')
    plt.ylabel('Retorno Total (%)')
    plt.title(f'Performance de Combinaciones de Pesos para {symbol}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'performance_weights_{symbol.replace("/", "_")}.png')
    plt.close()
    print(f"Gráfico guardado como performance_weights_{symbol.replace('/', '_')}.png")

def leer_ganancias(archivo):
    """
    Leer y mostrar las ganancias desde un archivo CSV.
    """
    try:
        df = pd.read_csv(archivo)
        print("\n--- Ganancias por Moneda ---")
        for index, row in df.iterrows():
            symbol = row['symbol']
            best_weights = (row['best_w1'], row['best_w2'], row['best_w3'], row['best_w4'])
            total_return = row['total_return']
            print(f"Moneda: {symbol}, Mejor Weights: {best_weights}, Ganancia: {total_return:.2f}%")
    except Exception as e:
        print(f"Error leyendo el archivo {archivo}: {e}")

def save_best_weights(best_weights_list, filename="best_weights.csv"):
    """
    Guardar los mejores pesos y retornos en un archivo CSV.
    """
    df = pd.DataFrame(best_weights_list)
    df.to_csv(filename, index=False)
    print(f"Mejores pesos guardados en {filename}")

if __name__ == "__main__":
    # Configuración del Exchange
    exchange = ccxt.binance({
        "enableRateLimit": True
    })

    # Configura tus claves API en las variables de entorno
    exchange.apiKey = os.getenv("BINANCE_API_KEY_REAL")
    exchange.secret = os.getenv("BINANCE_SECRET_KEY_REAL")

    # Lista de símbolos de criptomonedas
    symbols = ["ONT/USDT", "PROM/USDT", "FIRO/USDT"]

    # Ajusta el límite para cubrir 24 horas (timeframe de 15 minutos)
    limit = 96  # 96 * 15m = 24 horas

    best_weights_list = []

    for symbol in symbols:
        print(f"\n=== Optimización de Pesos para {symbol} ===")
        best_weights, best_return, all_results = optimize_weights_parallel(
            exchange, symbol, timeframe='15m', limit=limit
        )

        if best_weights is None:
            print(f"Optimización fallida para {symbol}.")
            continue

        # Almacenar los mejores pesos y retorno para cada símbolo
        best_weights_list.append({
            "symbol": symbol,
            "best_w1": best_weights[0],
            "best_w2": best_weights[1],
            "best_w3": best_weights[2],
            "best_w4": best_weights[3],
            "total_return": best_return
        })

        print(f"\nEjecutando backtest final para {symbol} con los mejores pesos...")
        final_results = backtest_strategy(
            exchange, symbol, timeframe='15m', limit=limit, weights=best_weights
        )

        # Guardar resultados a un archivo CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_results.to_csv(f"backtest_results_{symbol.replace('/', '_')}_{timestamp}.csv", index=False)
        print(f"Resultados de backtest guardados en backtest_results_{symbol.replace('/', '_')}_{timestamp}.csv")

        # Plot de todos los resultados
        plot_results(all_results, symbol)

    # Guardar los mejores pesos y retornos para cada símbolo a un archivo CSV
    save_best_weights(best_weights_list)

    # Leer y mostrar las ganancias desde el archivo CSV
    leer_ganancias("best_weights.csv")

    print("\nOptimización y backtesting completados para todas las monedas. Resultados guardados en archivos CSV.")
