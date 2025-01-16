import math
import time
import pandas as pd
import ta  # Technical Analysis library
import ccxt
import matplotlib.pyplot as plt

# Definir los pares de criptomonedas
symbols = [
    "COW/USDT", "JST/USDT", "USUAL/USDT", "LTC/USDT", "SAGA/USDT", "CGPT/USDT", "COOKIE/USDT"
]

# Definir los pesos de los indicadores
weight_ma_crossover = 0.4
weight_volume = 0.3
weight_rsi = 0.3

# Comisión ajustada
transaction_fee = 0.0005  # 0.05%

# Función para obtener datos históricos
def fetch_historical_data(symbols, timeframe="1h", limit=200, exchange_name="binance"):
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({"rateLimit": 1200, "enableRateLimit": True})
        historical_data = {}

        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                historical_data[symbol] = ohlcv
                print(f"Data fetched for {symbol} ({len(ohlcv)} candles).")
                time.sleep(exchange.rateLimit / 1000)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue

        return historical_data
    except Exception as e:
        print(f"Error initializing exchange: {e}")
        return {}

# Función de backtesting
def backtest_strategy(
    historical_data,
    weight_ma_crossover=0.4,
    weight_volume=0.3,
    weight_rsi=0.3,
    initial_balance=1000,
    trailing_stop_percentage=0.4,
    rsi_threshold=80,
    use_combined_strategy=True
):
    balance = initial_balance
    portfolio = {}
    trades = []
    portfolio_max_price = {}

    for symbol, ohlcv in historical_data.items():
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Calcular indicadores
        df["MA_7"] = ta.trend.SMAIndicator(df["close"], window=7).sma_indicator()
        df["MA_25"] = ta.trend.SMAIndicator(df["close"], window=25).sma_indicator()
        df["MA_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
        df["MA_200"] = ta.trend.SMAIndicator(df["close"], window=200).sma_indicator()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        df["Volume_Avg"] = df["volume"].rolling(window=20).mean()

        for i in range(50, len(df)):
            row = df.iloc[i]
            last_price = row["close"]

            # Señales
            ma_crossover = row["MA_7"] > row["MA_25"]
            rsi_signal = row["RSI"] < rsi_threshold
            volume_signal = row["volume"] > 1.3 * row["Volume_Avg"]
            long_term_trend = row["MA_50"] > row["MA_200"]

            # Score
            score = (
                weight_ma_crossover * ma_crossover
                + weight_volume * volume_signal
                + weight_rsi * rsi_signal
            )

            # Compra
            if (
                score > 0.9
                and ma_crossover
                and rsi_signal
                and volume_signal
                and long_term_trend
            ):
                if symbol not in portfolio:
                    allocation = min((score / 1.0) * (balance * 0.1), balance * 0.1)
                    amount = allocation / last_price
                    cost = amount * last_price * (1 + transaction_fee)
                    if cost <= balance:
                        portfolio[symbol] = {
                            "amount": amount,
                            "buy_price": last_price
                        }
                        balance -= cost
                        portfolio_max_price[symbol] = last_price
                        trades.append(
                            {
                                "type": "buy",
                                "symbol": symbol,
                                "price": last_price,
                                "amount": amount,
                                "timestamp": row["timestamp"],
                            }
                        )

            # Venta basada en estrategia combinada
            if symbol in portfolio:
                max_price = max(row["high"], portfolio_max_price.get(symbol, row["high"]))
                trailing_stop = max_price * (1 - trailing_stop_percentage)
                portfolio_max_price[symbol] = max_price

                if use_combined_strategy and (row["RSI"] > rsi_threshold or last_price < trailing_stop):
                    # Venta basada en RSI o Trailing Stop
                    amount = portfolio[symbol]["amount"]
                    revenue = amount * last_price * (1 - transaction_fee)
                    balance += revenue
                    trades.append(
                        {
                            "type": "sell",
                            "symbol": symbol,
                            "price": last_price,
                            "amount": amount,
                            "timestamp": row["timestamp"],
                        }
                    )
                    del portfolio[symbol]

    total_value = balance + sum(
        portfolio[symbol]["amount"] * df.iloc[-1]["close"] for symbol in portfolio
    )
    total_return = (total_value - initial_balance) / initial_balance

    return {
        "initial_balance": initial_balance,
        "final_balance": balance,
        "portfolio_value": total_value - balance,
        "total_value": total_value,
        "total_return": total_return,
        "trades": trades,
    }

# Programa Principal
if __name__ == "__main__":
    # Obtener datos históricos
    historical_data = fetch_historical_data(
        symbols=symbols, timeframe="1h", limit=500, exchange_name="binance"
    )

    # Verificar los datos obtenidos
    for symbol, data in historical_data.items():
        print(f"{symbol}: {len(data)} velas descargadas.")

    # Ejecutar estrategia combinada
    print("\nTesting combined strategy with RSI > 80 and Trailing Stop = 40%")
    results_combined = backtest_strategy(
        historical_data,
        weight_ma_crossover=weight_ma_crossover,
        weight_volume=weight_volume,
        weight_rsi=weight_rsi,
        trailing_stop_percentage=0.4,
        rsi_threshold=80,
        use_combined_strategy=True,
    )

    print("\nResultados de la estrategia combinada:")
    print(f"Saldo inicial: {results_combined['initial_balance']} USDT")
    print(f"Saldo final: {results_combined['final_balance']} USDT")
    print(f"Valor del portafolio: {results_combined['portfolio_value']} USDT")
    print(f"Valor total: {results_combined['total_value']} USDT")
    print(f"Retorno total: {results_combined['total_return'] * 100:.2f}%")

    # Imprimir operaciones realizadas
    print("\nOperaciones realizadas:")
    for trade in results_combined["trades"]:
        print(
            f"{trade['timestamp']} - {trade['type'].upper()} - {trade['symbol']} - "
            f"Precio: {trade['price']} - Cantidad: {trade['amount']}"
        )
