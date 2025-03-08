import json
import os
from datetime import datetime
import time
import logging
import ccxt
import pandas as pd
import numpy as np
import random
import pandas_ta as ta
from scipy.stats import linregress

# Configuración de logging
logging.basicConfig(filename='indicator_optimizer.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración inicial relajada (basada en tu lógica actual)
CONFIG = {
    "relative_volume_threshold": 2.0,
    "support_near_threshold": 0.1,
    "depth_threshold": 3000,
    "adx_threshold": 20,
    "rsi_range": [60, 80],
    "signals_score_threshold": 9,
    "spread_threshold_multiplier": 0.005
}

OPTIMIZED_CONFIG_FILE = "optimized_config.json"

def save_config(config):
    with open(OPTIMIZED_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def load_config():
    global CONFIG
    if os.path.exists(OPTIMIZED_CONFIG_FILE):
        with open(OPTIMIZED_CONFIG_FILE, 'r') as f:
            CONFIG = json.load(f)

def fetch_market_data(symbol, exchange):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=50)
        order_book = exchange.fetch_order_book(symbol, limit=10)
        return {
            "ohlcv": ohlcv,
            "depth": sum([b[1] for b in order_book['bids']]) * ohlcv[-1][4],
            "spread": order_book['asks'][0][0] - order_book['bids'][0][0],
            "imbalance": sum([b[1] for b in order_book['bids']]) / sum([a[1] for a in order_book['asks']]) if sum([a[1] for a in order_book['asks']]) > 0 else 1.0
        }
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

def detect_support_level(price_series, window=15, max_threshold_multiplier=3.0):
    if len(price_series) < window:
        return None
    recent_prices = price_series[-window:]
    local_mins = [
        recent_prices[i] for i in range(1, len(recent_prices) - 1)
        if recent_prices[i] < recent_prices[i - 1] and recent_prices[i] < recent_prices[i + 1]
    ]
    if not local_mins:
        return None
    support_level = min(local_mins)
    current_price = price_series[-1]
    threshold = 1 + (current_price * 0.02 * max_threshold_multiplier) / current_price
    threshold = min(threshold, 1.03)
    return support_level if current_price <= support_level * threshold else None

def calculate_short_volume_trend(volume_series, window=3):
    if len(volume_series) < window:
        return "insufficient_data"
    last_volume = volume_series[-1]
    avg_volume = np.mean(volume_series[-window:])
    if last_volume > avg_volume * 1.05:
        return "increasing"
    elif last_volume < avg_volume * 0.95:
        return "decreasing"
    else:
        return "stable"

def calculate_indicators(ohlcv, order_book_data):
    if not ohlcv or len(ohlcv) < 14:
        return None
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Indicadores usando pandas_ta
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['ROC'] = ta.roc(df['close'], length=7)

    current_price = df['close'].iloc[-1]
    volume_series = df['volume'].values
    relative_volume = volume_series[-1] / np.mean(volume_series[-10:]) if len(volume_series) >= 10 and np.mean(volume_series[-10:]) > 0 else 1.0
    roc = df['ROC'].iloc[-1] if not pd.isna(df['ROC'].iloc[-1]) else 0
    short_volume_trend = calculate_short_volume_trend(volume_series)

    # Tendencia de precio
    last_10_price = df['close'].values[-10:]
    if len(last_10_price) >= 10:
        slope_price, _, _, _, _ = linregress(range(10), last_10_price)
        price_trend = "increasing" if slope_price > 0.01 else "decreasing" if slope_price < -0.01 else "stable"
    else:
        price_trend = "insufficient_data"

    support_level = detect_support_level(df['close'].values)
    support_distance = (current_price - support_level) / support_level if support_level and current_price > 0 else 0

    has_crossover = df['MACD'].iloc[-1] > df['MACD_signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_signal'].iloc[-2]

    return {
        "rsi": df['RSI'].iloc[-1],
        "relative_volume": relative_volume,
        "roc": roc,
        "short_volume_trend": short_volume_trend,
        "price_trend": price_trend,
        "depth": order_book_data["depth"],
        "spread": order_book_data["spread"],
        "current_price": current_price,
        "support_level": support_level,
        "support_distance": support_distance,
        "adx": df['ADX'].iloc[-1],
        "has_macd_crossover": has_crossover
    }

def calculate_adaptive_strategy(indicators, config):
    rsi = indicators.get('rsi', None)
    relative_volume = indicators.get('relative_volume', None)
    roc = indicators.get('roc', None)
    short_volume_trend = indicators.get('short_volume_trend', 'insufficient_data')
    price_trend = indicators.get('price_trend', 'insufficient_data')
    depth = indicators.get('depth', 0)
    spread = indicators.get('spread', float('inf'))
    current_price = indicators.get('current_price', 0)
    support_distance = indicators.get('support_distance', float('inf'))
    adx = indicators.get('adx', None)
    has_macd_crossover = indicators.get('has_macd_crossover', False)

    if adx is None or adx < config["adx_threshold"]:
        return "mantener", 50, f"Tendencia débil (ADX < {config['adx_threshold']})"
    if support_distance > config["support_near_threshold"]:
        return "mantener", 50, f"Lejos del soporte (distancia: {support_distance:.2%})"
    if rsi is not None and (rsi < config["rsi_range"][0] or rsi > config["rsi_range"][1]):
        return "mantener", 50, f"RSI fuera del rango ({config['rsi_range'][0]}-{config['rsi_range'][1]}): {rsi}"

    weighted_signals = [
        4 * (relative_volume > config["relative_volume_threshold"]),
        3 * (short_volume_trend == "increasing"),
        2 * (price_trend == "increasing"),
        2 * (roc > 1.0),
        1 * (depth >= config["depth_threshold"]),
        1 * (spread <= config["spread_threshold_multiplier"] * current_price),
        1 * (support_distance <= config["support_near_threshold"]),
        1 * (rsi is not None and config["rsi_range"][0] <= rsi <= config["rsi_range"][1]) if rsi else 0
    ]
    signals_score = sum(weighted_signals)

    base_confidence = 50
    if signals_score >= config["signals_score_threshold"] and adx > config["adx_threshold"]:
        base_confidence = 70
        if has_macd_crossover:
            base_confidence = 90
        elif rsi and config["rsi_range"][0] <= rsi <= config["rsi_range"][1]:
            base_confidence = 85

    if base_confidence >= 70:
        return "comprar", base_confidence, f"Compra fuerte: Volumen relativo > {config['relative_volume_threshold']}, puntaje {signals_score}/14, ADX > {config['adx_threshold']}, {'con cruce MACD' if has_macd_crossover else 'sin cruce MACD'}, cerca del soporte, RSI en rango {config['rsi_range']}"
    return "mantener", 50, f"Condiciones insuficientes para comprar (puntaje: {signals_score}/14)"

def calculate_metrics(trades):
    completed_trades = [t for t in trades if "return" in t and t["return"] is not None]
    if not completed_trades:
        return 0.0, 0.0, 0
    successes = sum(1 for t in completed_trades if t["return"] > 1.0)
    total = len(completed_trades)
    success_rate = (successes / total * 100) if total > 0 else 0
    avg_return = sum(t["return"] for t in completed_trades) / total if total > 0 else 0
    return success_rate, avg_return, total

def adjust_indicators(trades, success_rate, avg_return, total_trades, config):
    adjustments = {}
    reasoning = []

    limits = {
        "relative_volume_threshold": (1.5, 5.0),
        "support_near_threshold": (0.03, 0.15),
        "depth_threshold": (3000, 10000),
        "adx_threshold": (20, 40),
        "signals_score_threshold": (8, 12)
    }

    if avg_return <= 0 and total_trades > 5:
        if config["relative_volume_threshold"] > limits["relative_volume_threshold"][0]:
            adjustments["relative_volume_threshold"] = max(config["relative_volume_threshold"] - 0.5, limits["relative_volume_threshold"][0])
            reasoning.append("Reduje relative_volume_threshold para capturar más trades y lograr ganancias.")
        if config["support_near_threshold"] < limits["support_near_threshold"][1]:
            adjustments["support_near_threshold"] = min(config["support_near_threshold"] + 0.02, limits["support_near_threshold"][1])
            reasoning.append("Aumenté support_near_threshold para permitir trades más lejos del soporte.")
        if config["depth_threshold"] > limits["depth_threshold"][0]:
            adjustments["depth_threshold"] = max(config["depth_threshold"] - 500, limits["depth_threshold"][0])
            reasoning.append("Reduje depth_threshold para incluir activos con menor liquidez.")
    elif avg_return > 0:
        if success_rate < 50 and total_trades > 10:
            if config["relative_volume_threshold"] < limits["relative_volume_threshold"][1]:
                adjustments["relative_volume_threshold"] = min(config["relative_volume_threshold"] + 0.5, limits["relative_volume_threshold"][1])
                reasoning.append("Aumenté relative_volume_threshold para filtrar señales débiles y mejorar % de éxito.")
            if config["support_near_threshold"] > limits["support_near_threshold"][0]:
                adjustments["support_near_threshold"] = max(config["support_near_threshold"] - 0.02, limits["support_near_threshold"][0])
                reasoning.append("Reduje support_near_threshold para priorizar trades más cerca del soporte.")
        elif success_rate >= 50:
            indicator_to_adjust = random.choice(list(limits.keys()))
            current_value = config[indicator_to_adjust]
            min_val, max_val = limits[indicator_to_adjust]
            adjustment = random.uniform(-0.1 * current_value, 0.1 * current_value)
            new_value = max(min_val, min(max_val, current_value + adjustment))
            adjustments[indicator_to_adjust] = round(new_value, 2)
            reasoning.append(f"Ajusté {indicator_to_adjust} a {new_value:.2f} para explorar mejoras en el % de éxito.")

    return adjustments, " ".join(reasoning)

def simulate_trade(symbol, exchange, config):
    market_data = fetch_market_data(symbol, exchange)
    if market_data and market_data["ohlcv"]:
        indicators = calculate_indicators(market_data["ohlcv"], market_data)
        if indicators:
            action, confidence, explanation = calculate_adaptive_strategy(indicators, config)
            if action == "comprar" and confidence >= 70:
                # Imprimir compra falsa en pantalla
                print(f"\n=== Compra Falsa Simulada ===\n"
                      f"Símbolo: {symbol}\n"
                      f"Precio de Compra: {indicators['current_price']:.4f} USDT\n"
                      f"Confianza: {confidence}%\n"
                      f"Explicación: {explanation}\n")
                logging.info(f"Simulando compra para {symbol} - Precio: {indicators['current_price']:.4f}, Confianza: {confidence}%, Explicación: {explanation}")
                
                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "action": action,
                    "confidence": confidence,
                    "explanation": explanation,
                    "indicators": indicators,
                    "price_at_trade": indicators["current_price"]
                }
                # Simular resultado después de 1 hora
                time.sleep(3600)
                future_price = exchange.fetch_ticker(symbol)["last"]
                trade_record["outcome"] = "success" if future_price > indicators["current_price"] * 1.01 else "failure"
                trade_record["price_after_1h"] = future_price
                trade_record["return"] = (future_price - indicators["current_price"]) / indicators["current_price"] * 100
                
                # Imprimir venta falsa en pantalla
                result = "Éxito" if trade_record["outcome"] == "success" else "Fracaso"
                print(f"\n=== Venta Falsa Simulada ===\n"
                      f"Símbolo: {symbol}\n"
                      f"Precio de Compra: {indicators['current_price']:.4f} USDT\n"
                      f"Precio de Venta: {future_price:.4f} USDT\n"
                      f"Retorno: {trade_record['return']:.2f}%\n"
                      f"Resultado: {result}\n")
                logging.info(f"Simulando venta para {symbol} - Precio de Venta: {future_price:.4f}, Retorno: {trade_record['return']:.2f}%, Resultado: {result}")
                
                return trade_record
    return None

def indicator_optimizer():
    exchange = ccxt.binance()  # Modo público
    high_volume_symbols = ["PROS/USDT", "BNX/USDT", "ENA/USDT", "AUCTION/USDT"]
    config = CONFIG.copy()
    completed_trades = []

    while True:
        # Fase de simulación
        for symbol in high_volume_symbols:
            trade = simulate_trade(symbol, exchange, config)
            if trade:
                completed_trades.append(trade)

        # Fase de optimización
        if len(completed_trades) >= 5:
            success_rate, avg_return, total_trades = calculate_metrics(completed_trades)
            
            # Mostrar indicadores y métricas en tiempo real
            log_message = f"\n=== Indicadores Actualizados ===\n{json.dumps(config, indent=4)}\n"
            log_message += f"% de Éxito: {success_rate:.2f}%\nRetorno Promedio: {avg_return:.2f}%\nTrades Completados: {total_trades}\n"
            logging.info(log_message)
            print(log_message)

            # Ajustar indicadores
            adjustments, reasoning = adjust_indicators(completed_trades, success_rate, avg_return, total_trades, config)
            if adjustments:
                config.update(adjustments)
                save_config(config)
                adjustment_message = f"Ajustes realizados: {json.dumps(adjustments, indent=4)}\nRazón: {reasoning}\n"
                logging.info(adjustment_message)
                print(adjustment_message)

            # Limpiar trades completados para evitar acumulación
            completed_trades = completed_trades[-5:]  # Mantener los últimos 5 para referencia

        time.sleep(3600)  # Ciclo cada hora

if __name__ == "__main__":
    indicator_optimizer()