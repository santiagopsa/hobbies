import ccxt
import pandas as pd
import numpy as np
import time
from ta.trend import ADXIndicator
from ta.momentum import RSIIndicator
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Configuración del exchange para Binance global
exchange = ccxt.binance({
    "rateLimit": 1200,
    "enableRateLimit": True,
    "apiKey": os.getenv("BINANCE_API_KEY_REAL"),
    "secret": os.getenv("BINANCE_SECRET_KEY_REAL"),
    'options': {'defaultType': 'spot'}
})

def fetch_ohlcv_safe(symbol, timeframe='1h', limit=48):
    """
    Obtiene datos OHLCV de manera segura para un símbolo.
    
    Args:
        symbol (str): Símbolo del par (e.g., "BTC/USDT").
        timeframe (str): Marco temporal (default: '1h').
        limit (int): Número de velas (default: 48 horas).
    
    Returns:
        pd.DataFrame: Datos OHLCV o None si falla.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error al obtener datos OHLCV para {symbol}: {e}")
        return None

def get_top_symbols_by_volume(base_currency="USDT", min_volume=100000):
    """
    Obtiene pares de trading con volumen significativo.
    
    Args:
        base_currency (str): Moneda base (e.g., "USDT").
        min_volume (float): Volumen mínimo en USD.
    
    Returns:
        list: Lista de símbolos filtrados.
    """
    try:
        markets = exchange.load_markets()
        symbols = [
            symbol for symbol, market in markets.items()
            if market['quote'] == base_currency and market['type'] == 'spot' and market['active']
        ]
        tickers = exchange.fetch_tickers(symbols)
        volumes = [
            {'symbol': symbol, 'volume': ticker['quoteVolume']}
            for symbol, ticker in tickers.items()
            if 'quoteVolume' in ticker and ticker['quoteVolume'] >= min_volume
        ]
        sorted_symbols = sorted(volumes, key=lambda x: x['volume'], reverse=True)
        return [item['symbol'] for item in sorted_symbols]
    except Exception as e:
        print(f"Error al obtener símbolos por volumen: {e}")
        return []

def choose_best_cryptos(base_currency="USDT", top_n=200):
    """
    Selecciona las mejores criptomonedas para invertir basadas en volumen, volatilidad,
    tendencia y momentum.
    
    Args:
        base_currency (str): Moneda base (e.g., "USDT").
        top_n (int): Número máximo de símbolos a devolver (default: 200).
    
    Returns:
        list: Lista de símbolos seleccionados.
    """
    # Obtener símbolos iniciales con volumen suficiente
    symbols = get_top_symbols_by_volume(base_currency, min_volume=100000)
    if not symbols:
        print("No se encontraron símbolos válidos.")
        return []

    crypto_data = []
    max_symbols_to_analyze = min(400, len(symbols))  # Analizar más para asegurar candidatos

    for symbol in symbols[:max_symbols_to_analyze]:
        df = fetch_ohlcv_safe(symbol, timeframe='1h', limit=48)
        if df is None or df.empty or len(df) < 48:
            continue

        # Calcular indicadores
        # Volatilidad (rango promedio en %)
        volatility = ((df['high'] - df['low']) / df['close']).mean() * 100
        
        # Volumen promedio en USDT
        avg_volume = (df['volume'] * df['close']).mean()
        
        # Tendencia (ADX)
        adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        adx_value = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        
        # Momentum (cambio porcentual en las últimas 6 horas)
        price_change_6h = (df['close'].iloc[-1] - df['close'].iloc[-7]) / df['close'].iloc[-7] * 100
        
        # RSI para evitar sobrecompra
        rsi = RSIIndicator(df['close'], window=14).rsi()
        rsi_value = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        # Filtros básicos para incluir solo candidatos válidos
        if (adx_value > 25 and  # Tendencia fuerte
            volatility > 1.0 and  # Volatilidad mínima del 1%
            rsi_value < 70 and  # Evitar sobrecompra
            price_change_6h > 0):  # Momentum positivo
            # Puntaje combinado
            score = (avg_volume * volatility * adx_value * (1 + price_change_6h / 100))
            crypto_data.append({
                'symbol': symbol,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'adx': adx_value,
                'price_change_6h': price_change_6h,
                'rsi': rsi_value,
                'score': score
            })
        
        time.sleep(0.05)  # Respetar límites de la API

    if not crypto_data:
        print("No se encontraron criptos que cumplan los criterios.")
        return []

    # Crear DataFrame y ordenar por puntaje
    df = pd.DataFrame(crypto_data)
    df = df.sort_values(by='score', ascending=False)
    
    # Seleccionar hasta top_n símbolos
    selected_symbols = df['symbol'].head(top_n).tolist()
    
    print(f"DEBUG: Símbolos analizados: {len(crypto_data)}, seleccionados: {len(selected_symbols)}")
    return selected_symbols

# Ejemplo de uso (para pruebas)
if __name__ == "__main__":
    selected_cryptos = choose_best_cryptos(base_currency="USDT", top_n=200)
    print(f"Top 200 criptos seleccionadas: {len(selected_cryptos)} símbolos")
    print(selected_cryptos)