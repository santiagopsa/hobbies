import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Funciones auxiliares (usar las mismas que definiste previamente)
# Incluye `get_klines_binance`, `add_features`, etc.
def predict_next_intra_price(model_path, symbol="BTCUSDT", interval="1m", lookback="3 hour ago UTC", look_back=20, future_minutes=5):
    """
    Genera predicciones para los próximos 'future_minutes' utilizando un modelo intradía previamente entrenado.
    
    Args:
        model_path (str): Ruta del modelo entrenado (archivo .h5).
        symbol (str): Par de trading (por ejemplo, 'BTCUSDT').
        interval (str): Intervalo de las velas (por ejemplo, '1m').
        lookback (str): Periodo de tiempo para obtener datos recientes (por ejemplo, '3 hour ago UTC').
        look_back (int): Número de velas previas que se usan para la predicción.
        future_minutes (int): Número de minutos para predecir.

    Returns:
        float: Precio proyectado para el futuro inmediato.
    """
    # 1. Descargar datos recientes
    df_raw = get_klines_binance(symbol, interval, lookback)
    if df_raw.empty:
        raise ValueError("No se pudo descargar datos para la predicción intradía.")

    # 2. Preprocesar datos
    df = add_features(df_raw)
    if len(df) < look_back:
        raise ValueError(f"No hay suficientes datos para la predicción intradía. Se necesitan al menos {look_back} velas.")

    # Escalar datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    scaler_close = MinMaxScaler()
    scaler_close.fit(df[['Close']])

    # 3. Cargar el modelo
    model = tf.keras.models.load_model(model_path)

    # 4. Preparar secuencia para predicción
    last_sequence = scaled_data[-look_back:]
    current_seq = last_sequence.copy()

    # Generar predicciones para los próximos `future_minutes`
    predictions = []
    for _ in range(future_minutes):
        x_input = np.reshape(current_seq, (1, look_back, current_seq.shape[1]))
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        predictions.append(pred_scaled)
        next_step = [pred_scaled] + list(current_seq[-1, 1:])
        current_seq = np.append(current_seq[1:], [next_step], axis=0)

    # Desescalar la predicción del primer minuto
    predictions_rescaled = scaler_close.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    return predictions_rescaled[0]

def add_features(df):
    """
    Agrega indicadores técnicos al DataFrame:
    - ATR (Average True Range)
    - Log Return
    - EMA (Exponential Moving Average) con ventanas de 9 y 21
    - MACD (Moving Average Convergence Divergence)

    Args:
        df (pd.DataFrame): DataFrame con columnas ['Open', 'High', 'Low', 'Close', 'Volume'].

    Returns:
        pd.DataFrame: DataFrame con las columnas originales más las columnas de los indicadores técnicos.
    """
    df = df.copy()

    # ATR (Average True Range)
    df['ATR'] = calculate_atr(df, window=14).bfill()

    # Log Return
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_return'].fillna(0, inplace=True)

    # EMAs (Exponential Moving Averages)
    df['EMA_9'] = ema(df['Close'], 9)
    df['EMA_21'] = ema(df['Close'], 21)

    # MACD (Moving Average Convergence Divergence)
    macd_vals = macd(df, fast=12, slow=26, signal=9)
    df['MACD_Line'] = macd_vals['macd_line']
    df['MACD_Signal'] = macd_vals['macd_signal']

    # Rellenar valores faltantes
    df.fillna(method='bfill', inplace=True)

    # Seleccionamos columnas en orden lógico
    df = df[['Close', 'Volume', 'ATR', 'log_return', 'EMA_9', 'EMA_21', 'MACD_Line', 'MACD_Signal']]
    df.dropna(inplace=True)

    return df

def calculate_atr(df, window=14):
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['prev_close']).abs()
    low_close = (df['Low'] - df['prev_close']).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    df['ema_fast'] = ema(df['Close'], fast)
    df['ema_slow'] = ema(df['Close'], slow)
    df['macd_line'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = ema(df['macd_line'], signal)
    return df[['macd_line', 'macd_signal']]


def get_klines_binance(symbol="BTCUSDT", interval="1m", lookback="3 day ago UTC"):
    """
    Descarga velas históricas desde Binance.
    
    Args:
        symbol (str): Símbolo del par de trading (por ejemplo, 'BTCUSDT').
        interval (str): Intervalo de las velas (por ejemplo, '1m', '5m', '1h').
        lookback (str): Periodo de tiempo a descargar (por ejemplo, '3 day ago UTC').
        
    Returns:
        pd.DataFrame: DataFrame con las columnas ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """
    # Inicializa el cliente de Binance (puedes usar claves de API si es necesario)
    client = Client()  # Si tienes claves de API, usa Client(api_key, api_secret)
    
    # Descarga datos de velas históricas
    klines = client.get_historical_klines(symbol, interval, lookback)

    if not klines:
        raise ValueError("No se obtuvo data de Binance. Revisa el símbolo o lookback.")

    # Convierte los datos a un DataFrame
    df = pd.DataFrame(klines, columns=[
        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    
    # Convierte columnas relevantes a float
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    # Convierte Open_time a datetime y lo establece como índice
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df.set_index('Open_time', inplace=True)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def predict_next_minutes(
    model_path, symbol="BTCUSDT", interval="1m", lookback="3 hour ago UTC",
    look_back=20, future_minutes=5
):
    """
    Genera predicciones para los próximos `future_minutes` utilizando un modelo entrenado.
    """
    # 1. Descarga los datos más recientes de Binance
    print("\nDescargando datos recientes...")
    df_raw = get_klines_binance(symbol, interval, lookback=lookback)
    if df_raw.empty:
        raise ValueError("No se pudo descargar datos. Revisa el símbolo o el intervalo.")

    print(f"Datos descargados: {len(df_raw)} filas")
    print(df_raw.tail())

    # 2. Preprocesa los datos
    print("\nAgregando features...")
    df = add_features(df_raw)

    if len(df) < look_back:
        raise ValueError(f"No hay suficientes datos: se necesitan al menos {look_back} filas.")

    # Escalado
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Escalador para el cierre
    scaler_close = MinMaxScaler()
    scaler_close.fit(df[['Close']])

    # 3. Carga el modelo existente
    print(f"\nCargando modelo desde {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 4. Predicción
    print(f"\nRealizando predicción para los próximos {future_minutes} minutos...")
    last_sequence = scaled_data[-look_back:]
    current_seq = last_sequence.copy()

    predictions = []
    for _ in range(future_minutes):
        x_input = np.reshape(current_seq, (1, look_back, current_seq.shape[1]))
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        predictions.append(pred_scaled)

        # Actualiza la secuencia con la predicción
        next_step = [pred_scaled] + list(current_seq[-1, 1:])  # Agrega valores adicionales si es necesario
        current_seq = np.append(current_seq[1:], [next_step], axis=0)

    # Desescalar las predicciones
    predictions_rescaled = scaler_close.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    # 5. Mostrar resultados
    pred_times = pd.date_range(
        start=df.index[-1], periods=future_minutes + 1, freq=interval
    )[1:]
    results = pd.DataFrame({'Time': pred_times, 'Predicted_Close': predictions_rescaled})
    results.set_index('Time', inplace=True)

    print("\n--- Predicciones ---")
    print(results)

    # 6. Visualización
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-100:], df['Close'].values[-100:], label='Precio Real')
    plt.plot(results.index, results['Predicted_Close'], 'r--', label='Predicción')
    plt.title(f"Predicción de los próximos {future_minutes} minutos ({symbol})")
    plt.xlabel("Tiempo")
    plt.ylabel("Precio")
    plt.legend()
    plt.show()

    return results


# MAIN
if __name__ == "__main__":
    MODEL_PATH = 'keras_intradia_model.h5'
    SYMBOL = 'BTCUSDT'
    INTERVAL = '1m'
    LOOKBACK = '3 hour ago UTC'
    LOOK_BACK = 20
    FUTURE_MINUTES = 5

    predictions = predict_next_minutes(
        model_path=MODEL_PATH,
        symbol=SYMBOL,
        interval=INTERVAL,
        lookback=LOOKBACK,
        look_back=LOOK_BACK,
        future_minutes=FUTURE_MINUTES
    )
