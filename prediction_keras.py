import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Función para cargar el modelo y predecir el precio de cierre para mañana
def predict_next_day_price(model_path, ticker='BTC-USD', look_back=60):
    # Descargar los datos más recientes de yfinance
    try:
        df = yf.download(ticker, period='6mo')  # Cambiar a un período válido
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    if df.empty:
        print("No data downloaded. Please check the ticker or the time period.")
        return

    df = df[['Close']].dropna()

    # Agregar los indicadores necesarios
    df['RSI'] = calculate_rsi(df['Close'])
    df['MA7'] = calculate_moving_average(df['Close'], window=7)
    df['MA25'] = calculate_moving_average(df['Close'], window=25)
    df['MA99'] = calculate_moving_average(df['Close'], window=99)
    df['Momentum'] = calculate_momentum(df['Close'])
    df['Stochastic'] = calculate_stochastic(df['Close'])
    df['UpperBand'], df['LowerBand'] = calculate_bollinger_bands(df['Close'])
    df.fillna(0, inplace=True)

    if df.empty or len(df) < look_back:
        print("Not enough data to make predictions. Ensure the dataset is large enough.")
        return

    # Escalar los datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    scaler_close = MinMaxScaler()
    scaled_close = scaler_close.fit_transform(df[['Close']])

    # Cargar el modelo guardado
    print("Loading existing model...")
    model = tf.keras.models.load_model(model_path)

    # Tomar la secuencia más reciente para predecir
    last_sequence = scaled_data[-look_back:]
    last_sequence = np.reshape(last_sequence, (1, look_back, scaled_data.shape[1]))

    # Realizar la predicción
    next_day_scaled = model.predict(last_sequence)[0, 0]
    next_day_price = scaler_close.inverse_transform([[next_day_scaled]])[0, 0]

    print(f"Predicted closing price for tomorrow: {next_day_price:.2f}")
    return next_day_price

# Indicadores necesarios
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_moving_average(data, window):
    return data.rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return upper_band, lower_band

def calculate_momentum(data, window=10):
    return data.diff(window)

def calculate_stochastic(data, window=14):
    lowest_low = data.rolling(window=window).min()
    highest_high = data.rolling(window=window).max()
    return 100 * ((data - lowest_low) / (highest_high - lowest_low))

# Ejecutar la predicción
if __name__ == '__main__':
    model_path = 'lstm_model.keras'
    predict_next_day_price(model_path=model_path)
