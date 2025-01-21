import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ===== Indicadores necesarios =====
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

# ===== Función principal =====
def predict_next_day_price(model_path, ticker='BTC-USD', look_back=30):
    """
    Carga el modelo entrenado y predice el precio de cierre de 'ticker' para el siguiente día
    usando la ventana (look_back) de últimos días. Ajustado para mantener coherencia
    con el entrenamiento (look_back=30).
    """

    # 1. Descargar datos (período de 6 meses debería ser suficiente para look_back=30)
    try:
        df = yf.download(ticker, period='6mo')
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    if df.empty:
        print("No data downloaded. Please check the ticker or the time period.")
        return

    # 2. Quedarnos solo con la columna 'Close' y eliminar NaN de esa columna
    df = df[['Close']].dropna()

    # 3. Calcular indicadores
    df['RSI'] = calculate_rsi(df['Close'])
    df['MA7'] = calculate_moving_average(df['Close'], window=7)
    df['MA25'] = calculate_moving_average(df['Close'], window=25)
    df['MA99'] = calculate_moving_average(df['Close'], window=99)
    df['Momentum'] = calculate_momentum(df['Close'])
    df['Stochastic'] = calculate_stochastic(df['Close'])
    df['UpperBand'], df['LowerBand'] = calculate_bollinger_bands(df['Close'])

    # 4. Rellenar NaN con 0 (igual que en el entrenamiento)
    df.fillna(0, inplace=True)

    # 5. Verificar que tengamos al menos 'look_back' filas para la predicción
    if len(df) < look_back:
        print("Not enough data to make predictions. Ensure the dataset is large enough.")
        return

    # 6. IMPORTANTE: Asegurar el ORDEN de columnas
    #    El modelo se entrenó con 'Close' como primera columna.
    #    Después RSI, MA7, MA25, MA99, Momentum, Stochastic, UpperBand, LowerBand.
    #    Ajustamos el DataFrame a ese orden explícito:
    columns_order = ['Close', 'RSI', 'MA7', 'MA25', 'MA99',
                     'Momentum', 'Stochastic', 'UpperBand', 'LowerBand']
    df = df[columns_order]

    # 7. Escalado de todas las columnas
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)  # df.shape -> (n_samples, 9)

    # 8. Escalado separado para 'Close' (solo para la inversa de la predicción)
    scaler_close = MinMaxScaler()
    scaled_close = scaler_close.fit_transform(df[['Close']])  # (n_samples, 1)

    # 9. Cargar el modelo
    print("Loading existing model...")
    model = tf.keras.models.load_model(model_path)

    # 10. Seleccionar la última ventana (look_back) de datos para predecir el siguiente día
    last_sequence = scaled_data[-look_back:]   # (look_back, 9)
    last_sequence = np.reshape(last_sequence, (1, look_back, scaled_data.shape[1]))  # (1, 30, 9)

    # 11. Realizar la predicción
    prediction_scaled = model.predict(last_sequence)[0, 0]
    next_day_price = scaler_close.inverse_transform([[prediction_scaled]])[0, 0]

    # 12. Mostrar resultado
    print(f"Predicted closing price for tomorrow: {next_day_price:.2f}")
    return next_day_price

# ===== Para ejecutar la predicción de forma directa =====
if __name__ == '__main__':
    model_path = 'lstm_model.keras'  # Asegúrate de que este nombre coincida con tu entrenamiento
    predict_next_day_price(model_path=model_path, ticker='BTC-USD', look_back=30)
