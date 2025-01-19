import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Librería de Binance
from binance.client import Client

# Para el modelo LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Para escalado
from sklearn.preprocessing import MinMaxScaler

###############################################################################
#                    FUNCIONES PARA DESCARGAR Y PROCESAR DATOS
###############################################################################
def get_klines_binance(symbol="BTCUSDT", interval="1m", lookback="3 day ago UTC"):
    """
    Descarga velas intradía desde Binance con python-binance.
    """
    client = Client()  # Usa API keys si lo requieres
    klines = client.get_historical_klines(symbol, interval, lookback)

    if not klines:
        raise ValueError("No se obtuvo data de Binance. Revisa el símbolo o lookback.")

    df = pd.DataFrame(klines, columns=[
        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_asset_volume', 'Number_of_trades',
        'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore'
    ])
    df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].astype(float)
    df['Open_time'] = pd.to_datetime(df['Open_time'], unit='ms')
    df.set_index('Open_time', inplace=True)

    return df[['Open','High','Low','Close','Volume']]

def calculate_atr(df, window=14):
    """
    ATR (Average True Range) para medir volatilidad.
    """
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['prev_close']).abs()
    low_close = (df['Low'] - df['prev_close']).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def ema(series, window=9):
    """
    Cálculo de la EMA (Exponential Moving Average).
    """
    return series.ewm(span=window, adjust=False).mean()

def macd(df, fast=12, slow=26, signal=9):
    """
    Cálculo del MACD:
      - MACD line = EMA(fast) - EMA(slow)
      - Signal line = EMA(MACD line, signal)
    Retorna 2 columnas: macd_line, macd_signal
    """
    df = df.copy()
    df['ema_fast'] = ema(df['Close'], fast)
    df['ema_slow'] = ema(df['Close'], slow)
    df['macd_line'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = ema(df['macd_line'], signal)
    return df[['macd_line', 'macd_signal']]

def add_features(df):
    """
    Agrega ATR, log_return, EMA(9), EMA(21) y MACD (12,26,9).
    """
    df = df.copy()

    # ATR
    df['ATR'] = calculate_atr(df, window=14).bfill()

    # Log Return
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_return'].fillna(0, inplace=True)

    # EMAs
    df['EMA_9'] = ema(df['Close'], 9)
    df['EMA_21'] = ema(df['Close'], 21)

    # MACD
    macd_vals = macd(df, fast=12, slow=26, signal=9)
    df['MACD_Line'] = macd_vals['macd_line']
    df['MACD_Signal'] = macd_vals['macd_signal']

    # Rellenar NaNs iniciales
    df.fillna(method='bfill', inplace=True)

    # Seleccionamos columnas y su orden
    df = df[['Close', 'Volume', 'ATR', 'log_return',
             'EMA_9', 'EMA_21', 'MACD_Line', 'MACD_Signal']]
    df.dropna(inplace=True)

    return df

###############################################################################
#                    MODELO LSTM Y FUNCIONES AUXILIARES
###############################################################################
@tf.function(reduce_retracing=True)
def predict_with_model(model, x_input):
    return model(x_input, training=False)

def create_sequences(scaled_data, look_back):
    """
    Crea secuencias de tamaño look_back a partir de los datos escalados.
    """
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, 0])  # columna 0 => 'Close'
    return np.array(X), np.array(y)

def build_lstm_model(look_back, num_features):
    """
    Arquitectura LSTM con mayor capacidad (128 unidades) y dropout.
    """
    model = Sequential([
        Input(shape=(look_back, num_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

###############################################################################
#                     FUNCIÓN PRINCIPAL DE BACKTEST & PREDICCIÓN
###############################################################################
def backtest_intraday_binance(
    symbol='BTCUSDT',
    interval='1m',
    lookback_period='3 day ago UTC',
    look_back=20,
    future_minutes=5,  
    epochs=10,         
    batch_size=32,
    plot=True,
    model_save_path='keras_intradia_model.h5'  # <-- Cambiamos el archivo de salida
):
    """
    Realiza un backtest intradía con walk-forward:
    - Con más features: ATR, log_return, EMA_9, EMA_21, MACD
    - LSTM de mayor capacidad
    - Predicción a 5 minutos
    - Guarda el modelo en keras_intradia_model.h5
    """
    start_time = time.time()

    # 1. Descarga datos de Binance
    print("\nDescargando datos desde Binance...")
    df_raw = get_klines_binance(symbol, interval, lookback=lookback_period)
    if df_raw.empty:
        raise ValueError(f"DataFrame vacío. No se pudo descargar datos para {symbol}.")

    print(f"Datos descargados: {len(df_raw)} filas")
    print(df_raw.head())

    # 2. Calcular las nuevas features
    print("\nProcesando features (ATR, log_return, EMA_9, EMA_21, MACD)...")
    df = add_features(df_raw)
    print(f"Número de filas en los datos procesados: {len(df)}")
    print(df.head(5))

    if len(df) < look_back + future_minutes:
        raise ValueError(f"No hay suficientes datos: se necesitan al menos {look_back + future_minutes} filas, pero solo hay {len(df)}.")

    # 3. Escalado
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Escalador para 'Close'
    scaler_close = MinMaxScaler()
    scaler_close.fit(df[['Close']])

    dates = df.index
    pred_dates = []
    pred_values = []
    actual_values = []

    start_index = look_back
    total_steps = len(df) - look_back - future_minutes

    # 4. Modelo (crear o cargar)
    if os.path.exists(model_save_path):
        print(f"\nCargando modelo existente: {model_save_path}")
        model = tf.keras.models.load_model(model_save_path)
    else:
        model = None

    # 5. Walk-forward
    print("\nIniciando simulación (Walk-Forward)...")
    while (start_index + future_minutes) <= len(df):
        progress = (start_index - look_back) / total_steps * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (progress / 100) if progress > 0 else 0
        remaining_time = estimated_total_time - elapsed_time

        print(f"Progreso: {progress:.2f}% | Tiempo transcurrido: {elapsed_time:.2f}s | Tiempo restante estimado: {remaining_time:.2f}s")

        train_data = scaled_data[:start_index]
        if len(train_data) <= look_back:
            print(f"train_data insuficiente para generar secuencias: {len(train_data)} filas, se necesitan > {look_back}")
            start_index += 1
            continue

        X_train, y_train = create_sequences(train_data, look_back)
        if X_train.shape[0] == 0 or len(y_train) == 0:
            raise ValueError("Las secuencias generadas están vacías. Revisa los datos de entrada.")

        num_features = X_train.shape[2]

        if model is None:
            model = build_lstm_model(look_back, num_features)

        # Entrenar
        model.fit(
            X_train, y_train,
            validation_split=0.1 if len(X_train) > 20 else None,
            epochs=epochs,      
            batch_size=batch_size,
            verbose=0
        )

        # Predicción de los próximos future_minutes
        last_seq = scaled_data[start_index - look_back:start_index]
        current_seq = last_seq.copy()
        block_preds = []

        for _ in range(future_minutes):
            x_input = np.reshape(current_seq, (1, look_back, num_features))
            pred = predict_with_model(model, tf.convert_to_tensor(x_input, dtype=tf.float32))
            block_preds.append(pred.numpy()[0, 0])
            next_step = [pred.numpy()[0, 0]] + list(current_seq[-1, 1:])
            current_seq = np.append(current_seq[1:], [next_step], axis=0)

        block_preds_rescaled = scaler_close.inverse_transform(
            np.array(block_preds).reshape(-1, 1)
        ).flatten()

        block_pred_dates = dates[start_index:start_index + future_minutes]
        block_actual = df['Close'].values[start_index:start_index + future_minutes]

        pred_dates.extend(block_pred_dates)
        pred_values.extend(block_preds_rescaled)
        actual_values.extend(block_actual)

        start_index += 1

    # 6. Construir df_result
    df_result = pd.DataFrame({
        'Date': pred_dates,
        'Actual': actual_values,
        'Predicted': pred_values
    }).set_index('Date')

    df_result['Error'] = df_result['Actual'] - df_result['Predicted']
    df_result['AbsError'] = df_result['Error'].abs()
    mae = df_result['AbsError'].mean()
    rmse = np.sqrt((df_result['Error'] ** 2).mean())

    print(f"\nMAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # 7. Guardar modelo
    model.save(model_save_path)
    print(f"Modelo guardado en {model_save_path}")

    # 8. Visualización final
    if plot:
        plt.figure(figsize=(12, 7))
        last_points = 200 if len(df) > 200 else len(df)
        plt.plot(df.index[-last_points:], df['Close'].values[-last_points:], label='Close Real', color='blue')

        tail = min(len(df_result), future_minutes)
        plt.plot(df_result.index[-tail:], df_result['Predicted'][-tail:], 'r--', label='Predicción')

        plt.title(f"{symbol} (último {lookback_period}) - Predicción a {future_minutes} min (LSTM)")
        plt.xlabel("Fecha-Hora")
        plt.ylabel("Precio")
        plt.legend()
        plt.show()

    end_time = time.time()
    print(f"\nSimulación completada en: {end_time - start_time:.2f} segundos")
    return df_result

###############################################################################
#                                MAIN
###############################################################################
if __name__ == '__main__':
    SYMBOL = 'BTCUSDT'
    INTERVAL = '1m'
    LOOKBACK_PERIOD = '3 day ago UTC'
    LOOK_BACK = 20      # Ventana de 20
    FUTURE_MINUTES = 5  # Predicción a 5 min
    EPOCHS = 10
    BATCH_SIZE = 32
    
    # Nuevo nombre de archivo de modelo
    MODEL_PATH = 'keras_intradia_model.h5'

    result_df = backtest_intraday_binance(
        symbol=SYMBOL,
        interval=INTERVAL,
        lookback_period=LOOKBACK_PERIOD,
        look_back=LOOK_BACK,
        future_minutes=FUTURE_MINUTES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        plot=True,
        model_save_path=MODEL_PATH
    )

    print("\n--- MUESTRA DE RESULTADOS ---")
    print(result_df.head())
    print(result_df.tail())
