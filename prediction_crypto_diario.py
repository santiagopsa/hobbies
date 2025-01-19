import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

@tf.function(reduce_retracing=True)
def predict_with_model(model, x_input):
    return model(x_input, training=False)

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

def create_sequences(scaled_data, look_back):
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

def build_lstm_model(look_back, num_features):
    model = Sequential([
        Input(shape=(look_back, num_features)),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def backtest_and_predict(
    ticker='BTC-USD',
    start_date='2022-01-01',
    look_back=30,
    future_days=1,
    step_days=1,
    epochs=5,
    batch_size=32,
    plot=True,
    model_save_path='lstm_model.h5'
):
    start_time = time.time()

    df = yf.download(ticker, start=start_date)
    df = df[['Close']].dropna()

    # Add technical indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['MA7'] = calculate_moving_average(df['Close'], window=7)
    df['MA25'] = calculate_moving_average(df['Close'], window=25)
    df['MA99'] = calculate_moving_average(df['Close'], window=99)
    df['Momentum'] = calculate_momentum(df['Close'])
    df['Stochastic'] = calculate_stochastic(df['Close'])
    df['UpperBand'], df['LowerBand'] = calculate_bollinger_bands(df['Close'])
    df.fillna(0, inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    scaler_close = MinMaxScaler()
    scaled_close = scaler_close.fit_transform(df[['Close']])

    dates = df.index
    pred_dates = []
    pred_values = []
    actual_values = []

    start_index = look_back + 1
    total_steps = len(scaled_close) - start_index - future_days

    if os.path.exists(model_save_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_save_path)
    else:
        model = None

    while start_index + future_days <= len(scaled_close):
        train_data = scaled_data[:start_index]
        X_train, y_train = create_sequences(train_data, look_back)

        if len(X_train) == 0:
            print("No hay suficientes datos para crear secuencias.")
            break

        num_features = X_train.shape[2]

        if model is None:
            model = build_lstm_model(look_back, num_features)

        if len(X_train) > 20:
            model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
        else:
            print("Training data too small for validation split. Training without validation.")
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )

        last_seq = scaled_data[start_index - look_back:start_index]
        current_seq = last_seq.copy()
        block_preds = []

        for _ in range(future_days):
            x_input = np.reshape(current_seq, (1, look_back, num_features))
            x_input_tf = tf.convert_to_tensor(x_input, dtype=tf.float32)
            pred = predict_with_model(model, x_input_tf)
            block_preds.append(pred.numpy()[0, 0])
            current_seq = np.append(current_seq[1:], [[pred.numpy()[0, 0]] * num_features], axis=0)

        block_preds_rescaled = scaler_close.inverse_transform(
            np.array(block_preds).reshape(-1, 1)
        ).flatten()

        block_pred_dates = dates[start_index:start_index + future_days]
        block_actual = df['Close'].values[start_index:start_index + future_days]

        pred_dates.extend(block_pred_dates)
        pred_values.extend(block_preds_rescaled)
        actual_values.extend(block_actual)

        start_index += step_days

        progress = (start_index - look_back - 1) / total_steps * 100
        print(f"Progress: {progress:.2f}% | Start Index: {start_index}")

    df_result = pd.DataFrame({
        'Date': pred_dates,
        'Actual': actual_values,
        'Predicted': pred_values
    })
    df_result.set_index('Date', inplace=True)

    df_result['Error'] = df_result['Actual'] - df_result['Predicted']
    df_result['AbsError'] = df_result['Error'].abs()
    mae = df_result['AbsError'].mean()
    rmse = np.sqrt((df_result['Error'] ** 2).mean())

    # Asegurar valores escalares para impresión
    mae = mae.item() if isinstance(mae, np.ndarray) else mae
    rmse = rmse.item() if isinstance(rmse, np.ndarray) else rmse

    print(f"\nWalk-Forward MAE: {mae:.2f}")
    print(f"Walk-Forward RMSE: {rmse:.2f}")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Close'], label='Close Real', color='blue')
        plt.plot(df_result.index, df_result['Predicted'], 'r--', label='Prediction')
        plt.title(f"Backtest Walk-Forward {ticker} LSTM (CPU)")
        plt.xlabel("Fecha")
        plt.ylabel("Precio")
        plt.legend()
        plt.show()

    model.save(model_save_path)
    print(f"Modelo guardado en {model_save_path}")

    prediction_input = scaled_data[-look_back:]
    prediction_input = np.reshape(prediction_input, (1, look_back, scaled_data.shape[1]))
    next_day_scaled = predict_with_model(model, prediction_input).numpy()[0, 0]
    next_day_price = scaler_close.inverse_transform([[next_day_scaled]])[0, 0]

    print(f"Predicción del precio de cierre para mañana: {next_day_price:.2f}")

    end_time = time.time()
    print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")

    return df_result

if __name__ == '__main__':
    result = backtest_and_predict(
        ticker='BTC-USD',
        start_date='2021-01-01',
        look_back=30,
        future_days=1,
        step_days=1,
        epochs=5,
        batch_size=32,
        plot=True
    )
    print(result.head(10))
