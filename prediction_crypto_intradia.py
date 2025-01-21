import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Librería de Binance
from binance.client import Client

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Preprocesamiento
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Hyperparameter Tuning
import keras_tuner as kt  # Asegúrate de tener keras-tuner instalado

###############################################################################
#                           FUNCIONES AUXILIARES
###############################################################################
def get_klines_binance(symbol="BTCUSDT", interval="15m", lookback="60 day ago UTC"):
    """
    Descarga velas históricas desde Binance para el símbolo e intervalo dados.
    lookback: por ejemplo, '60 day ago UTC'.
    """
    client = Client()  # Puedes usar tus API keys si lo requieres: Client(api_key, api_secret)

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

def calculate_rsi(series, window=14):
    """
    RSI clásico: 100 - (100 / (1 + RS)), donde RS = avg_gain / avg_loss
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_acceleration(series):
    """
    Calcula la aceleración de la serie de precios.
    - Primera Derivada: Tasa de cambio (diferencia entre cierres consecutivos).
    - Segunda Derivada: Aceleración (cambio en la tasa de cambio).
    """
    # Primera derivada: Tasa de cambio
    rate_of_change = series.diff()

    # Segunda derivada: Aceleración
    acceleration = rate_of_change.diff()

    return acceleration

def add_bollinger_bands(df, window=20, num_std=2):
    """
    Agrega Bollinger Bands al DataFrame.
    """
    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()
    df['Bollinger_High'] = rolling_mean + (rolling_std * num_std)
    df['Bollinger_Low'] = rolling_mean - (rolling_std * num_std)
    return df

def calculate_macd(df, span1=12, span2=26, span_signal=9):
    """
    Calcula el MACD y su señal.
    """
    ema1 = df['Close'].ewm(span=span1, adjust=False).mean()
    ema2 = df['Close'].ewm(span=span2, adjust=False).mean()
    df['MACD'] = ema1 - ema2
    df['MACD_Signal'] = df['MACD'].ewm(span=span_signal, adjust=False).mean()
    return df

def add_trend_features(df):
    """
    Agrega al DataFrame:
      - RSI (14)
      - MA7, MA25, MA99
      - Diferencias: diff_7_25, diff_25_99 (para medir 'espacio' entre medias)
      - Pendiente de cada MA (slope_7, slope_25, slope_99)
      - Cruces (cross_7_25, cross_25_99)
      - avg_volume_24h (rolling de 96 velas de 15m = 24h)
      - Aceleración de la tendencia
      - Bollinger Bands
      - MACD
    """
    df = df.copy()

    # RSI
    df['RSI'] = calculate_rsi(df['Close'], window=14)

    # Medias móviles
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA25'] = df['Close'].rolling(25).mean()
    df['MA99'] = df['Close'].rolling(99).mean()

    # Diferencias entre MAs
    df['diff_7_25'] = df['MA7'] - df['MA25']
    df['diff_25_99'] = df['MA25'] - df['MA99']

    # Pendiente (slope) de cada MA
    df['slope_7'] = df['MA7'].diff(1)
    df['slope_25'] = df['MA25'].diff(1)
    df['slope_99'] = df['MA99'].diff(1)

    # Cruces
    df['cross_7_25'] = (df['MA7'] > df['MA25']).astype(int)
    df['cross_25_99'] = (df['MA25'] > df['MA99']).astype(int)

    # avg_volume_24h (96 velas de 15m = 24h)
    df['avg_volume_24h'] = df['Volume'].rolling(96).mean()

    # Aceleración de la tendencia
    df['acceleration'] = calculate_acceleration(df['Close'])

    # Agregar Bollinger Bands
    df = add_bollinger_bands(df, window=20, num_std=2)

    # Agregar MACD
    df = calculate_macd(df)

    # Rellenar NaN (iniciales por rolling, RSI, aceleración, etc.)
    df.bfill(inplace=True)  # Actualizado para evitar FutureWarning

    # Seleccionar columnas relevantes
    cols = [
        'Close',            # 0
        'RSI',              # 1
        'MA7', 'MA25', 'MA99', 
        'diff_7_25', 'diff_25_99',
        'slope_7', 'slope_25', 'slope_99',
        'cross_7_25', 'cross_25_99',
        'Volume', 'avg_volume_24h',
        'acceleration',    # Aceleración
        'Bollinger_High', 'Bollinger_Low',
        'MACD', 'MACD_Signal'
    ]
    df = df[cols]
    df.dropna(inplace=True)
    return df

def create_sequences_multi_output(scaled_data, target_data, look_back=30):
    """
    Crea secuencias de tamaño 'look_back' para entrenamiento con múltiples salidas.
    X: (n_samples, look_back, n_features)
    y: (n_samples, n_outputs)  -> [High, Mean, Low] en el tiempo futuro
    """
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(target_data[i])
    return np.array(X), np.array(y)

def split_data(X, y, train_size=0.7, val_size=0.2, test_size=0.1):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba de manera temporal.
    """
    # Modificado para permitir una pequeña tolerancia en la suma
    assert abs(train_size + val_size + test_size -1.0) <1e-6, "Las proporciones deben sumar 1"
    n_samples = X.shape[0]
    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_lstm_model(look_back, num_features, num_outputs):
    """
    Construye el modelo LSTM optimizado:
    - 2 capas LSTM con regularización L2 y recurrent dropout
    - BatchNormalization
    - Dropout
    - Dense final
    """
    model = Sequential([
        Input(shape=(look_back, num_features)),
        LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001), dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(128, kernel_regularizer=l2(0.001), dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_outputs)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=tf.keras.losses.Huber(delta=1.0), 
                  metrics=['mae'])
    return model

def build_model_tuner(hp, look_back, num_features, num_outputs):
    """
    Construye el modelo para Keras Tuner.
    """
    model = Sequential()
    model.add(Input(shape=(look_back, num_features)))

    # Número de capas LSTM
    for i in range(hp.Int('num_layers', 1, 3)):
        units = hp.Int(f'units_{i}', min_value=64, max_value=256, step=64)
        dropout_rate = hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)
        recurrent_dropout_rate = hp.Float(f'recurrent_dropout_{i}', 0.1, 0.5, step=0.1)
        model.add(LSTM(units, return_sequences=True if i < hp.Int('num_layers', 1, 3)-1 else False,
                       kernel_regularizer=l2(hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')),
                       dropout=dropout_rate,
                       recurrent_dropout=recurrent_dropout_rate))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_outputs))

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )
    return model

###############################################################################
#                ENTRENAMIENTO Y PREDICCIÓN MULTI-SALIDA CON CONDICIONAL
###############################################################################
def train_and_predict_price_metrics(
    symbol='BTCUSDT',
    interval='15m',
    lookback_period='60 day ago UTC',  # Aumentado a 60 días
    look_back=30,      # Aumentada la ventana de observación
    future_bars=1,     # predecir el próximo intervalo (15m)
    epochs=100,        # Aumentamos el número de épocas para mejor entrenamiento
    batch_size=32,
    model_path='lstm_15m_price_metrics.keras',
    patience=10,       # Número de épocas sin mejora para Early Stopping
    tuner_search=False  # Cambiar a True para realizar búsqueda de hiperparámetros
):
    """
    - Descarga datos a 15m de Binance.
    - Agrega features de tendencia (RSI, MAs, diffs, slopes, cruces, volumen, aceleración, Bollinger Bands, MACD).
    - Prepara variables objetivo: High, Mean, Low en el siguiente intervalo.
    - Escala las características y las variables objetivo.
    - Divide los datos en entrenamiento, validación y prueba.
    - Entrena un LSTM para predecir High, Mean, Low.
    - Implementa Early Stopping para prevenir sobreentrenamiento.
    - Si tuner_search=True, realiza búsqueda de hiperparámetros con Keras Tuner.
    - Guarda el modelo en .keras
    - Evalúa el modelo y muestra métricas de regresión.
    - Visualiza las curvas de pérdida y MAE.
    - Implementa un ensemble de modelos para mejorar la robustez.
    - Visualiza las predicciones futuras si la tendencia es alcista.
    """
    start_time = time.time()

    # 1. Descargar datos
    print("Descargando datos desde Binance...")
    df_raw = get_klines_binance(symbol, interval, lookback_period)
    print(f"Datos descargados: {len(df_raw)} filas -> {symbol} ({interval})")

    # 2. Agregar features
    df_feat = add_trend_features(df_raw)
    print(f"Filas tras features: {len(df_feat)}")
    if len(df_feat) < look_back + future_bars:
        raise ValueError(f"No hay suficientes datos para look_back={look_back} y future_bars={future_bars}.")

    # 3. Definir variables objetivo: High, Mean, Low del siguiente intervalo
    # Shift negativo para alinear las variables objetivo
    df_feat['High_future'] = df_raw['High'].shift(-future_bars)
    df_feat['Low_future'] = df_raw['Low'].shift(-future_bars)
    df_feat['Mean_future'] = df_feat[['High_future', 'Low_future']].mean(axis=1)

    # Eliminar filas con NaN en las variables objetivo
    df_feat.dropna(inplace=True)

    # 4. Escalar características y variables objetivo
    feature_cols = [
        'Close', 'RSI', 'MA7', 'MA25', 'MA99', 
        'diff_7_25', 'diff_25_99',
        'slope_7', 'slope_25', 'slope_99',
        'cross_7_25', 'cross_25_99',
        'Volume', 'avg_volume_24h',
        'acceleration',
        'Bollinger_High', 'Bollinger_Low',
        'MACD', 'MACD_Signal'
    ]
    scaler_features = StandardScaler()
    scaled_features = scaler_features.fit_transform(df_feat[feature_cols])

    # Variables objetivo
    target_cols = ['High_future', 'Mean_future', 'Low_future']
    scaler_targets = StandardScaler()
    scaled_targets = scaler_targets.fit_transform(df_feat[target_cols])

    # 5. Crear secuencias
    X, y = create_sequences_multi_output(scaled_features, scaled_targets, look_back=look_back)
    n_samples = X.shape[0]
    n_features = X.shape[2]
    num_outputs = y.shape[1]
    print(f"Secuencias de entrenamiento: {n_samples} | n_features: {n_features} | Outputs: {num_outputs}")

    # 6. Dividir los datos
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Validación: {X_val.shape[0]} muestras")
    print(f"Prueba: {X_test.shape[0]} muestras")

    # 7. Verificar Compatibilidad del Modelo
    def is_model_compatible(model_path, expected_input_shape):
        if not os.path.exists(model_path):
            return False
        try:
            model = tf.keras.models.load_model(model_path)
            return model.input_shape[1:] == expected_input_shape
        except:
            return False

    expected_input_shape = (look_back, n_features)
    if not is_model_compatible(model_path, expected_input_shape):
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"Modelo incompatible eliminado: {model_path}")
        print("\nCreando un nuevo modelo LSTM optimizado para predicción de métricas de precio...")
        model = build_lstm_model(look_back, n_features, num_outputs)
    else:
        print(f"\nCargando modelo existente: {model_path}")
        model = tf.keras.models.load_model(model_path)

    # 8. Configurar Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',              # Métrica a monitorear
        patience=patience,               # Número de épocas sin mejora antes de detener
        restore_best_weights=True,       # Restaurar los mejores pesos obtenidos durante el entrenamiento
        mode='min',                      # Minimizar la métrica monitoreada
        verbose=1                        # Verbosidad para mensajes durante el entrenamiento
    )

    # 9. Ajuste de Hiperparámetros con Keras Tuner (Opcional)
    if tuner_search:
        print("\nIniciando búsqueda de hiperparámetros con Keras Tuner...")
        tuner = kt.RandomSearch(
            lambda hp: build_model_tuner(hp, look_back, n_features, num_outputs),
            objective='val_loss',
            max_trials=20,
            executions_per_trial=2,
            directory='kt_dir',
            project_name='lstm_price_prediction'
        )

        tuner.search(X_train, y_train,
                     epochs=50,
                     batch_size=32,
                     validation_data=(X_val, y_val),
                     callbacks=[early_stop])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nMejores hiperparámetros encontrados:")
        print(best_hps.values)

        # Construir el mejor modelo
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[early_stop]
        )
    else:
        # 10. Entrenar el modelo
        print("\nEntrenando el modelo...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[early_stop]
        )

    # 11. Guardar modelo en .keras
    model.save(model_path)
    print(f"\nModelo guardado en {model_path}")

    # 12. Visualizar Curvas de Pérdida y MAE
    plt.figure(figsize=(14,6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Curvas de Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE de Entrenamiento')
    plt.plot(history.history['val_mae'], label='MAE de Validación')
    plt.title('Curvas de MAE')
    plt.xlabel('Épocas')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)

    plt.show()

    # 13. Evaluar el modelo en el conjunto de prueba
    print("\nEvaluando el modelo en el conjunto de prueba...")
    y_pred_test = model.predict(X_test, verbose=0)
    y_true_test = y_test

    # Inversa del escalado
    y_true_test_rescaled = scaler_targets.inverse_transform(y_true_test)
    y_pred_test_rescaled = scaler_targets.inverse_transform(y_pred_test)

    # Calcular métricas de regresión para cada salida
    mae_test = mean_absolute_error(y_true_test_rescaled, y_pred_test_rescaled, multioutput='raw_values')
    mse_test = mean_squared_error(y_true_test_rescaled, y_pred_test_rescaled, multioutput='raw_values')
    rmse_test = np.sqrt(mse_test)

    for i, col in enumerate(target_cols):
        print(f"\nMétricas para {col} (Test):")
        print(f"Mean Absolute Error (MAE): {mae_test[i]:.6f}")
        print(f"Mean Squared Error (MSE): {mse_test[i]:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_test[i]:.6f}")

    # 14. Implementar un Ensemble de Modelos (Opcional)
    print("\nEntrenando un ensemble de modelos para mejorar la robustez...")
    num_ensembles = 5
    ensemble_models = []
    for i in range(num_ensembles):
        print(f"\nEntrenando modelo {i+1}/{num_ensembles}...")
        model_i = build_lstm_model(look_back, n_features, num_outputs)
        model_i.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(X_val, y_val),
            callbacks=[early_stop]
        )
        ensemble_models.append(model_i)

    # Realizar predicciones promedio en el conjunto de prueba
    print("\nRealizando predicciones con el ensemble de modelos...")
    y_preds = [model.predict(X_test, verbose=0) for model in ensemble_models]
    y_pred_ensemble = np.mean(y_preds, axis=0)
    y_pred_ensemble_rescaled = scaler_targets.inverse_transform(y_pred_ensemble)

    # Calcular métricas para el ensemble
    mae_ensemble = mean_absolute_error(y_true_test_rescaled, y_pred_ensemble_rescaled, multioutput='raw_values')
    mse_ensemble = mean_squared_error(y_true_test_rescaled, y_pred_ensemble_rescaled, multioutput='raw_values')
    rmse_ensemble = np.sqrt(mse_ensemble)

    for i, col in enumerate(target_cols):
        print(f"\nMétricas para {col} (Ensemble Test):")
        print(f"Mean Absolute Error (MAE): {mae_ensemble[i]:.6f}")
        print(f"Mean Squared Error (MSE): {mse_ensemble[i]:.6f}")
        print(f"Root Mean Squared Error (RMSE): {rmse_ensemble[i]:.6f}")

    # 15. Determinar la Tendencia Actual
    print("\nDeterminando la tendencia actual...")
    latest_data = df_feat.iloc[-1]
    if (latest_data['MA7'] > latest_data['MA25']) and (latest_data['MA25'] > latest_data['MA99']):
        current_trend = 'Alcista'
    elif (latest_data['MA7'] < latest_data['MA25']) and (latest_data['MA25'] < latest_data['MA99']):
        current_trend = 'Bajista'
    else:
        current_trend = 'Neutral'
    print(f"Tendencia actual: {current_trend}")

    # 16. Condicionar las Predicciones Basadas en la Tendencia
    if current_trend == 'Alcista':
        print("\nLa tendencia es alcista. Realizando predicción para el próximo intervalo...")
        # Predicción para el próximo intervalo usando el ensemble
        last_seq = scaled_features[-look_back:]  # últimas 'look_back' filas
        current_seq = last_seq.copy()

        preds_scaled = []
        for _ in range(future_bars):
            x_input = np.reshape(current_seq, (1, look_back, n_features))
            # Predicciones de cada modelo en el ensemble
            preds_ensemble = np.mean([model.predict(x_input, verbose=0) for model in ensemble_models], axis=0)[0]
            preds_scaled.append(preds_ensemble)

            # Actualizar la ventana para la siguiente predicción
            # Reemplazar 'Close' con predicción, pero considerar la naturaleza alcista
            next_step = current_seq[-1].copy()
            next_step[0] = preds_ensemble[0]  # 'Close' futuro predicho
            next_step[-1] = 0  # 'acceleration' no se puede determinar sin un nuevo precio real
            current_seq = np.vstack([current_seq[1:], next_step])

        # Inversa del escalado para las predicciones
        preds_rescaled = scaler_targets.inverse_transform(np.array(preds_scaled))

        # Generar índices de tiempo para las próximas velas
        last_time = df_feat.index[-1]
        future_times = pd.date_range(
            start=last_time,
            periods=future_bars + 1,  
            freq='15min'  # Reemplazado '15T' por '15min' para evitar FutureWarning
        )[1:]  # omite la primera que es la actual

        df_pred = pd.DataFrame({
            'Time': future_times,
            'Predicted_High': preds_rescaled[:,0],
            'Predicted_Mean': preds_rescaled[:,1],
            'Predicted_Low': preds_rescaled[:,2]
        }).set_index('Time')

        # 17. Visualización de las Predicciones Futuras
        plt.figure(figsize=(12,6))
        plt.plot(df_feat.index[-50:], df_feat['Close'].values[-50:], label='Close Real', marker='o')
        plt.plot(df_pred.index, df_pred['Predicted_High'], 'r--', label='Predicción High Futuro')
        plt.plot(df_pred.index, df_pred['Predicted_Mean'], 'g--', label='Predicción Mean Futuro')
        plt.plot(df_pred.index, df_pred['Predicted_Low'], 'b--', label='Predicción Low Futuro')
        plt.title(f"Predicción de High, Mean y Low a {future_bars} intervalos de 15m - {symbol}")
        plt.xlabel("Fecha-Hora")
        plt.ylabel("Precio")
        plt.legend()
        plt.grid(True)
        plt.show()

        print("\nPredicciones de las siguientes velas (15m cada una):")
        print(df_pred)
    else:
        print("\nLa tendencia no es alcista. No se realizará la predicción para el próximo intervalo.")

    end_time = time.time()
    print(f"\nTiempo total: {end_time - start_time:.2f} seg")

    return None  # Retornamos None ya que no siempre hay predicciones

###############################################################################
#                                   MAIN
###############################################################################
if __name__ == '__main__':
    # Parámetros
    SYMBOL = 'BTCUSDT'
    INTERVAL = '15m'
    LOOKBACK_PERIOD = '60 day ago UTC'  # Aumentado a 60 días
    LOOK_BACK = 30      # Aumentada la ventana de observación
    FUTURE_BARS = 1     # predecir el próximo intervalo (15m)
    EPOCHS = 100        # Aumentamos el número de épocas para mejor entrenamiento
    BATCH_SIZE = 32
    MODEL_PATH = 'lstm_15m_price_metrics.keras'  # <- Se guarda en formato .keras
    PATIENCE = 10       # Número de épocas sin mejora para Early Stopping
    TUNER_SEARCH = False  # Cambiar a True para realizar búsqueda de hiperparámetros

    train_and_predict_price_metrics(
        symbol=SYMBOL,
        interval=INTERVAL,
        lookback_period=LOOKBACK_PERIOD,
        look_back=LOOK_BACK,
        future_bars=FUTURE_BARS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_path=MODEL_PATH,
        patience=PATIENCE,
        tuner_search=TUNER_SEARCH
    )
