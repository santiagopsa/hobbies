import schedule
import time
import datetime
import os
import numpy as np
import pandas as pd
import ccxt
import tensorflow as tf

# IMPORTA TUS FUNCIONES DE PREDICCION (YA EXISTENTES Y FUNCIONANDO):
from prediction_keras_day import predict_next_day_price
from prediction_keras_intradia import predict_next_intra_price

############################################################
#                CONFIGURACIONES Y VARIABLES
############################################################

exchange = ccxt.binance()
# Rutas de los modelos (ajusta si es necesario)
MODEL_PATH_DAY = 'models/keras_intradia_model.h5'
MODEL_PATH_INTRA = 'models/intra_model.keras'

# Carpeta donde se guardarán los logs diarios
LOG_DIR = 'logs'

# Variables globales
daily_trend = None            # "alcista", "bajista" o "lateral"
daily_predicted_close = None  # Predicción de precio de cierre (float)
current_position = None       # "long", "short" o None
trailing_stop_price = None    # Nivel dinámico de trailing stop
entry_price = None            # Precio al que abrimos la posición
position_size = 0.01          # Ejemplo de tamaño de posición (en BTC, ajusta según tu gestión de riesgo)

############################################################
#                   FUNCIONES DE LOG
############################################################

def log_message(message):
    """
    Guarda 'message' en un archivo de texto diario y también lo muestra en pantalla.
    """
    today_str = datetime.date.today().isoformat()
    log_filename = f"trading_log_{today_str}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)

    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    final_message = f"{timestamp} {message}"

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(final_message + '\n')

    print(final_message)

############################################################
#               PREDICCION CON VALIDACION
############################################################

def validate_input_shape(sequence, expected_shape):
    """
    Valida y ajusta la forma de la secuencia para que coincida con la forma esperada del modelo.
    """
    sequence_shape = sequence.shape
    if sequence_shape[1] < expected_shape[1]:
        padding = np.zeros((sequence_shape[0], expected_shape[1] - sequence_shape[1]))
        sequence = np.hstack((sequence, padding))
    elif sequence_shape[1] > expected_shape[1]:
        sequence = sequence[:, :expected_shape[1]]
    return sequence

def predict_next_day_price_safe(model_path, last_sequence):
    """
    Predice el precio del próximo día con validación de la forma de entrada.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        expected_shape = model.input_shape
        last_sequence = validate_input_shape(last_sequence, expected_shape)
        predicted_price = model.predict(last_sequence)[0, 0]
        return predicted_price
    except Exception as e:
        log_message(f"Error en predict_next_day_price_safe: {str(e)}")
        return None

############################################################
#        FUNCIONES PARA OBTENER PRECIOS/DATOS REALES
############################################################

def get_current_price():
    ticker = exchange.fetch_ticker('BTC/USDT')
    current_price = ticker['last']
    return current_price

def get_historical_data():
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=30)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

############################################################
#                   CALCULO DE ATR
############################################################

def calculate_atr(data, period=14):
    df = data.copy()
    df['previous_close'] = df['close'].shift(1)
    df['tr'] = df.apply(lambda row: max(
        row['high'] - row['low'],
        abs(row['high'] - row['previous_close']),
        abs(row['low'] - row['previous_close'])
    ), axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    return df['atr'].iloc[-1]

############################################################
#       LOGICA DE PREDICCION DIARIA Y INTRADIA
############################################################

def get_daily_prediction():
    global daily_trend, daily_predicted_close

    current_price = get_current_price()
    last_sequence = np.random.rand(1, 9)  # Replace with your actual input sequence
    predicted_price = predict_next_day_price_safe(MODEL_PATH_DAY, last_sequence)
    
    if predicted_price is None:
        log_message("No se pudo realizar la predicción diaria.")
        return
    
    daily_predicted_close = predicted_price

    if predicted_price > current_price:
        daily_trend = "alcista"
    elif predicted_price < current_price:
        daily_trend = "bajista"
    else:
        daily_trend = "lateral"

    log_message(f"Predicción diaria: {predicted_price:.2f}, Precio actual: {current_price:.2f}, Tendencia: {daily_trend}")

def intra_trading_logic():
    global daily_trend, daily_predicted_close

    current_price = get_current_price()
    last_sequence = np.random.rand(1, 9)  # Replace with your actual input sequence
    predicted_intra = predict_next_day_price_safe(MODEL_PATH_INTRA, last_sequence)
    
    if predicted_intra is None:
        log_message("No se pudo realizar la predicción intradía.")
        return

    log_message(f"Predicción intradía: {predicted_intra:.2f}, Precio actual: {current_price:.2f}, Tendencia diaria: {daily_trend}")

############################################################
#                      MAIN
############################################################

def main():
    schedule.every().day.at("00:01").do(get_daily_prediction)
    schedule.every(5).minutes.do(intra_trading_logic)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()
