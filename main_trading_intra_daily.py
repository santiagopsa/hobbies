import schedule
import time
import os
import numpy as np
import pandas as pd
import ccxt
import tensorflow as tf

# Manejo de fechas/horas
from datetime import datetime

# Rich para la tabla en consola
from rich.console import Console
from rich.table import Table

# IMPORTA TUS FUNCIONES DE PREDICCIÓN (YA EXISTENTES Y FUNCIONANDO)
from prediction_keras_day import predict_next_day_price
from prediction_keras_intradia import predict_next_intra_price

############################################################
#                CONFIGURACIONES Y VARIABLES
############################################################

exchange = ccxt.binance()

MODEL_PATH_DAY = 'lstm_model.keras'        # Modelo diario
MODEL_PATH_INTRA = 'keras_intradia_model.h5'  # Modelo intradía

LOG_DIR = 'logs'
CSV_LOG_PATH = 'predictions_log.csv'  # CSV donde se guardarán las predicciones

daily_trend = None
daily_predicted_close = None
current_position = None
trailing_stop_price = None
entry_price = None
position_size = 0.01

# Consola y tabla para visualización en tiempo real
console = Console()
table = Table(title="Predicciones en Tiempo Real")

# Define las columnas de la tabla
table.add_column("Fecha y Hora", justify="left")
table.add_column("Precio Actual (BTC/USDT)", justify="right")
table.add_column("Predicción Diaria", justify="right")
table.add_column("Tendencia Diaria", justify="right")

############################################################
#                   FUNCIONES DE LOG
############################################################

def log_message(message, context=None):
    """
    Guarda 'message' en un archivo de texto diario y también lo muestra en pantalla.
    Agrega información contextual si está disponible.
    
    Args:
        message (str): Mensaje principal para registrar.
        context (dict, optional): Información adicional a incluir en el log.
    """
    today_str = datetime.now().date().isoformat()
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    
    # Construimos el mensaje final
    full_log_message = f"{timestamp} {message}"
    if context:
        full_log_message += f" | Contexto: {context}"
    
    # Ruta del archivo de log diario
    log_filename = f"trading_log_{today_str}.txt"
    log_path = os.path.join(LOG_DIR, log_filename)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Escribimos en el archivo
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(full_log_message + '\n')
    
    # También se imprime en consola
    print(full_log_message)


def save_prediction_to_csv(data_dict, csv_path=CSV_LOG_PATH):
    """
    Guarda un registro de predicción en un CSV.
    Si el archivo no existe, crea encabezados; si existe, anexa la fila.
    
    Args:
        data_dict (dict): Diccionario con las claves y valores a guardar.
        csv_path (str): Ruta del CSV donde se guardarán las predicciones.
    """
    # Verificar si el archivo ya existe
    file_exists = os.path.isfile(csv_path)
    
    df_new = pd.DataFrame([data_dict])  # Un solo registro en formato DataFrame
    
    if not file_exists:
        # Crear archivo nuevo con encabezados
        df_new.to_csv(csv_path, index=False, mode='w', encoding='utf-8')
    else:
        # Anexar al archivo existente
        df_new.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8')

############################################################
#               FUNCIONES DE VISUALIZACIÓN
############################################################

def update_real_time_table(current_price, predicted_price, trend):
    """
    Actualiza la tabla en tiempo real con los datos más recientes.
    """
    console.clear()  # Limpia la consola
    table.rows.clear()  # Limpia las filas previas de la tabla

    # Añade una nueva fila con los datos más recientes
    table.add_row(
        datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"),  # Fecha y hora actuales
        f"{current_price:.2f}" if current_price else "N/A",  # Precio actual
        f"{predicted_price:.2f}" if predicted_price else "N/A",  # Predicción diaria
        trend if trend else "N/A"  # Tendencia
    )
    console.print(table)  # Muestra la tabla actualizada en la consola

############################################################
#               PREDICCIÓN CON VALIDACIÓN
############################################################

def validate_input_shape(sequence, expected_shape):
    """
    Ajusta la forma de 'sequence' para que coincida con 'expected_shape' (del modelo).
    Este ejemplo es simplificado y solo actúa sobre la segunda dimensión.
    """
    sequence_shape = sequence.shape
    # Ejemplo: el modelo espera (None, 60, 9) y la secuencia es (1, 60, 9)
    # No obstante, en la práctica podrías necesitar rellenar o recortar en la 3a dimensión.
    if len(sequence_shape) == 2 and len(expected_shape) == 3:
        # Caso en el que se requiere expandir dimensión
        sequence = np.expand_dims(sequence, axis=-1)  # (1, 60, 1) en lugar de (1, 60)
    
    # Si hicieras validaciones más complejas, agrégalas aquí.

    return sequence

def predict_next_day_price_safe(model_path, last_sequence):
    """
    Predice el precio con validación de la forma de entrada.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        expected_shape = model.input_shape  # p.ej. (None, 60, 9)

        last_sequence = validate_input_shape(last_sequence, expected_shape)
        predicted_price = model.predict(last_sequence)[0, 0]
        return predicted_price
    
    except Exception as e:
        log_message(
            "Error al realizar la predicción.",
            context={
                "error": str(e),
                "model_path": model_path,
                "last_sequence_shape": (last_sequence.shape if last_sequence is not None else None)
            }
        )
        return None

############################################################
#        FUNCIONES PARA OBTENER PRECIOS/DATOS REALES
############################################################

def get_current_price():
    """
    Obtiene el precio actual de BTC/USDT desde el exchange.
    """
    ticker = exchange.fetch_ticker('BTC/USDT')
    current_price = ticker['last']
    return current_price

def get_historical_data():
    """
    Descarga datos históricos diarios (30 velas) de BTC/USDT.
    """
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=30)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

############################################################
#       LÓGICA DE PREDICCIÓN DIARIA Y INTRADÍA
############################################################

def get_daily_prediction():
    """
    Realiza la predicción diaria con el modelo configurado, 
    actualiza la tendencia global y registra los resultados.
    """
    global daily_trend, daily_predicted_close
    
    # Precio actual
    current_price = get_current_price()
    
    # Secuencia de ejemplo (1, 60, 9). Ajustar a tus datos reales.
    last_sequence = np.random.rand(1, 60, 9)
    
    predicted_price = predict_next_day_price_safe(MODEL_PATH_DAY, last_sequence)
    
    # Registrar el resultado
    if predicted_price is None:
        log_message("No se pudo realizar la predicción diaria.", context={
            "current_price": current_price,
            "last_sequence_shape": last_sequence.shape,
            "model_path": MODEL_PATH_DAY
        })
        return
    
    daily_predicted_close = predicted_price
    
    # Determinar tendencia
    if predicted_price > current_price:
        daily_trend = "alcista"
    elif predicted_price < current_price:
        daily_trend = "bajista"
    else:
        daily_trend = "lateral"
    
    # Log detallado
    log_message("Predicción diaria realizada exitosamente.", context={
        "current_price": current_price,
        "predicted_price": predicted_price,
        "daily_trend": daily_trend,
        "last_sequence_shape": last_sequence.shape
    })

    # Guardar en tabla en tiempo real
    update_real_time_table(current_price, predicted_price, daily_trend)

    # Guardar en CSV
    save_prediction_to_csv({
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "daily",
        "current_price": current_price,
        "predicted_price": predicted_price,
        "trend": daily_trend
    })


def intra_trading_logic():
    """
    Realiza la predicción intradía y registra los resultados.
    """
    global daily_trend, daily_predicted_close
    
    # Precio actual
    current_price = get_current_price()
    
    # Secuencia de ejemplo (1, 60, 9) o la forma que requiera tu modelo intradía.
    # Este modelo parece esperar (None, 9) según tu código, ajústalo a tu realidad.
    last_sequence = np.random.rand(1, 60, 9)  
    
    predicted_intra = predict_next_day_price_safe(MODEL_PATH_INTRA, last_sequence)
    
    if predicted_intra is None:
        log_message("No se pudo realizar la predicción intradía.", context={
            "current_price": current_price,
            "model_path": MODEL_PATH_INTRA
        })
        return
    
    # Log de predicción
    log_message(
        "Predicción intradía realizada.",
        context={
            "predicted_intra": predicted_intra,
            "current_price": current_price,
            "daily_trend": daily_trend
        }
    )
    update_real_time_table(current_price, predicted_intra, daily_trend)

    # Guardar en CSV
    save_prediction_to_csv({
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "intradia",
        "current_price": current_price,
        "predicted_price": predicted_intra,
        "trend": daily_trend
    })

############################################################
#                      MAIN
############################################################

def main():
    # Programar las tareas diarias e intradía
    schedule.every().day.at("00:01").do(get_daily_prediction)
    schedule.every(5).minutes.do(intra_trading_logic)
    
    # Ejecutar inmediatamente la predicción diaria si ya pasó la hora de cierre (00:01)
    now = datetime.now()
    if now.hour >= 0 and now.minute > 1:
        log_message("Ejecutando predicción diaria inmediatamente porque el programa inició después de la hora programada.")
        get_daily_prediction()
    
    # Bucle principal
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            # En caso de error en la ejecución programada, lo registramos
            log_message(
                "Error en la ejecución de tareas programadas.",
                context={"error": str(e)}
            )
        time.sleep(1)


if __name__ == '__main__':
    main()
