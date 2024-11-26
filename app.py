import os
import io
import sys
from flask import Flask, jsonify
import demo_inversion

app = Flask(__name__)

@app.route('/')
def index():
    return "¡Bienvenido al simulador de trading!"

@app.route('/simular', methods=['GET'])
def simular():
    # Redirigir stdout para capturar los print
    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        # Llamar la función demo_trading() y capturar la salida
        demo_inversion.demo_trading()
    finally:
        # Restaurar stdout para no afectar otros prints
        sys.stdout = sys.__stdout__

    # Obtener el contenido del buffer
    output = buffer.getvalue()
    return jsonify({"output": output})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

