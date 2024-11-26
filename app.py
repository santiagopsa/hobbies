import os
from flask import Flask, render_template, jsonify
import demo_inversion

app = Flask(__name__)

@app.route('/')
def index():
    return "¡Bienvenido al simulador de trading!"

@app.route('/simular', methods=['GET'])
def simular():
    # Llamar la función demo_trading() y devolver un resultado
    resultado = demo_inversion.demo_trading()
    return jsonify({"resultado": resultado})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
