import os
from flask import Flask, render_template, jsonify
from celery.result import AsyncResult
from demo_inversion import demo_trading  # Importa la tarea Celery

app = Flask(__name__)

@app.route('/')
def index():
    """
    Muestra el archivo HTML para iniciar la simulación.
    """
    return render_template('loader.html')

@app.route('/simular', methods=['GET'])
def simular():
    """
    Lanza la tarea demo_trading() en segundo plano con Celery.
    """
    task = demo_trading.apply_async()  # Lanza la tarea de Celery
    return jsonify({"task_id": task.id})  # Devuelve el ID de la tarea al cliente

@app.route('/status/<task_id>', methods=['GET'])
def task_status(task_id):
    """
    Consulta el estado de una tarea usando su ID.
    """
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        return jsonify({"status": "PENDING"})
    elif task_result.state == 'SUCCESS':
        return jsonify({"status": "SUCCESS", "result": task_result.result})
    elif task_result.state == 'FAILURE':
        return jsonify({"status": "FAILURE", "result": str(task_result.info)})
    else:
        return jsonify({"status": task_result.state})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


