import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Configuración de Celery
celery_app = Celery(
    'tasks',
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Agregar opciones de SSL si se utiliza rediss://
if REDIS_URL.startswith("rediss://"):
    celery_app.conf.broker_use_ssl = {
        "ssl_cert_reqs": "none"  # Cambiar a "required" si tienes certificados válidos
    }
    celery_app.conf.redis_backend_use_ssl = {
        "ssl_cert_reqs": "none"  # Cambiar a "required" si tienes certificados válidos
    }

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Asegúrate de importar el módulo que contiene las tareas
import demo_inversion