import os
from celery import Celery

# Configuración de Celery
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    'tasks',
    broker=REDIS_URL,  # URL del broker (Redis en este caso)
    backend=REDIS_URL  # Backend para almacenar resultados
)

celery_app.conf.update(
    broker_use_ssl={
        'ssl_cert_reqs': 'CERT_NONE'  # Puedes usar 'CERT_REQUIRED' en producción con certificados válidos
    },
    redis_backend_use_ssl={
        'ssl_cert_reqs': 'CERT_NONE'  # Configuración para backend
    },
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
