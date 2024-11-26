from celery import Celery
import os

# Configuración de Celery
celery_app = Celery(
    'tasks',
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),  # URL del broker (Redis en este caso)
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")  # Backend para almacenar resultados
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
