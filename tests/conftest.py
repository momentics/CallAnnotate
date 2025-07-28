# tests/conftest.py

import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient
from app.app import app
import app.api.deps as deps

# Добавляем src в PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Настройка тестового окружения:
    - Определяем тестовую директорию volume через VOLUME_PATH
    - Создаём полную структуру папок для тестов
    """
    from app.utils import ensure_volume_structure

    test_volume = Path(__file__).parent / "test_volume"
    os.environ["VOLUME_PATH"] = str(test_volume)

    # Обеспечиваем создание структуры volume
    ensure_volume_structure(str(test_volume))

    yield test_volume

    # Очистка после тестов
    import shutil
    shutil.rmtree(test_volume, ignore_errors=True)

@pytest.fixture(scope="session", autouse=True)
def mock_queue_and_annotation():
    """
    Мок для AsyncQueueManager и AnnotationService
    """
    # Подготовка mock-задачи для job_id "test-job"
    mock_task = MagicMock()
    mock_task.task_id = "test-job"
    mock_task.status = "completed"
    mock_task.message = "done"
    mock_task.result = {"foo": "bar"}
    mock_task.progress = 100
    mock_task.created_at = "2025-07-27T00:00:00Z"
    mock_task.updated_at = "2025-07-27T00:01:00Z"

    jobs = {}

    async def add_task(job_id, metadata):
        # Для каждого нового задания убеждаемся в структуре volume
        from app.utils import ensure_volume_structure
        volume_path = os.getenv("VOLUME_PATH", "./volume")
        ensure_volume_structure(volume_path)

        tr = AsyncMock()
        tr.task_id = job_id
        tr.status = "processing"
        tr.message = "queued"
        tr.result = None
        tr.progress = 0
        tr.created_at = "2025-07-27T00:00:00Z"
        tr.updated_at = "2025-07-27T00:00:00Z"
        jobs[job_id] = tr
        return True

    async def get_task_result(job_id):
        if job_id == "test-job":
            return mock_task
        return jobs.get(job_id)

    async def cancel_task(job_id):
        return False

    async def get_queue_info():
        return {"queue_length": 0, "processing_jobs": [], "queued_jobs": []}

    mock_q = AsyncMock()
    mock_q.add_task.side_effect = add_task
    mock_q.get_task_result.side_effect = get_task_result
    mock_q.cancel_task.side_effect = cancel_task
    mock_q.get_queue_info.side_effect = get_queue_info
    mock_q.start.return_value = None
    mock_q.stop.return_value = None

    # Заменяем singleton-очередь
    deps._queue = mock_q
    return mock_q

@pytest.fixture
def client(mock_queue_and_annotation, setup_test_environment):
    """
    Тестовый клиент FastAPI.
    Запуск TestClient в контекстном менеджере для выполнения событий startup.
    """
    with TestClient(app) as c:
        yield c
