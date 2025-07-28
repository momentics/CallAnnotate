# tests/conftest.py

import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app.app import app
import app.api.deps as deps

# Включаем src в PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

@pytest.fixture(scope="session", autouse=True)
def mock_queue_and_annotation():
    # Заготовленный mock-результат для test-job
    mock_task = MagicMock()
    mock_task.task_id = "test-job"
    mock_task.status = "completed"
    mock_task.message = "done"
    mock_task.result = {"foo": "bar"}
    mock_task.progress = 100
    mock_task.created_at = "2025-07-27T00:00:00Z"
    mock_task.updated_at = "2025-07-27T00:01:00Z"

    # Внутреннее хранилище задач
    jobs = {}

    async def add_task(job_id, metadata):
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

    # Подменяем singleton-очередь в модуле deps
    deps._queue = mock_q

    return mock_q

@pytest.fixture
def client(mock_queue_and_annotation):
    return TestClient(app)
