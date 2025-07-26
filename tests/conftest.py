# tests/conftest.py
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


from app.app import create_app
from app.queue_manager import QueueManager

@pytest.fixture(scope="session")
def temp_volume():
    """Создаёт временный volume для тестов и очищает после."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)

@pytest.fixture(scope="session")
def queue_manager(temp_volume):
    """Настраивает QueueManager на временном volume."""
    qm = QueueManager(volume_path=temp_volume)
    return qm

@pytest.fixture(scope="function")
def client(monkeypatch, temp_volume):
    """Тестовый FastAPI-клиент с поправленным VOLUME_PATH."""
    # Переопределяем переменную окружения
    monkeypatch.setenv("VOLUME_PATH", temp_volume)
    monkeypatch.setenv("MAX_FILE_SIZE", "1073741824")
    
    # Создаём новый экземпляр приложения для каждого теста
    from app import app
    app.queue = QueueManager(volume_path=temp_volume)
    
    client = TestClient(app, base_url="http://testserver")
    return client
