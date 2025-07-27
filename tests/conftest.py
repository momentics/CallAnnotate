# tests/conftest.py
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import sys
import os
import pytest
from unittest.mock import AsyncMock
from pathlib import Path

from fastapi.testclient import TestClient

# Добавляем src в PATH для корректного импорта приложения
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from app.app import app  # импорт FastAPI

@pytest.fixture(scope="session")
def client():
    """
    Фикстура для TestClient, позволяет переиспользовать одно приложение
    и выставлять scope session, чтобы избежать проблем с event loop.
    """
    return TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Настройка тестового окружения"""
    # Создаем временные директории для тестов
    test_volume_path = Path(__file__).parent / "test_volume"
    test_volume_path.mkdir(exist_ok=True)
    
    # Устанавливаем переменные окружения для тестов
    os.environ["VOLUME_PATH"] = str(test_volume_path)
    os.environ["MAX_FILE_SIZE"] = "10485760"  # 10MB для тестов
    
    yield
    
    # Очистка после тестов
    import shutil
    if test_volume_path.exists():
        shutil.rmtree(test_volume_path)

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Мокирование зависимостей для всех тестов"""
    # Мокируем queue_manager
    mock_queue_manager = AsyncMock()
    mock_queue_manager.get_queue_size.return_value = 0
    mock_queue_manager.get_active_tasks_count.return_value = 0
    mock_queue_manager.subscribe_to_task.return_value = None
    mock_queue_manager.add_task.return_value = True
    mock_queue_manager.start.return_value = None
    mock_queue_manager.stop.return_value = None
    
    # Мокируем annotation_service
    mock_annotation_service = AsyncMock()
    
    # Применяем моки через patching
    with pytest.MonkeyPatch().context() as m:
        m.setattr("app.queue_manager.QueueManager", lambda *args, **kwargs: mock_queue_manager)
        m.setattr("app.annotation.AnnotationService", lambda *args, **kwargs: mock_annotation_service)
        yield
