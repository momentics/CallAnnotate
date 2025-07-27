# tests/test_websocket_integration.py
# Автор: akoodoy@capilСсылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import sys
import os
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Добавляем src в путь для корректного импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from app.app import create_app

# Создаем приложение для тестов
app = create_app()
client = TestClient(app)

@pytest.fixture
def mock_queue_manager():
    """Мок для queue_manager"""
    mock = AsyncMock()
    mock.get_queue_size.return_value = 0
    mock.get_active_tasks_count.return_value = 0
    mock.subscribe_to_task.return_value = None
    mock.add_task.return_value = True
    return mock

@pytest.fixture
def mock_annotation_service():
    """Мок для annotation_service"""
    mock = AsyncMock()
    return mock

def test_websocket_connection_and_heartbeat():
    """Тест установки соединения и heartbeat функциональности"""
    client_id = "test-client-123"
    
    with client.websocket_connect(f"/ws/{client_id}") as websocket:
        # Отправляем ping и ожидаем pong
        ping_message = {"type": "ping", "timestamp": "2025-01-27T00:00:00Z"}
        websocket.send_text(json.dumps(ping_message))
        
        # Получаем ответ
        response = websocket.receive_text()
        response_data = json.loads(response)
        
        assert response_data.get("type") == "pong"
        assert response_data.get("timestamp") == ping_message["timestamp"]

def test_websocket_task_subscription():
    """Тест подписки на задачу через WebSocket"""
    client_id = "test-client-456"
    task_id = "test-task-789"
    
    with client.websocket_connect(f"/ws/{client_id}") as websocket:
        # Отправляем запрос на подписку
        subscribe_message = {
            "type": "subscribe_task",
            "task_id": task_id
        }
        websocket.send_text(json.dumps(subscribe_message))
        
        # Получаем подтверждение подписки
        response = websocket.receive_text()
        response_data = json.loads(response)
        
        assert response_data.get("type") == "subscribed"
        assert response_data.get("task_id") == task_id
        assert "Subscribed to" in response_data.get("message", "")

def test_websocket_audio_upload():
    """Тест загрузки аудио через WebSocket"""
    client_id = "test-client-upload"
    
    # Создаем тестовые аудио данные (base64)
    import base64
    test_audio = b"fake_audio_data_for_testing"
    audio_base64 = base64.b64encode(test_audio).decode('utf-8')
    
    with client.websocket_connect(f"/ws/{client_id}") as websocket:
        # Отправляем аудио данные
        upload_message = {
            "type": "upload_audio",
            "data": audio_base64,
            "filename": "test_audio.wav",
            "priority": 3
        }
        websocket.send_text(json.dumps(upload_message))
        
        # Получаем подтверждение создания задачи
        response = websocket.receive_text()
        response_data = json.loads(response)
        
        assert response_data.get("type") == "task_created"
        assert response_data.get("status") == "queued"
        assert "task_id" in response_data

def test_websocket_connection_disconnect():
    """Тест корректного отключения WebSocket"""
    client_id = "test-client-disconnect"
    
    with client.websocket_connect(f"/ws/{client_id}") as websocket:
        # Отправляем ping для проверки соединения
        websocket.send_text(json.dumps({"type": "ping"}))
        response = websocket.receive_text()
        assert json.loads(response).get("type") == "pong"
        
        # Закрываем соединение
        websocket.close()
    
    # После закрытия соединения попытка использования должна вызвать ошибку
    # Это проверяется автоматически при выходе из контекстного менеджера

def test_websocket_invalid_message():
    """Тест обработки некорректных сообщений"""
    client_id = "test-client-invalid"
    
    with client.websocket_connect(f"/ws/{client_id}") as websocket:
        # Отправляем неизвестный тип сообщения
        invalid_message = {
            "type": "unknown_type",
            "data": "some_data"
        }
        websocket.send_text(json.dumps(invalid_message))
        
        # WebSocket должен остаться активным и отвечать на ping
        websocket.send_text(json.dumps({"type": "ping"}))
        response = websocket.receive_text()
        response_data = json.loads(response)
        
        assert response_data.get("type") == "pong"
