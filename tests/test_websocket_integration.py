# -*- coding: utf-8 -*-
"""
Интеграционные тесты WebSocket функциональности CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com
Лицензия: Apache-2.0
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from app.app import app

# ИСПРАВЛЕНО: используем позиционный аргумент вместо именованного
client = TestClient(app)


class TestWebSocketIntegration:
    """Тесты WebSocket интеграции"""

    def test_websocket_connection(self):
        """Тест подключения к WebSocket"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Отправляем ping
            ping_message = {"type": "ping", "timestamp": "2024-01-01T00:00:00"}
            websocket.send_text(json.dumps(ping_message))
            
            # Получаем pong
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "pong"
            assert response_data["timestamp"] == "2024-01-01T00:00:00"

    def test_websocket_task_subscription(self):
        """Тест подписки на задачу через WebSocket"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Подписываемся на задачу
            subscribe_message = {
                "type": "subscribe_task",
                "task_id": "test_task_123"
            }
            websocket.send_text(json.dumps(subscribe_message))
            
            # Получаем подтверждение подписки
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "subscribed"
            assert response_data["task_id"] == "test_task_123"
            assert "Subscribed to test_task_123" in response_data["message"]

    def test_websocket_audio_upload(self):
        """Тест загрузки аудио через WebSocket"""
        import base64
        
        # Создаем тестовые аудио данные
        test_audio_data = b"fake_audio_data_for_testing"
        encoded_audio = base64.b64encode(test_audio_data).decode('utf-8')
        
        with client.websocket_connect("/ws/test_client") as websocket:
            # Отправляем аудио
            upload_message = {
                "type": "upload_audio",
                "data": encoded_audio,
                "filename": "test_audio.wav",
                "priority": 5
            }
            websocket.send_text(json.dumps(upload_message))
            
            # Получаем подтверждение создания задачи
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "task_created"
            assert "task_id" in response_data
            assert response_data["status"] == "queued"

    def test_websocket_disconnect(self):
        """Тест корректного отключения WebSocket"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Отправляем сообщение для проверки соединения
            ping_message = {"type": "ping", "timestamp": "2024-01-01T00:00:00"}
            websocket.send_text(json.dumps(ping_message))
            
            # Получаем ответ
            response = websocket.receive_text()
            response_data = json.loads(response)
            assert response_data["type"] == "pong"
        
        # После выхода из контекста соединение должно быть закрыто
        # Это проверяется автоматически TestClient


@pytest.mark.asyncio
async def test_websocket_manager_functionality():
    """Тест функциональности WebSocketManager"""
    from app.app import WebSocketManager
    
    manager = WebSocketManager()
    
    # Мокаем WebSocket
    mock_websocket = AsyncMock()
    mock_websocket.accept = AsyncMock()
    mock_websocket.send_text = AsyncMock()
    
    # Тестируем подключение
    await manager.connect(mock_websocket, "test_client")
    assert "test_client" in manager.active_connections
    
    # Тестируем отправку сообщения
    test_message = {"type": "test", "data": "hello"}
    await manager.send_personal_message(test_message, "test_client")
    mock_websocket.send_text.assert_called_once_with(json.dumps(test_message))
    
    # Тестируем отключение
    manager.disconnect("test_client")
    assert "test_client" not in manager.active_connections


if __name__ == "__main__":
    pytest.main([__file__])
