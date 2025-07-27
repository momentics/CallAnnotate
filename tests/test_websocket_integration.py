# -*- coding: utf-8 -*-
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from app.app import app

client = TestClient(app)


class TestWebSocketIntegration:
    """Тесты WebSocket интеграции с корректным API"""

    def test_websocket_connection_and_ping(self):
        """Тест подключения к WebSocket и ping/pong"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Отправляем ping
            ping_message = {"type": "ping", "timestamp": "2025-07-27T20:00:00Z"}
            websocket.send_text(json.dumps(ping_message))
            
            # Получаем pong
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "pong"
            assert response_data["timestamp"] == "2025-07-27T20:00:00Z"

    def test_websocket_job_subscription(self):
        """Тест подписки на задачу через WebSocket"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Подписываемся на задачу
            subscribe_message = {
                "type": "subscribe_job",
                "job_id": "550e8400-e29b-41d4-a716-446655440000"
            }
            websocket.send_text(json.dumps(subscribe_message))
            
            # Получаем подтверждение подписки
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "subscribed"
            assert response_data["job_id"] == "550e8400-e29b-41d4-a716-446655440000"

    def test_websocket_create_job_file_not_found(self):
        """Тест создания задачи через WebSocket с несуществующим файлом"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Отправляем запрос на создание задачи с несуществующим файлом
            create_message = {
                "type": "create_job",
                "filename": "nonexistent.wav",
                "priority": 5
            }
            websocket.send_text(json.dumps(create_message))
            
            # Получаем ошибку
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "error"
            assert response_data["code"] == "FILE_NOT_FOUND"
            assert "not found" in response_data["message"]

    def test_websocket_create_job_success(self, tmp_path, monkeypatch):
        """Тест успешного создания задачи через WebSocket"""
        # Настройка тестового volume
        test_volume = tmp_path / "test_volume"
        incoming_dir = test_volume / "incoming"
        incoming_dir.mkdir(parents=True)
        
        # Создание тестового аудиофайла
        test_file = incoming_dir / "ws_test.wav"
        test_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        
        monkeypatch.setenv("VOLUME_PATH", str(test_volume))
        
        with client.websocket_connect("/ws/test_client") as websocket:
            # Создание задачи
            create_message = {
                "type": "create_job",
                "filename": "ws_test.wav",
                "priority": 8
            }
            websocket.send_text(json.dumps(create_message))
            
            # Получаем подтверждение создания
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "job_created"
            assert "job_id" in response_data
            assert response_data["status"] == "queued"

    def test_websocket_create_job_missing_filename(self):
        """Тест создания задачи без указания filename"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Отправляем неполное сообщение
            create_message = {
                "type": "create_job",
                "priority": 5
                # filename отсутствует
            }
            websocket.send_text(json.dumps(create_message))
            
            # Получаем ошибку
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "error"
            assert response_data["code"] == "MISSING_FILENAME"

    def test_websocket_invalid_json(self):
        """Тест отправки невалидного JSON"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Отправляем невалидный JSON
            websocket.send_text("invalid json {")
            
            # Получаем ошибку
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "error"
            assert response_data["code"] == "INVALID_JSON"

    def test_websocket_unknown_message_type(self):
        """Тест неизвестного типа сообщения"""
        with client.websocket_connect("/ws/test_client") as websocket:
            # Отправляем неизвестный тип сообщения
            unknown_message = {
                "type": "unknown_type",
                "data": "test"
            }
            websocket.send_text(json.dumps(unknown_message))
            
            # Получаем ошибку
            response = websocket.receive_text()
            response_data = json.loads(response)
            
            assert response_data["type"] == "error"
            assert response_data["code"] == "UNKNOWN_MESSAGE_TYPE"
            assert "unknown_type" in response_data["message"]

    def test_websocket_multiple_clients(self):
        """Тест подключения нескольких клиентов"""
        with client.websocket_connect("/ws/client_1") as ws1, \
             client.websocket_connect("/ws/client_2") as ws2:
            
            # Каждый клиент отправляет ping
            ping1 = {"type": "ping", "timestamp": "2025-07-27T20:00:00Z"}
            ping2 = {"type": "ping", "timestamp": "2025-07-27T20:01:00Z"}
            
            ws1.send_text(json.dumps(ping1))
            ws2.send_text(json.dumps(ping2))
            
            # Получаем ответы
            response1 = json.loads(ws1.receive_text())
            response2 = json.loads(ws2.receive_text())
            
            assert response1["type"] == "pong"
            assert response1["timestamp"] == "2025-07-27T20:00:00Z"
            
            assert response2["type"] == "pong"
            assert response2["timestamp"] == "2025-07-27T20:01:00Z"


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
    
    # Тестируем broadcast
    await manager.broadcast({"type": "broadcast", "message": "to all"})
    
    # Тестируем отключение
    manager.disconnect("test_client")
    assert "test_client" not in manager.active_connections


@pytest.mark.asyncio 
async def test_websocket_job_workflow(tmp_path, monkeypatch):
    """Интеграционный тест WebSocket workflow"""
    # Настройка тестового окружения
    test_volume = tmp_path / "test_volume"
    incoming_dir = test_volume / "incoming"
    incoming_dir.mkdir(parents=True)
    
    test_file = incoming_dir / "workflow.wav"
    test_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    
    monkeypatch.setenv("VOLUME_PATH", str(test_volume))
    
    with client.websocket_connect("/ws/workflow_client") as websocket:
        # 1. Создание задачи
        websocket.send_text(json.dumps({
            "type": "create_job",
            "filename": "workflow.wav",
            "priority": 5
        }))
        
        # Получение подтверждения
        create_response = json.loads(websocket.receive_text())
        assert create_response["type"] == "job_created"
        job_id = create_response["job_id"]
        
        # 2. Подписка на обновления
        websocket.send_text(json.dumps({
            "type": "subscribe_job",
            "job_id": job_id
        }))
        
        # Получение подтверждения подписки
        subscribe_response = json.loads(websocket.receive_text())
        assert subscribe_response["type"] == "subscribed"
        assert subscribe_response["job_id"] == job_id


if __name__ == "__main__":
    pytest.main([__file__])
