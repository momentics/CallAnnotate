# tests/test_app_endpoints.py
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

from fastapi import status
import pytest


def test_health_and_info(client):
    """Тест эндпоинтов health и info"""
    # Health check
    res = client.get("/health")
    assert res.status_code == status.HTTP_200_OK
    body = res.json()
    assert body["status"] == "healthy"
    assert "queue_length" in body
    assert "components" in body

    # Info
    res = client.get("/info")
    assert res.status_code == status.HTTP_200_OK
    info = res.json()
    assert info["service"] == "CallAnnotate"
    assert "supported_formats" in info
    assert "volume_paths" in info


def test_create_job_file_not_found(client):
    """Тест создания задачи с несуществующим файлом"""
    request_data = {
        "filename": "nonexistent.wav",
        "priority": 5
    }
    
    res = client.post("/api/v1/jobs", json=request_data)
    assert res.status_code == status.HTTP_404_NOT_FOUND
    error = res.json()
    assert "not found" in error["detail"].lower()


def test_create_job_success(client, tmp_path, monkeypatch):
    """Тест успешного создания задачи"""
    # Настройка тестового volume
    test_volume = tmp_path / "test_volume"
    incoming_dir = test_volume / "incoming"
    incoming_dir.mkdir(parents=True)
    
    # Создание тестового аудиофайла
    test_file = incoming_dir / "test.wav"
    test_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")  # Минимальный WAV заголовок
    
    # Patch переменной окружения
    monkeypatch.setenv("VOLUME_PATH", str(test_volume))
    
    request_data = {
        "filename": "test.wav",
        "priority": 7
    }
    
    res = client.post("/api/v1/jobs", json=request_data)
    assert res.status_code == status.HTTP_201_CREATED
    
    response_data = res.json()
    assert "job_id" in response_data
    assert response_data["status"] == "queued"
    assert response_data["file_info"]["filename"] == "test.wav"


def test_create_job_file_too_large(client, tmp_path, monkeypatch):
    """Тест создания задачи с слишком большим файлом"""
    # Настройка маленького лимита
    monkeypatch.setenv("MAX_FILE_SIZE", "100")
    
    test_volume = tmp_path / "test_volume"
    incoming_dir = test_volume / "incoming"
    incoming_dir.mkdir(parents=True)
    
    # Создание большого файла
    large_file = incoming_dir / "large.wav"
    large_file.write_bytes(b"RIFF" + b"\x00" * 200)  # Больше 100 байт
    
    monkeypatch.setenv("VOLUME_PATH", str(test_volume))
    
    request_data = {
        "filename": "large.wav"
    }
    
    res = client.post("/api/v1/jobs", json=request_data)
    assert res.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


def test_create_job_invalid_json(client):
    """Тест создания задачи с невалидным JSON"""
    res = client.post("/api/v1/jobs", json={})  # Отсутствует filename
    assert res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_job_not_found(client):
    """Тест получения несуществующей задачи"""
    res = client.get("/api/v1/jobs/nonexistent-job-id")
    assert res.status_code == status.HTTP_404_NOT_FOUND


def test_get_job_result_not_found(client):
    """Тест получения результата несуществующей задачи"""
    res = client.get("/api/v1/jobs/nonexistent-job-id/result")
    assert res.status_code == status.HTTP_404_NOT_FOUND


def test_delete_job_not_found(client):
    """Тест удаления несуществующей задачи"""
    res = client.delete("/api/v1/jobs/nonexistent-job-id")
    assert res.status_code == status.HTTP_404_NOT_FOUND


def test_queue_status(client):
    """Тест получения статуса очереди"""
    res = client.get("/api/v1/queue/status")
    assert res.status_code == status.HTTP_200_OK
    
    queue_status = res.json()
    assert "queue_length" in queue_status
    assert "processing_jobs" in queue_status
    assert "queued_jobs" in queue_status


@pytest.mark.asyncio
async def test_job_workflow(client, tmp_path, monkeypatch):
    """Интеграционный тест полного рабочего процесса"""
    # Настройка тестового окружения
    test_volume = tmp_path / "test_volume"
    incoming_dir = test_volume / "incoming"
    incoming_dir.mkdir(parents=True)
    
    test_file = incoming_dir / "workflow_test.wav"
    test_file.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    
    monkeypatch.setenv("VOLUME_PATH", str(test_volume))
    
    # 1. Создание задачи
    create_response = client.post("/api/v1/jobs", json={
        "filename": "workflow_test.wav",
        "priority": 5
    })
    assert create_response.status_code == status.HTTP_201_CREATED
    
    job_data = create_response.json()
    job_id = job_data["job_id"]
    
    # 2. Проверка статуса
    status_response = client.get(f"/api/v1/jobs/{job_id}")
    assert status_response.status_code == status.HTTP_200_OK
    
    status_data = status_response.json()
    assert status_data["job_id"] == job_id
    assert status_data["status"] in ["queued", "processing"]
    
    # 3. Попытка получения результата (должен быть еще не готов)
    result_response = client.get(f"/api/v1/jobs/{job_id}/result")
    # Результат может быть 202 (не готов) или 404 (не найден в моке)
    assert result_response.status_code in [status.HTTP_202_ACCEPTED, status.HTTP_404_NOT_FOUND]
    
    # 4. Удаление задачи
    delete_response = client.delete(f"/api/v1/jobs/{job_id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT
    
    # 5. Проверка что задача удалена
    final_status = client.get(f"/api/v1/jobs/{job_id}")
    assert final_status.status_code == status.HTTP_404_NOT_FOUND
