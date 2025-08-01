# tests/test_transcription_e2e.py

import pytest
from fastapi.testclient import TestClient
from app.app import app
import os

@pytest.fixture(autouse=True)
def setup_volume(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    incoming = vol / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    # создаём фиктивный WAV
    f = incoming / "test.wav"
    f.write_bytes(b"\x00" * 1024)

def test_transcription_endpoint_end_to_end(monkeypatch):
    # Мокаем внутреннюю очередь и AnnotationService
    client = TestClient(app)
    # POST на создание задачи
    res = client.post("/api/v1/jobs", json={"filename": "test.wav", "priority": 5})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    # Ждем обновления через WS (мок Queue)
    import time
    time.sleep(0.1)

    # Получаем результат
    res2 = client.get(f"/api/v1/jobs/{job_id}/result")
    assert res2.status_code == 200
    body = res2.json()
    # Проверяем наличие полей транскрипции
    assert "transcription" in body
    t = body["transcription"]
    assert "confidence" in t
    assert "segments" in t
    assert "processing_time" in t
