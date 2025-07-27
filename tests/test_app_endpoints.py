# tests/test_app_endpoints.py
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0


import os
import time
from fastapi import status

def test_health_and_info(client):
    # Health
    res = client.get("/health")
    assert res.status_code == status.HTTP_200_OK
    body = res.json()
    assert body["status"] == "healthy"
    assert "queue_length" in body

    # Info
    res = client.get("/info")
    assert res.status_code == status.HTTP_200_OK
    info = res.json()
    assert info["service"] == "CallAnnotate"
    assert info["max_file_size"] == int(os.getenv("MAX_FILE_SIZE", 1073741824))

def test_create_job_too_large(client, monkeypatch, tmp_path):
    # Monkey-patch MAX_FILE_SIZE малый
    monkeypatch.setenv("MAX_FILE_SIZE", "2")
    large = tmp_path / "big.mp3"
    large.write_bytes(b"\x00\x01\x02")
    with open(large, "rb") as f:
        res = client.post("/jobs", files={"file": ("big.mp3", f, "audio/mp3")})
    assert res.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

def test_create_job_unsupported_type(client, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("not audio")
    with open(bad, "rb") as f:
        res = client.post("/jobs", files={"file": ("bad.txt", f, "text/plain")})
    assert res.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

def test_delete_job_endpoint(client, tmp_path):
    # Создаём и ждем jobb
    audio = tmp_path / "d.wav"
    audio.write_bytes(b"\x00")
    with open(audio, "rb") as f:
        res = client.post("/jobs", files={"file": ("d.wav", f, "audio/wav")})
    job_id = res.json()["job_id"]
    time.sleep(2)
    # Удаляем
    res = client.delete(f"/jobs/{job_id}")
    assert res.status_code == status.HTTP_204_NO_CONTENT
    # Повторный DELETE даёт 404
    res = client.delete(f"/jobs/{job_id}")
    assert res.status_code == status.HTTP_404_NOT_FOUND
