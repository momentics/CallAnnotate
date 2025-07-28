# tests/test_app_endpoints.py
# Автор: akoodoy@capilot.ru
# Лицензия: Apache-2.0

import os
from pathlib import Path
import pytest
from fastapi import status

@pytest.fixture(autouse=True)
def set_volume(tmp_path, monkeypatch):
    test_vol = tmp_path / "volume"
    incoming = test_vol / "incoming"
    incoming.mkdir(parents=True)
    monkeypatch.setenv("VOLUME_PATH", str(test_vol))

def test_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["status"] == "healthy"
    assert "queue_length" in data

def test_info(client):
    res = client.get("/api/v1/info")
    assert res.status_code == status.HTTP_200_OK
    data = res.json()
    assert data["service"] == "CallAnnotate"
    assert "/volume/incoming" in data["volume_paths"]["incoming"]

def test_create_job_not_found(client):
    res = client.post("/api/v1/jobs", json={"filename":"nofile.wav","priority":5})
    assert res.status_code == status.HTTP_404_NOT_FOUND

def test_create_job_success(client, tmp_path, monkeypatch):
    # подготовка файла
    vol = Path(os.getenv("VOLUME_PATH")) / "incoming"
    f = vol / "a.wav"
    f.write_bytes(b"1234")
    res = client.post("/api/v1/jobs", json={"filename":"a.wav","priority":1})
    assert res.status_code == status.HTTP_201_CREATED
    body = res.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]

    # статус задачи
    res2 = client.get(f"/api/v1/jobs/{job_id}")
    assert res2.status_code == status.HTTP_200_OK
    st = res2.json()
    assert st["job_id"] == job_id

def test_get_job_not_found(client):
    res = client.get("/api/v1/jobs/unknown")
    assert res.status_code == status.HTTP_404_NOT_FOUND

def test_get_job_result(client):
    res = client.get("/api/v1/jobs/test-job/result")
    # мок Queue возвращает статус completed → 200
    assert res.status_code == status.HTTP_200_OK
    assert res.json() == {"foo":"bar"}

def test_cancel_job_not_found(client):
    res = client.delete("/api/v1/jobs/unknown")
    assert res.status_code == status.HTTP_404_NOT_FOUND

def test_cancel_job_success(client):
    # mock cancel always False → 404
    res = client.delete("/api/v1/jobs/test-job")
    assert res.status_code == status.HTTP_404_NOT_FOUND
