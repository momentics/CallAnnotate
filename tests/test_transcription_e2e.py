import pytest
from fastapi.testclient import TestClient
from app.app import app
import os

@pytest.fixture(autouse=True)
def setup_volume(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    (vol / "incoming").mkdir(parents=True)
    # создаём фиктивный WAV
    f = vol / "incoming" / "test.wav"
    f.write_bytes(b"\x00" * 1024)

def test_transcription_endpoint_end_to_end():
    client = TestClient(app)
    # отправляем файл
    res = client.post("/api/v1/jobs", json={"filename":"test.wav","priority":5})
    assert res.status_code == 201
    job_id = res.json()["job_id"]

    # ждём обработки (фиктивный queue)
    import time; time.sleep(0.1)

    res2 = client.get(f"/api/v1/jobs/{job_id}/result")
    assert res2.status_code == 200
    body = res2.json()
    assert "transcription" in body
    t = body["transcription"]
    assert "confidence" in t
    assert "segments" in t
    assert "processing_time" in t

