import os
import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_logfile(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    logs = vol / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lf = logs / "callannotate.log"
    lf.write_text("log")
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    monkeypatch.setenv("LOG_FILE_PATH", str(lf))
    app.state.volume_path = str(vol)

def test_download_log():
    r = client.get("/api/v1/logs/")
    assert r.status_code == 200
    assert "log" in r.text
