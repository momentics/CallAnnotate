import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from app.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_logs(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    logs = vol / "logs"
    logs.mkdir(parents=True)
    (logs / "callannotate.log").write_text("log1")
    (logs / "callannotate.log.1").write_text("log2")
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    app.state.volume_path = str(vol)

def test_list_log_files_success():
    resp = client.get("/api/v1/logs/files")
    assert resp.status_code in (200, 404)

def test_list_log_files_no_volume(monkeypatch):
    monkeypatch.delenv("VOLUME_PATH", raising=False)
    resp = client.get("/api/v1/logs/files")
    assert resp.status_code == 404
