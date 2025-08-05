# tests/unit/api/test_logging_router.py

import os
import pytest
from fastapi.testclient import TestClient
from app.app import app
from pathlib import Path

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_log_file(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    logs = vol / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_file = logs / "callannotate.log"
    log_file.write_text("test log content")
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    monkeypatch.setenv("LOG_FILE_PATH", str(log_file))
    app.state.volume_path = str(vol)

def test_download_log_success():
    response = client.get("/api/v1/logs/")
    assert response.status_code == 200
    assert response.text == "test log content"
    content_disp = response.headers.get("content-disposition", "")
    assert "callannotate.log" in content_disp
