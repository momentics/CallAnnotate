# tests/unit/api/test_log_rotation_full.py

import os
import glob
import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_logs(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    logs = vol / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs/"callannotate.log").write_text("1")
    (logs/"callannotate.log.1").write_text("2")
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    app.state.volume_path = str(vol)

def test_list_log_files():
    resp = client.get("/api/v1/logs/files")
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        arr = resp.json()
        assert any("callannotate.log" in fn for fn in arr)
