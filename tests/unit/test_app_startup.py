import os
import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_health_and_info_endpoints(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    incoming = vol/"incoming"; incoming.mkdir(parents=True)
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    app.state.volume_path = str(vol)

    resp_h = client.get("/api/v1/health")
    assert resp_h.status_code == 200
    resp_i = client.get("/api/v1/info")
    assert resp_i.status_code == 200
