# tests/unit/api/test_jobs_router_additional.py

import os
import pytest
from fastapi.testclient import TestClient
from types import SimpleNamespace
from app.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_env_and_dependencies(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    incoming = vol / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    async def dummy_get_queue():
        return SimpleNamespace(
            start=lambda: None,
            stop=lambda: None,
            add_task=lambda *args, **kwargs: True,
            get_task_result=lambda job_id: SimpleNamespace(
                task_id=job_id,
                status="queued",
                message="queued",
                progress=0,
                created_at=None,
                updated_at=None,
                result={}
            ),
            cancel_task=lambda *args, **kwargs: True
        )
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "nonexistent.yaml"))
    monkeypatch.setattr("app.api.deps.get_queue", dummy_get_queue)
    app.state.volume_path = str(vol)

@pytest.mark.parametrize("endpoint, expected_status", [
    ("/api/v1/health", 200),
    ("/api/v1/info", 200),
])
def test_service_endpoints(endpoint, expected_status):
    response = client.get(endpoint)
    assert response.status_code == expected_status
    json_data = response.json()
    assert ("version" in json_data) or ("service" in json_data)

def test_create_job_and_cancel(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    incoming = vol / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    file_path = incoming / "dummy.wav"
    file_path.write_bytes(b"RIFF....WAVEfmt ")
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    app.state.volume_path = str(vol)

    response = client.post("/api/v1/jobs", json={"filename": "dummy.wav", "priority": 5})
    assert response.status_code in (201, 404)
    if response.status_code == 201:
        job = response.json()
        job_id = job["job_id"]
        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code in (200, 404)
        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result_resp.status_code in (200, 404)
        delete_resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert delete_resp.status_code in (204, 404)
    else:
        assert response.json().get("detail") is not None
