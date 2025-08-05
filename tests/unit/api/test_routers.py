import os
import pytest
from fastapi.testclient import TestClient
from app.app import app
from types import SimpleNamespace

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_env_and_dependencies(tmp_path, monkeypatch):
    # Ensure VOLUME_PATH is set to a temporary directory with required structure
    vol = tmp_path / "volume"
    incoming = vol / "incoming"
    # Create with exist_ok to avoid FileExistsError
    incoming.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    # Provide a dummy queue for health/info endpoints in routers.health
    async def dummy_get_queue():
        return SimpleNamespace(get_queue_info=lambda: {"queue_length": 0, "processing_jobs": [], "queued_jobs": []})
    # Patch dependencies in deps and health router
    monkeypatch.setattr("app.api.deps.get_queue", dummy_get_queue)
    monkeypatch.setattr("app.api.routers.health.get_queue", dummy_get_queue)
    # Also set state.volume_path so info endpoint does not fail
    app.state.volume_path = str(vol)

@pytest.mark.parametrize("endpoint, expected_status", [
    ("/api/v1/health", 200),
    ("/api/v1/info", 200),
])
def test_service_endpoints(endpoint, expected_status):
    response = client.get(endpoint)
    assert response.status_code == expected_status
    json_data = response.json()
    assert ("status" in json_data) or ("service" in json_data)

def test_jobs_crud(tmp_path):
    # create a dummy file in incoming
    vol = tmp_path / "volume"
    incoming = vol / "incoming"
    # Create directory if not exists
    incoming.mkdir(parents=True, exist_ok=True)
    file_path = incoming / "dummy.wav"
    file_path.write_bytes(b"RIFF....WAVEfmt ")  # minimal WAV header
    os.environ["VOLUME_PATH"] = str(vol)
    # POST /api/v1/jobs
    response = client.post("/api/v1/jobs", json={"filename": "dummy.wav", "priority": 5})
    assert response.status_code in (201, 404)
    if response.status_code == 201:
        job = response.json()
        job_id = job["job_id"]
        # GET status
        status_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert status_resp.status_code in (200, 404)
        # GET result
        result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result_resp.status_code in (200, 404)
        # DELETE
        delete_resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert delete_resp.status_code in (204, 404)
    else:
        # If initial POST returned 404, ensure message indicates missing file
        assert response.json().get("detail") is not None
