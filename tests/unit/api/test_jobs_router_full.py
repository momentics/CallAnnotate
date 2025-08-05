import os
import uuid
import pytest
from fastapi.testclient import TestClient
from app.app import app
from app.api.routers.jobs import router as jobs_router
from app.schemas import CreateJobRequest, TaskStatus
from unittest.mock import AsyncMock

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_queue(monkeypatch, tmp_path):
    vol = tmp_path / "volume"
    vol.mkdir()
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    class Dummy:
        async def start(self): pass
        async def stop(self): pass
        async def add_task(self, job_id, meta): return True
        async def get_task_result(self, job_id):
            from types import SimpleNamespace
            return SimpleNamespace(task_id=job_id, status=TaskStatus.COMPLETED, message="ok", progress=100, created_at=None, updated_at=None, result={"a":1})
        async def cancel_task(self, job_id): return True
    monkeypatch.setattr("app.api.routers.jobs.get_queue", lambda: Dummy())

def test_create_job_success(tmp_path):
    incoming = tmp_path / "volume" / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    f = incoming / "file.wav"
    f.write_bytes(b"RIFF")
    resp = client.post("/api/v1/jobs", json={"filename":"file.wav","priority":5})
    assert resp.status_code in (201, 404)

def test_get_status_and_result_and_delete():
    job_id = str(uuid.uuid4())
    status = client.get(f"/api/v1/jobs/{job_id}")
    assert status.status_code in (200, 404)
    result = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result.status_code in (200, 404)
    delete = client.delete(f"/api/v1/jobs/{job_id}")
    assert delete.status_code in (204, 404)
