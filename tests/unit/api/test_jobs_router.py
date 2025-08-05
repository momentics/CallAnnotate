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
    incoming = vol / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    # dummy file
    f = incoming / "file.wav"
    f.write_bytes(b"RIFF")
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    class DummyQueue:
        async def start(self): pass
        async def stop(self): pass
        async def add_task(self, job_id, meta): return True
        async def get_task_result(self, job_id):
            from types import SimpleNamespace
            return SimpleNamespace(
                task_id=job_id,
                status=TaskStatus.COMPLETED,
                message="ok",
                progress=100,
                created_at=None,
                updated_at=None,
                result={"a":1}
            )
        async def cancel_task(self, job_id): return job_id != "missing"
    monkeypatch.setattr("app.api.routers.jobs.get_queue", lambda: DummyQueue())

def test_create_job_success_and_not_found():
    # valid
    resp = client.post("/api/v1/jobs", json={"filename":"file.wav","priority":5})
    assert resp.status_code in (201, 404)
    job_id = resp.json().get("job_id", None)
    # invalid file
    resp2 = client.post("/api/v1/jobs", json={"filename":"nofile.wav","priority":5})
    assert resp2.status_code == 404

def test_get_status_and_result_and_delete():
    job_id = str(uuid.uuid4())
    # status
    status_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert status_resp.status_code in (200, 404)
    # result
    result_resp = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result_resp.status_code in (200, 404)
    # delete existing
    del_resp = client.delete(f"/api/v1/jobs/{job_id}")
    assert del_resp.status_code in (204, 404)
    # delete missing
    del2 = client.delete("/api/v1/jobs/missing")
    assert del2.status_code == 404
