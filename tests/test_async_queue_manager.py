# tests/test_async_queue_manager.py

import asyncio
import pytest
from datetime import datetime, timedelta

import app.queue.manager as qm_mod
from app.core.interfaces.queue import TaskResultProtocol


@pytest.fixture
def tmp_cfg(tmp_path, monkeypatch):
    vol = tmp_path / "volume_test"
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    cfg = {
        "queue": {
            "volume_path": str(vol),
            "max_concurrent_tasks": 1,
            "max_queue_size": 10,
            "task_timeout": 1,
            "cleanup_interval": 1
        }
    }
    return cfg


@pytest.mark.asyncio
async def test_add_get_cancel_queue_info(tmp_cfg):
    q = qm_mod.AsyncQueueManager(tmp_cfg)
    # before start
    assert await q.get_queue_info() == {
        "queue_length": 0,
        "processing_jobs": [],
        "queued_jobs": []
    }
    # add task
    ok = await q.add_task("job1", {"file_path": "p", "filename": "f", "priority": 1})
    assert ok
    info = await q.get_queue_info()
    assert info["queue_length"] == 1
    # get
    tr = await q.get_task_result("job1")
    assert isinstance(tr, TaskResultProtocol)
    assert tr.task_id == "job1"
    # cancel
    ok2 = await q.cancel_task("job1")
    assert ok2
    tr2 = await q.get_task_result("job1")
    assert tr2.status == qm_mod.TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_start_and_cleanup(tmp_cfg, monkeypatch):
    # patch AnnotationService to return immediately
    class FakeAnn:
        async def process_audio(self, fp, jid, progress_callback=None):
            return {"foo": "bar"}
    monkeypatch.setattr(qm_mod, "AnnotationService", lambda cfg: FakeAnn())

    q = qm_mod.AsyncQueueManager(tmp_cfg)
    # add two tasks
    await q.add_task("j1", {"file_path": "x"})
    await q.add_task("j2", {"file_path": "y"})
    await q.start()
    # allow worker to pick up tasks
    await asyncio.sleep(0.1)
    # both should be processed
    tr1 = await q.get_task_result("j1")
    tr2 = await q.get_task_result("j2")
    assert tr1.status == qm_mod.TaskStatus.COMPLETED
    assert tr2.status == qm_mod.TaskStatus.COMPLETED

    # simulate old tasks
    cutoff = (datetime.utcnow() - timedelta(days=8)).isoformat()
    tr1.updated_at = cutoff
    tr2.updated_at = cutoff

    # explicit cleanup_once
    await q.cleanup_once()
    # now tasks removed
    assert await q.get_task_result("j1") is None
    assert await q.get_task_result("j2") is None

    await q.stop()
