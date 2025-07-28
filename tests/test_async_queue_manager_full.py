# tests/test_async_queue_manager_full.py

import asyncio
import pytest
import os
from datetime import datetime, timedelta

import app.queue.manager as qm_mod

@pytest.fixture
def cfg(tmp_path, monkeypatch):
    vol = tmp_path / "vol"
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    data = {
        "queue": {
            "volume_path": str(vol),
            "max_concurrent_tasks": 1,
            "max_queue_size": 5,
            "task_timeout": 10,
            "cleanup_interval": 1
        }
    }
    return data

@pytest.mark.asyncio
async def test_worker_lifecycle_and_cleanup(cfg, monkeypatch):
    # мок AnnotationService
    class FakeAnn:
        async def process_audio(self, fp, jid, progress_callback=None):
            return {"ok": True}
    monkeypatch.setattr(qm_mod, "AnnotationService", lambda cfg: FakeAnn())

    q = qm_mod.AsyncQueueManager(cfg)
    # создаём файл
    vol = os.getenv("VOLUME_PATH")
    incoming = os.path.join(vol, "incoming")
    os.makedirs(incoming, exist_ok=True)
    filepath = os.path.join(incoming, "x.wav")
    open(filepath, "wb").write(b"1234")

    # добавляем и запускаем
    await q.add_task("t1", {"file_path": filepath})
    await q.start()
    await asyncio.sleep(0.1)

    tr = await q.get_task_result("t1")
    assert tr.status == qm_mod.TaskStatus.COMPLETED
    assert tr.result == {"ok": True}

    # демонтируем старые задачи
    old_time = (datetime.utcnow() - timedelta(days=8)).isoformat()
    tr.updated_at = old_time

    await q.cleanup_once()
    info = await q.get_queue_info()
    assert info["queue_length"] == 0

    await q.stop()
