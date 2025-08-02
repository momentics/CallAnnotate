# tests/test_queue_file_movement.py

import os
import asyncio
import shutil
import tempfile
from pathlib import Path
import pytest

import app.queue.manager as qm_mod
from app.core.interfaces.queue import TaskResultProtocol

@pytest.fixture
def tmp_volume(tmp_path, monkeypatch):
    vol = tmp_path / "volume_test"
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    # Ensure structure
    qm_mod.AsyncQueueManager({"queue": {
        "volume_path": str(vol),
        "max_concurrent_tasks": 1,
        "max_queue_size": 10,
        "task_timeout": 1,
        "cleanup_interval": 60
    }})
    return vol

@pytest.mark.asyncio
async def test_add_task_copies_to_incoming(tmp_volume):
    # Create a source file outside incoming
    src_dir = tmp_volume / "src"
    src_dir.mkdir()
    src_file = src_dir / "test.wav"
    src_file.write_bytes(b"dummy audio")
    # Configure manager
    cfg = {"queue": {
        "volume_path": str(tmp_volume),
        "max_concurrent_tasks": 1,
        "max_queue_size": 5,
        "task_timeout": 1,
        "cleanup_interval": 60
    }}
    q = qm_mod.AsyncQueueManager(cfg)
    # Add task
    added = await q.add_task("job1", {"file_path": str(src_file), "filename": "test.wav", "priority":1})
    assert added
    # Incoming directory should contain a copy
    incoming = tmp_volume / "incoming" / "test.wav"
    assert incoming.exists()
    # Original remains
    assert src_file.exists()

@pytest.mark.asyncio
async def test_worker_moves_and_archives(tmp_volume, monkeypatch):
    # Prepare an audio file in incoming
    incoming_dir = tmp_volume / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    wav = incoming_dir / "move.wav"
    wav.write_bytes(b"\x00"*100)
    # Monkey-patch AnnotationService
    class FakeAnn:
        async def process_audio(self, fp, jid, progress_callback=None):
            return {"foo":"bar"}
    monkeypatch.setattr(qm_mod, "AnnotationService", lambda cfg: FakeAnn())

    cfg = {"queue": {
        "volume_path": str(tmp_volume),
        "max_concurrent_tasks": 1,
        "max_queue_size": 5,
        "task_timeout": 1,
        "cleanup_interval": 60
    }}
    q = qm_mod.AsyncQueueManager(cfg)
    # Add and start processing
    await q.add_task("job2", {"file_path": str(wav), "filename":"move.wav","priority":1})
    await q.start()
    # Allow the worker to run
    await asyncio.sleep(0.2)
    # After processing, file should be in completed or failed
    completed = tmp_volume / "completed" / "move.wav"
    assert completed.exists()
    # No file remains in incoming or processing
    assert not (tmp_volume / "incoming" / "move.wav").exists()
    assert not (tmp_volume / "processing" / "move.wav").exists()

    await q.stop()
