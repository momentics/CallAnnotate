# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

from ..core.interfaces.queue import QueueService
from ..annotation import AnnotationService
from ..utils import ensure_volume_structure


class TaskStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskResult:
    __slots__ = ("task_id","status","message","created_at","updated_at","progress","result","error","metadata","subscribers")
    def __init__(self, task_id: str, metadata: Dict[str, Any]):
        self.task_id = task_id
        self.status = TaskStatus.QUEUED
        self.message = "Queued"
        ts = datetime.utcnow().isoformat()
        self.created_at = ts
        self.updated_at = ts
        self.progress = 0
        self.result = None
        self.error = None
        self.metadata = metadata
        self.subscribers: set[str] = set()

# src/app/queue/manager.py

class AsyncQueueManager(QueueService):
    def __init__(self, cfg: Dict[str, Any]):
        qcfg = cfg["queue"]
        self.volume = Path(qcfg["volume_path"]).expanduser().resolve()

        # Обеспечить создание всей структуры volume сразу
        ensure_volume_structure(str(self.volume))

        self.incoming   = self.volume / "incoming"
        self.processing = self.volume / "processing"
        self.completed  = self.volume / "completed"
        self.failed     = self.volume / "failed"

        self.max_queue      = qcfg["max_queue_size"]
        self.max_concurrent = qcfg["max_concurrent_tasks"]
        self.timeout        = qcfg["task_timeout"]
        self.cleanup_interval = qcfg["cleanup_interval"]

        self._queue = asyncio.Queue(self.max_queue)
        self._tasks = {}
        self._lock = asyncio.Lock()
        self._workers = []
        self._running = asyncio.Event()
        self._logger = logging.getLogger(__name__)
        self._cfg = cfg

        self._logger.info(f"AsyncQueueManager initialized with volume: {self.volume}")

    async def add_task(self, job_id: str, metadata: Dict[str, Any]) -> bool:
        async with self._lock:
            if job_id in self._tasks: return False
            self._tasks[job_id] = TaskResult(job_id, metadata)
        await self._queue.put(job_id)
        return True

    async def cancel_task(self, job_id: str) -> bool:
        async with self._lock:
            tr = self._tasks.get(job_id)
            if not tr or tr.status not in {TaskStatus.QUEUED,TaskStatus.PROCESSING}: return False
            tr.status = TaskStatus.CANCELLED; tr.updated_at = datetime.utcnow().isoformat()
            return True

    async def get_task_result(self, job_id: str) -> Optional[TaskResult]:
        async with self._lock:
            return self._tasks.get(job_id)

    async def get_queue_info(self) -> Dict[str,Any]:
        async with self._lock:
            return {"queue_length":self._queue.qsize(),"processing_jobs":[tid for tid,t in self._tasks.items() if t.status==TaskStatus.PROCESSING],"queued_jobs":[tid for tid,t in self._tasks.items() if t.status==TaskStatus.QUEUED]}

    async def subscribe_to_task(self, job_id: str, client_id: str):
        async with self._lock:
            if job_id in self._tasks:
                self._tasks[job_id].subscribers.add(client_id)

    async def start(self) -> None:
        if self._running.is_set(): return
        self._running.set()
        for i in range(self.max_concurrent):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))
        asyncio.create_task(self._cleanup_loop()); self._logger.info("QueueManager started")

    async def stop(self) -> None:
        if not self._running.is_set(): return
        self._running.clear(); 
        for w in self._workers: w.cancel()
        await asyncio.gather(*self._workers,return_exceptions=True); self._logger.info("QueueManager stopped")

    async def _worker_loop(self, wid:int):
        while self._running.is_set():
            try:
                job_id = await asyncio.wait_for(self._queue.get(),timeout=1.0)
            except asyncio.TimeoutError:
                continue
            tr = self._tasks[job_id]; tr.status=TaskStatus.PROCESSING; tr.updated_at=datetime.utcnow().isoformat()
            try:
                ann = AnnotationService(self._cfg)
                result = await ann.process_audio(tr.metadata["file_path"],job_id,progress_callback=lambda p,m: self._update_progress(job_id,p,m))
                tr.result = result; tr.status=TaskStatus.COMPLETED
            except Exception as e:
                tr.error=str(e); tr.status=TaskStatus.FAILED
            finally:
                tr.updated_at=datetime.utcnow().isoformat(); self._queue.task_done()

    async def _update_progress(self, job_id:int, percent:int, _msg:str):
        async with self._lock:
            tr = self._tasks.get(job_id)
            if tr:
                tr.progress=percent; tr.updated_at=datetime.utcnow().isoformat()

    async def _cleanup_loop(self):
        while self._running.is_set():
            await asyncio.sleep(self.cleanup_interval)
            cutoff=datetime.utcnow()-timedelta(days=7)
            async with self._lock:
                old=[tid for tid,t in self._tasks.items() if t.status in {TaskStatus.COMPLETED,TaskStatus.FAILED,TaskStatus.CANCELLED} and datetime.fromisoformat(t.updated_at)<cutoff]
                for tid in old: del self._tasks[tid]

    async def cleanup_once(self) -> None:
        """
        Однократный запуск логики очистки для тестов.
        Удаляет задачи, завершённые более 7 дней назад.
        """
        from datetime import datetime, timedelta
        cutoff = datetime.utcnow() - timedelta(days=7)
        async with self._lock:
            old = [
                tid for tid, t in self._tasks.items()
                if t.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
                and datetime.fromisoformat(t.updated_at) < cutoff
            ]
            for tid in old:
                del self._tasks[tid]
