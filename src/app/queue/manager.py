# src/app/queue/manager.py

"""
Асинхронный менеджер очереди задач для CallAnnotate
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import asyncio
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

from ..core.interfaces.queue import QueueService
from ..annotation import AnnotationService
from ..utils import ensure_volume_structure, cleanup_temp_files


class TaskStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResult:
    __slots__ = (
        "task_id", "status", "message", "created_at", "updated_at",
        "progress", "result", "error", "metadata", "subscribers"
    )

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


class AsyncQueueManager(QueueService):
    def __init__(self, cfg: Dict[str, Any]):
        qcfg = cfg["queue"]
        self.volume = Path(qcfg["volume_path"]).expanduser().resolve()

        # Обеспечить создание всей структуры volume сразу
        ensure_volume_structure(str(self.volume))

        self.incoming = self.volume / "incoming"
        self.processing = self.volume / "processing"
        self.completed = self.volume / "completed"
        self.failed = self.volume / "failed"
        self.archived = self.volume / "archived"

        self.max_queue = qcfg["max_queue_size"]
        self.max_concurrent = qcfg["max_concurrent_tasks"]
        self.timeout = qcfg["task_timeout"]
        self.cleanup_interval = qcfg["cleanup_interval"]

        self._queue = asyncio.Queue(self.max_queue)
        self._tasks: Dict[str, TaskResult] = {}
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task] = []
        self._running = asyncio.Event()
        self._logger = logging.getLogger(__name__)
        self._cfg = cfg

        self._logger.info(f"AsyncQueueManager initialized with volume: {self.volume}")

    async def add_task(self, job_id: str, metadata: Dict[str, Any]) -> bool:
        async with self._lock:
            if job_id in self._tasks:
                return False
            self._tasks[job_id] = TaskResult(job_id, metadata)

        src = Path(metadata["file_path"]).expanduser().resolve()
        dest = (self.incoming / src.name).resolve()
        try:
            shutil.copy(src, dest)
            metadata["file_path"] = str(dest)
        except Exception as e:
            self._logger.warning(f"Не удалось скопировать файл в incoming: {e}")
            metadata["file_path"] = str(src)

        await self._queue.put(job_id)
        await self._notify_subscribers(job_id)
        return True

    async def cancel_task(self, job_id: str) -> bool:
        async with self._lock:
            tr = self._tasks.get(job_id)
            if not tr or tr.status not in {TaskStatus.QUEUED, TaskStatus.PROCESSING}:
                return False
            tr.status = TaskStatus.CANCELLED
            tr.updated_at = datetime.utcnow().isoformat()
        await self._notify_subscribers(job_id)
        return True

    async def get_task_result(self, job_id: str) -> Optional[TaskResult]:
        async with self._lock:
            return self._tasks.get(job_id)

    async def get_queue_info(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                "queue_length": self._queue.qsize(),
                "processing_jobs": [
                    tid for tid, t in self._tasks.items() if t.status == TaskStatus.PROCESSING
                ],
                "queued_jobs": [
                    tid for tid, t in self._tasks.items() if t.status == TaskStatus.QUEUED
                ]
            }

    async def subscribe_to_task(self, job_id: str, client_id: str):
        async with self._lock:
            tr = self._tasks.get(job_id)
            if tr:
                tr.subscribers.add(client_id)
        await self._notify_subscribers(job_id)

    async def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        for i in range(self.max_concurrent):
            self._workers.append(asyncio.create_task(self._worker_loop(i)))
        asyncio.create_task(self._cleanup_loop())
        self._logger.info("QueueManager started")

    async def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._logger.info("QueueManager stopped")

    async def _worker_loop(self, wid: int):
        while self._running.is_set():
            try:
                job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            tr = self._tasks[job_id]
            tr.status = TaskStatus.PROCESSING
            tr.updated_at = datetime.utcnow().isoformat()
            await self._notify_subscribers(job_id)

            src_path = Path(tr.metadata["file_path"]).expanduser().resolve()
            proc_path = self.processing / src_path.name
            try:
                shutil.move(str(src_path), str(proc_path))
                tr.metadata["file_path"] = str(proc_path)
            except Exception as e:
                self._logger.warning(f"Не удалось переместить в processing: {e}")

            try:
                ann = AnnotationService(self._cfg)
                result = await ann.process_audio(
                    tr.metadata["file_path"], job_id,
                    progress_callback=lambda p, m: self._update_progress(job_id, p, m)
                )
                tr.result = result
                tr.status = TaskStatus.COMPLETED
            except Exception as e:
                tr.error = str(e)
                tr.status = TaskStatus.FAILED

            final_dest = (self.completed if tr.status == TaskStatus.COMPLETED else self.failed) / proc_path.name
            try:
                shutil.move(str(proc_path), str(final_dest))
                tr.metadata["file_path"] = str(final_dest)
            except Exception as e:
                self._logger.warning(f"Не удалось переместить в завершённую папку: {e}")

            tr.updated_at = datetime.utcnow().isoformat()
            await self._notify_subscribers(job_id)
            self._queue.task_done()

            try:
                final_path = Path(tr.metadata["file_path"])
                archived_path = self.archived / final_path.name
                shutil.move(str(final_path), str(archived_path))
                tr.metadata["file_path"] = str(archived_path)
            except Exception as e:
                self._logger.warning(f"Не удалось переместить в archived: {e}")

    async def _update_progress(self, job_id: str, percent: int, _msg: str):
        async with self._lock:
            tr = self._tasks.get(job_id)
            if tr:
                tr.progress = percent
                tr.updated_at = datetime.utcnow().isoformat()
        await self._notify_subscribers(job_id)

    async def _notify_subscribers(self, job_id: str):
        from ..api.routers.ws import ws_manager

        tr = self._tasks.get(job_id)
        if not tr:
            return
        message = {
            "type": "status_update",
            "job_id": tr.task_id,
            "status": tr.status,
            "progress": tr.progress,
            "timestamp": tr.updated_at
        }
        for client_id in list(tr.subscribers):
            await ws_manager.send(client_id, message)

    async def _cleanup_loop(self):
        while self._running.is_set():
            await asyncio.sleep(self.cleanup_interval)
            cutoff = datetime.utcnow() - timedelta(days=7)
            async with self._lock:
                old = [
                    tid for tid, t in self._tasks.items()
                    if t.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
                    and datetime.fromisoformat(t.updated_at) < cutoff
                ]
                for tid in old:
                    del self._tasks[tid]
            # Очистка устаревших файлов из архива
            cleanup_temp_files(str(self.archived), max_age_hours=24 * 7)

    async def cleanup_once(self) -> None:
        """
        Однократный запуск логики очистки для тестов.
        Удаляет задачи, завершённые более 7 дней назад и старые файлы архива.
        """
        cutoff = datetime.utcnow() - timedelta(days=7)
        async with self._lock:
            old = [
                tid for tid, t in self._tasks.items()
                if t.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
                and datetime.fromisoformat(t.updated_at) < cutoff
            ]
            for tid in old:
                del self._tasks[tid]
        cleanup_temp_files(str(self.archived), max_age_hours=24 * 7)
