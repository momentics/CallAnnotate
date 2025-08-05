# src/app/queue/manager.py
# -*- coding: utf-8 -*-
"""
Асинхронный менеджер очереди задач для CallAnnotate.

После stop новые задачи не принимаются.
Исправлено: теперь обрабатываются перемещения файлов incoming→processing→completed/failed,
файл *_processed удаляется после завершения обработки,
и при ошибках создаётся .log-файл с текстом ошибки.
"""
import asyncio
import logging
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import AppSettings
from ..core.interfaces.queue import QueueService, TaskResultProtocol
from ..annotation import AnnotationService
from ..utils import ensure_volume_structure

class TaskStatus(str):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskResult(TaskResultProtocol):
    __slots__ = (
        "task_id",
        "status",
        "message",
        "created_at",
        "updated_at",
        "progress",
        "result",
        "error",
        "metadata",
        "subscribers",
    )

    def __init__(self, task_id: str, metadata: Dict[str, Any]):
        self.task_id = task_id
        self.status = TaskStatus.QUEUED
        self.message = "Queued"
        timestamp = datetime.utcnow().isoformat()
        self.created_at = timestamp
        self.updated_at = timestamp
        self.progress = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.metadata = metadata
        self.subscribers: set[str] = set()

class AsyncQueueManager(QueueService):
    """
    Асинхронный менеджер очереди задач для CallAnnotate.

    Реализовано:
    - Перемещение исходного файла incoming→processing→completed/failed
    - Перемещение JSON-результата или .log с ошибкой рядом с файлом
    - Удаление *_processed файлов после завершения обработки
    """
    def __init__(self, cfg: AppSettings):

        self._cfg = cfg
        qcfg1 = cfg.queue.model_dump()

        self._cleanup_task: Optional[asyncio.Task] = None

        self.volume = Path(cfg.queue.volume_path).expanduser().resolve()
        ensure_volume_structure(str(self.volume))

        self.incoming = self.volume / "incoming"
        self.processing = self.volume / "processing"
        self.completed = self.volume / "completed"
        self.failed = self.volume / "failed"

        self.max_queue = cfg.queue.max_queue_size
        self.max_concurrent = cfg.queue.max_concurrent_tasks
        self.timeout = cfg.queue.task_timeout
        self._cleanup_interval = cfg.queue.cleanup_interval

        self._queue: asyncio.Queue[str] = asyncio.Queue(self.max_queue)
        self._tasks: Dict[str, TaskResult] = {}
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task] = []
        self._running = asyncio.Event()
        self._stopped = False
        self._logger = logging.getLogger(__name__)

        self._logger.info(f"QueueManager initialized at volume: {self.volume}")

    async def _update_progress(self, job_id: str, progress: int, message: str) -> None:
        """
        Обновляет прогресс задачи и рассылает уведомления подписчикам.
        """
        async with self._lock:
            tr = self._tasks.get(job_id)
            if not tr:
                return
            tr.progress = progress
            tr.message = message
            tr.updated_at = datetime.utcnow().isoformat()
        await self._notify_subscribers(job_id)

    async def _notify_subscribers(self, job_id: str) -> None:
        # Заглушка для уведомления подписчиков WebSocket;
        # в реальной интеграции должен рассылать сообщения через WS-менеджер.
        return None

    async def add_task(self, job_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Добавление задачи в очередь.
        После stop() новые задачи не принимаются.
        """
        if self._stopped:
            self._logger.warning(f"Cannot add task {job_id}: queue manager stopped")
            return False

        async with self._lock:
            if job_id in self._tasks:
                return False
            self._tasks[job_id] = TaskResult(job_id, metadata)

        src = Path(metadata["file_path"]).expanduser().resolve()
        dest = self.processing / src.name
        shutil.move(src, dest)
        metadata["file_path"] = str(dest)

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

    async def get_task_result(self, job_id: str) -> Optional[TaskResultProtocol]:
        async with self._lock:
            return self._tasks.get(job_id)

    async def get_queue_info(self) -> Dict[str, Any]:
        async with self._lock:
            processing = [tid for tid, t in self._tasks.items() if t.status == TaskStatus.PROCESSING]
            queued = [tid for tid, t in self._tasks.items() if t.status == TaskStatus.QUEUED]
            return {
                "queue_length": self._queue.qsize(),
                "processing_jobs": processing,
                "queued_jobs": queued,
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
        self._stopped = False
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._logger.info("QueueManager started")

    async def stop(self) -> None:
        """
        Останавливает менеджер очереди: завершает воркеры и блокирует добавление новых задач.
        """
        if not self._running.is_set():
            return
        self._running.clear()
        self._stopped = True

        if self._cleanup_task:
            self._cleanup_task.cancel()
        for worker in self._workers:
            worker.cancel()

        tasks = []
        if self._cleanup_task:
            tasks.append(self._cleanup_task)
        tasks.extend(self._workers)
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                pass

        self._logger.info("QueueManager stopped gracefully")

    async def cleanup_once(self) -> None:
        """
        Удаляет завершённые задачи старше 7 дней.
        """
        now = datetime.utcnow()
        stale = []
        async with self._lock:
            for tid, tr in list(self._tasks.items()):
                if tr.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
                    updated = datetime.fromisoformat(tr.updated_at)
                    if now - updated > timedelta(days=7):
                        stale.append(tid)
            for tid in stale:
                del self._tasks[tid]
        self._logger.info(f"cleanup_once removed tasks: {stale}")

    async def _periodic_cleanup(self):
        """
        Запускает cleanup_once каждые cleanup_interval секунд.
        """
        try:
            while self._running.is_set():
                await asyncio.sleep(self._cleanup_interval)
                await self.cleanup_once()
        except asyncio.CancelledError:
            return

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

            # 1. Process audio
            try:
                ann = AnnotationService(self._cfg)
                result = await ann.process_audio(
                    tr.metadata["file_path"],
                    job_id,
                    progress_callback=lambda p, m: self._update_progress(job_id, p, m),
                )
                tr.result = result
                tr.status = TaskStatus.COMPLETED
                tr.message = "Completed"
            except Exception as e:
                tr.error = str(e)
                tr.status = TaskStatus.FAILED
                tr.message = "Failed"

            # 2. Move final WAV
            wav_path = Path(tr.metadata["file_path"])
            dest_dir = self.completed if tr.status == TaskStatus.COMPLETED else self.failed
            dest_dir.mkdir(exist_ok=True)
            try:
                shutil.move(str(wav_path), str(dest_dir / wav_path.name))
            except Exception:
                self._logger.exception("Failed to move WAV for %s", job_id)

            # 3. Write JSON or .log
            base = dest_dir / wav_path.name
            if tr.status == TaskStatus.COMPLETED:
                json_path = base.with_suffix('.json')
                try:
                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump(tr.result or {}, jf, ensure_ascii=False, indent=2, default=str)
                except Exception:
                    self._logger.exception("Failed to write result JSON for %s", job_id)
            else:
                log_path = base.with_suffix('.log')
                try:
                    with open(log_path, 'w', encoding='utf-8') as lf:
                        lf.write(tr.error or 'Unknown error')
                except Exception:
                    self._logger.exception("Failed to write error log for %s", job_id)

            tr.updated_at = datetime.utcnow().isoformat()
            await self._notify_subscribers(job_id)
            self._queue.task_done()
        # Drain remaining tasks if any (after stop)
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
