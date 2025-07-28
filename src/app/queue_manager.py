# src/app/queue_manager.py

# -*- coding: utf-8 -*-
"""
Менеджер очереди задач для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import asyncio
import json
import logging
import threading
import yaml

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

from .annotation import AnnotationService


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskResult:
    """Результат выполнения задачи"""

    def __init__(
        self,
        task_id: str,
        status: TaskStatus,
        message: str,
        created_at: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.status = status
        self.message = message
        self.created_at = created_at
        self.updated_at: Optional[str] = None
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None
        self.progress: int = 0
        self.metadata: Optional[Dict[str, Any]] = metadata
        self.result: Optional[Any] = None


class QueueManager:
    def __init__(
        self,
        config: Dict[str, Any] = None,
        *,
        volume_path: str = None,
    ):
        """
        Инициализация менеджера очереди.

        Args:
            config: полная конфигурация приложения из default.yaml.
            volume_path: путь к корневой папке volume (переопределяет конфиг).
        """
        # Загрузка конфигурации
        if config is None:
            base_dir = Path(__file__).resolve().parent.parent
            cfg_file = base_dir.parent / "config" / "default.yaml"
            with open(cfg_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

        # Секция queue
        queue_cfg = config.get('queue', {})
        if volume_path:
            queue_cfg = {**queue_cfg, 'volume_path': volume_path}
        self.cfg = queue_cfg
        self.full_cfg = config

        self.logger = logging.getLogger(__name__)

        # Параметры
        self.max_concurrent = queue_cfg.get('max_concurrent_tasks', 1)
        self.max_queue = queue_cfg.get('max_queue_size', 100)
        self.timeout = queue_cfg.get('task_timeout', 3600)
        self.cleanup_interval = queue_cfg.get('cleanup_interval', 300)

        # Пути (единственный корень)
        self.base = Path(queue_cfg.get('volume_path', './volume')).resolve()
        self.incoming = self.base / 'incoming'
        self.processing = self.base / 'processing'
        self.completed = self.base / 'completed'
        self.failed = self.base / 'failed'
        self.archived = self.base / 'archived'
        self.logs = self.base / 'logs'
        self._mkdirs([
            self.incoming, self.processing, self.completed,
            self.failed, self.archived, self.logs
        ])

        self.tasks: Dict[str, TaskResult] = {}
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue)
        self.proc_tasks: Dict[str, asyncio.Task] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.shutdown = asyncio.Event()

    def _mkdirs(self, dirs: List[Path]):
        for d in dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Не удалось создать {d}: {e}")

    async def start(self):
        if self.running:
            return
        self.running = True
        await self._restore()
        w = asyncio.create_task(self._worker())
        c = asyncio.create_task(self._cleanup())
        self.workers += [w, c]
        self.logger.info("QueueManager запущен")

    async def stop(self):
        if not self.running:
            return
        self.running = False
        self.shutdown.set()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.executor.shutdown(wait=True)
        self.logger.info("QueueManager остановлен")

    async def add_task(self, task_id: str, metadata: Dict[str, Any]) -> bool:
        with self.lock:
            if task_id in self.tasks:
                return False
            tr = TaskResult(task_id, TaskStatus.QUEUED, "В очереди", datetime.now().isoformat(), metadata)
            self.tasks[task_id] = tr
        await self.queue.put(task_id)
        return True

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        with self.lock:
            return self.tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        with self.lock:
            t = self.tasks.get(task_id)
            if not t or t.status not in {TaskStatus.QUEUED, TaskStatus.PROCESSING}:
                return False
            t.status = TaskStatus.CANCELLED
        ct = self.proc_tasks.get(task_id)
        if ct:
            ct.cancel()
        return True

    async def _worker(self):
        while self.running:
            try:
                tid = await asyncio.wait_for(self.queue.get(), timeout=1)
            except asyncio.TimeoutError:
                continue
            task = asyncio.create_task(self._process(tid))
            self.proc_tasks[tid] = task
            try:
                await task
            finally:
                self.proc_tasks.pop(tid, None)
                self.queue.task_done()

    async def _process(self, task_id: str):
        dst = None
        try:
            with self.lock:
                tr = self.tasks[task_id]
                tr.status = TaskStatus.PROCESSING
                tr.started_at = datetime.now().isoformat()
            tlogger = logging.getLogger(f"task.{task_id}")
            log_file = self.logs / f"{task_id}.log"
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
            tlogger.addHandler(fh)
            tlogger.setLevel(logging.INFO)

            src = Path(tr.metadata['file_path'])
            if not src.exists():
                raise FileNotFoundError(f"{src} not found")
            dst = self.processing / src.name
            src.rename(dst)

            svc = AnnotationService(self.full_cfg)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: svc.run_sync(str(dst), task_id)
            )

            task_out = self.completed / task_id
            task_out.mkdir(parents=True, exist_ok=True)
            (task_out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            (task_out / dst.name).rename(task_out / dst.name)

            with self.lock:
                tr.status = TaskStatus.COMPLETED
                tr.completed_at = datetime.now().isoformat()
                tr.result = result
                tr.progress = 100
            tlogger.info("Completed")
        except asyncio.CancelledError:
            with self.lock:
                tr = self.tasks.get(task_id)
                if tr:
                    tr.status = TaskStatus.CANCELLED
            tlogger.info("Cancelled")
        except Exception as e:
            with self.lock:
                tr = self.tasks.get(task_id)
                if tr:
                    tr.status = TaskStatus.FAILED
                    tr.error = str(e)
            logging.error(f"Error {e}")
            if dst and dst.exists():
                try:
                    dst.rename(self.failed / dst.name)
                except: pass
        finally:
            try:
                tlogger.removeHandler(fh)
                fh.close()
            except: pass

    async def _cleanup(self):
        while self.running:
            await asyncio.sleep(self.cleanup_interval)
            cutoff = datetime.now() - timedelta(days=7)
            to_archive = []
            with self.lock:
                for tid, tr in self.tasks.items():
                    created = datetime.fromisoformat(tr.created_at)
                    if tr.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED} and created < cutoff:
                        to_archive.append(tid)
            for tid in to_archive:
                src = self.completed / tid
                if src.exists():
                    (self.archived / tid).mkdir(parents=True, exist_ok=True)
                    src.rename(self.archived / tid)
                with self.lock:
                    self.tasks.pop(tid, None)

    async def _restore(self):
        for f in self.processing.iterdir():
            if f.is_file():
                try:
                    f.rename(self.incoming / f.name)
                except: pass
        # восстановление из логов в будущем
