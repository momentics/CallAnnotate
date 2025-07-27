# src/app/queue_manager.py

# -*- coding: utf-8 -*-
"""
Менеджер очереди задач для CallAnnotate

Автор: akoodoy@capitol.ruСсылка: https://github.com/momentics/CallAnnotate
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
from typing import Callable, Dict, List, Optional, Any

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
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.progress: int = 0
        self.metadata: Optional[Dict[str, Any]] = metadata


class QueueManager:
    def __init__(
        self,
        config: Dict[str, Any] = None,
        *,
        volume_path: str = None,
        logs_base: str = None,
        max_queue_size: int = None,
        max_concurrent_tasks: int = None,
        task_timeout: int = None,
        cleanup_interval: int = None,
    ):
        """
        Инициализация менеджера очереди.

        Args:
            config: словарь конфигурации раздела 'queue' из default.yaml.
            volume_path: путь к корневой папке volume, переопределяет config['volume_path'].
            logs_base: базовый путь для логов (вне volume), переопределяет config['logs_base'].
            max_queue_size: максимальный размер очереди.
            max_concurrent_tasks: максимальное число параллельных задач.
            task_timeout: таймаут выполнения задачи в секундах.
            cleanup_interval: интервал фоновой очистки в секундах.
        """
        # Загрузка конфигурации
        if config is None:
            base_dir = Path(__file__).resolve().parent.parent
            cfg_file = base_dir.parent / "config" / "default.yaml"
            with open(cfg_file, 'r', encoding='utf-8') as f:
                full_conf = yaml.safe_load(f)
            config = full_conf.get('queue', {})

        # Переопределение параметров из аргументов
        if volume_path is not None:
            config['volume_path'] = volume_path
        if logs_base is not None:
            config['logs_base'] = logs_base
        if max_queue_size is not None:
            config['max_queue_size'] = max_queue_size
        if max_concurrent_tasks is not None:
            config['max_concurrent_tasks'] = max_concurrent_tasks
        if task_timeout is not None:
            config['task_timeout'] = task_timeout
        if cleanup_interval is not None:
            config['cleanup_interval'] = cleanup_interval

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Параметры очереди
        self.max_concurrent_tasks: int = config.get('max_concurrent_tasks', 1)
        self.max_queue_size: int = config.get('max_queue_size', 100)
        self.task_timeout: int = config.get('task_timeout', 3600)
        self.cleanup_interval: int = config.get('cleanup_interval', 300)

        # Пути согласно README.md
        self.base_volume: Path = Path(config.get('volume_path', '/volume'))
        self.queue_path: Path = self.base_volume  # очередь – корень volume
        self.incoming_path: Path = self.base_volume / 'incoming'
        self.processing_path: Path = self.base_volume / 'processing'
        self.completed_path: Path = self.base_volume / 'completed'
        self.failed_path: Path = self.base_volume / 'failed'
        self.archived_path: Path = self.base_volume / 'archived'
        self.logs_volume: Path = self.base_volume / 'logs'
        self.logs_system: Path = self.logs_volume / 'system'
        self.models_path: Path = self.base_volume / 'models' / 'embeddings'
        self.temp_path: Path = self.base_volume / 'temp'

        # Дополнительные логи вне volume
        self.logs_base: Path = Path(config.get('logs_base', '/var/log/callannotate'))

        # Создание директорий
        self._create_directories()

        # Внутренние структуры
        self.tasks: Dict[str, TaskResult] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.task_subscribers: Dict[str, List[str]] = {}

        # Воркеры и управление
        self.workers: List[asyncio.Task] = []
        self.is_running: bool = False
        self.shutdown_event: asyncio.Event = asyncio.Event()

        # Пул для CPU-задач
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        self.lock: threading.RLock = threading.RLock()

    def _create_directories(self):
        """Создание структуры тома /volume и логов"""
        dirs = [
            # Основные папки очереди
            self.incoming_path,
            self.processing_path,
            self.completed_path,
            self.failed_path,
            self.archived_path,
            # Логи внутри volume
            self.logs_system,
            # Модели эмбеддингов
            self.models_path,
            # Временные файлы
            self.temp_path,
            # Дополнительные логи вне volume
            self.logs_base / 'tasks',
            self.logs_base / 'system',
            self.logs_base / 'queue',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    async def get_queue_info(self):
        """Получение информации о состоянии очереди"""
        with self.lock:
            queue_length = self.task_queue.qsize()
            processing = list(self.processing_tasks.keys())
            queued = [tid for tid, t in self.tasks.items() if t.status == TaskStatus.QUEUED]
        # среднее время обработки
        times: List[float] = []
        with self.lock:
            for t in self.tasks.values():
                if t.status == TaskStatus.COMPLETED and t.started_at and t.completed_at:
                    start = datetime.fromisoformat(t.started_at)
                    end = datetime.fromisoformat(t.completed_at)
                    times.append((end - start).total_seconds())
        avg_time = sum(times) / len(times) if times else 0
        return {
            "queue_length": queue_length,
            "processing_jobs": processing,
            "queued_jobs": queued,
            "average_processing_time": avg_time,
        }

    async def start(self):
        """Запуск менеджера очереди"""
        if self.is_running:
            return
        self.is_running = True
        self.shutdown_event.clear()
        await self._restore_tasks()
        # Один файл за раз
        worker = asyncio.create_task(self._worker("worker-0"))
        self.workers.append(worker)
        cleanup = asyncio.create_task(self._cleanup_task())
        self.workers.append(cleanup)
        self.logger.info("QueueManager запущен")

    async def stop(self):
        """Остановка менеджера очереди"""
        if not self.is_running:
            return
        self.logger.info("Остановка QueueManager...")
        self.is_running = False
        self.shutdown_event.set()
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        self.executor.shutdown(wait=True)
        await self._save_state()
        self.logger.info("QueueManager остановлен")

    async def add_task(self, task_id: str, metadata: Dict[str, Any]) -> bool:
        """Добавление задачи в очередь"""
        try:
            with self.lock:
                if task_id in self.tasks:
                    return False
                tr = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.QUEUED,
                    message="Задача в очереди",
                    created_at=datetime.now().isoformat(),
                    metadata=metadata,
                )
                self.tasks[task_id] = tr
            await self.task_queue.put(task_id)
            await self._save_task_metadata(task_id, metadata)
            self.logger.info(f"Задача {task_id} добавлена")
            return True
        except Exception as e:
            self.logger.error(f"add_task error {e}")
            return False

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Получение результата задачи"""
        with self.lock:
            return self.tasks.get(task_id)

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, TaskResult]:
        """Список задач с фильтрацией"""
        with self.lock:
            items = list(self.tasks.items())
        if status:
            items = [(tid, res) for tid, res in items if res.status == status]
        items.sort(key=lambda x: x[1].created_at, reverse=True)
        selected = items[offset:offset + limit]
        return {tid: res for tid, res in selected}

    async def cancel_task(self, task_id: str) -> bool:
        """Отмена задачи"""
        removed = False
        with self.lock:
            if task_id in self.tasks and self.tasks[task_id].status in {
                TaskStatus.QUEUED, TaskStatus.PROCESSING
            }:
                t = self.tasks[task_id]
                t.status = TaskStatus.CANCELLED
                t.message = "Задача отменена"
                t.updated_at = datetime.now().isoformat()
                removed = True
        if task_id in self.processing_tasks:
            self.processing_tasks[task_id].cancel()
        self.logger.info(f"cancel_task {task_id}: {removed}")
        return removed

    async def get_queue_size(self) -> int:
        """Размер очереди"""
        return self.task_queue.qsize()

    async def get_active_tasks_count(self) -> int:
        """Количество активных задач"""
        return len(self.processing_tasks)

    async def subscribe_to_task(self, task_id: str, client_id: str):
        """Подписка на уведомления"""
        if task_id not in self.task_subscribers:
            self.task_subscribers[task_id] = []
        if client_id not in self.task_subscribers[task_id]:
            self.task_subscribers[task_id].append(client_id)

    async def _worker(self, worker_name: str):
        """Воркер: один файл за раз"""
        self.logger.info(f"Worker {worker_name} старт")
        while self.is_running:
            try:
                try:
                    tid = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                task = asyncio.create_task(self._process_task(tid))
                self.processing_tasks[tid] = task
                try:
                    await task
                except asyncio.CancelledError:
                    self.logger.info(f"Task {tid} cancelled")
                finally:
                    self.processing_tasks.pop(tid, None)
                    self.task_queue.task_done()
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
        self.logger.info(f"Worker {worker_name} стоп")

    async def _process_task(self, task_id: str):
        """Обработка задачи"""
        log_file = self.logs_base / 'tasks' / f"{task_id}.log"
        tlogger = logging.getLogger(f"task.{task_id}")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        tlogger.addHandler(handler)
        tlogger.setLevel(logging.INFO)
        try:
            with self.lock:
                tr = self.tasks.get(task_id)
            if not tr or tr.status == TaskStatus.CANCELLED:
                return
            tr.status = TaskStatus.PROCESSING
            tr.started_at = datetime.now().isoformat()
            tr.updated_at = tr.started_at
            tlogger.info(f"Start processing {task_id}")

            # Работа с файлами согласно структуре
            src = self.incoming_path / Path(tr.metadata.get('file_path', '')).name
            if not src.exists():
                raise FileNotFoundError(f"{src} not found")
            dst = self.processing_path / src.name
            src.rename(dst)

            # Вызов AnnotationService
            annotation = AnnotationService(self.config)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_annotation_sync,
                annotation,
                str(dst),
                task_id,
                self._progress_cb(task_id)
            )

            # Запись результата
            out = self.completed_path / f"{task_id}.json"
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Перемещение файла в completed
            dst.rename(self.completed_path / dst.name)

            with self.lock:
                tr.status = TaskStatus.COMPLETED
                tr.completed_at = datetime.now().isoformat()
                tr.updated_at = tr.completed_at
                tr.result = result
                tr.progress = 100
            tlogger.info(f"Task {task_id} completed")
        except asyncio.CancelledError:
            with self.lock:
                if task_id in self.tasks:
                    t = self.tasks[task_id]
                    t.status = TaskStatus.CANCELLED
                    t.updated_at = datetime.now().isoformat()
            tlogger.info(f"Task {task_id} cancelled")
        except Exception as e:
            err = str(e)
            with self.lock:
                t = self.tasks.get(task_id)
                if t:
                    t.status = TaskStatus.FAILED
                    t.error = err
                    t.updated_at = datetime.now().isoformat()
            tlogger.error(f"Error in task {task_id}: {e}")
            # Перемещение в failed
            for f in self.processing_path.glob(dst.name):
                f.rename(self.failed_path / f.name)
        finally:
            tlogger.removeHandler(handler)
            handler.close()

    def _run_annotation_sync(
        self,
        annotation_service: AnnotationService,
        file_path: str,
        task_id: str,
        progress_cb: Callable,
    ) -> Dict[str, Any]:
        """Синхронный вызов AnnotationService"""
        import asyncio as _asyncio
        loop = _asyncio.new_event_loop()
        _asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                annotation_service.process_audio(file_path, task_id, progress_cb)
            )
        finally:
            loop.close()

    def _progress_cb(self, task_id: str) -> Callable:
        async def cb(progress: int, message: str):
            with self.lock:
                if task_id in self.tasks:
                    t = self.tasks[task_id]
                    t.progress = progress
                    t.message = message
                    t.updated_at = datetime.now().isoformat()
        return cb

    async def _cleanup_task(self):
        """Фоновая очистка старых задач и файлов"""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                threshold = datetime.now() - timedelta(days=7)
                to_archive: List[str] = []
                with self.lock:
                    for tid, tr in self.tasks.items():
                        created = datetime.fromisoformat(tr.created_at)
                        if tr.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED} and created < threshold:
                            to_archive.append(tid)
                for tid in to_archive:
                    await self._archive_task(tid)
                if to_archive:
                    self.logger.info(f"Archived {len(to_archive)} tasks")
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    async def _archive_task(self, task_id: str):
        """Архивирование задачи"""
        try:
            for name in self.completed_path.glob(f"{task_id}*"):
                name.rename(self.archived_path / name.name)
            for name in self.failed_path.glob(f"{task_id}*"):
                name.rename(self.archived_path / name.name)
            # Удаляем из памяти
            with self.lock:
                self.tasks.pop(task_id, None)
        except Exception as e:
            self.logger.error(f"Archive error {e}")

    async def _save_task_metadata(self, task_id: str, metadata: Dict[str, Any]):
        """Сохранение метаданных задачи"""
        meta = {**metadata, 'task_id': task_id, 'created_at': datetime.now().isoformat()}
        path = self.logs_base / 'queue' / f"{task_id}_meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    async def _restore_tasks(self):
        """Восстановление задач после перезапуска"""
        try:
            for f in self.processing_path.glob('*'):
                if f.is_file():
                    f.rename(self.incoming_path / f.name)
            for mf in (self.logs_base / 'queue').glob('*_meta.json'):
                md = json.loads(mf.read_text(encoding='utf-8'))
                tid = mf.stem.replace('_meta', '')
                tr = TaskResult(
                    task_id=tid,
                    status=TaskStatus.QUEUED,
                    message="Восстановлена после перезапуска",
                    created_at=md.get('created_at', datetime.now().isoformat()),
                    metadata=md,
                )
                with self.lock:
                    self.tasks[tid] = tr
                await self.task_queue.put(tid)
        except Exception as e:
            self.logger.error(f"Restore error: {e}")

    async def _save_state(self):
        """Сохранение состояния менеджера очереди"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'tasks': {}
            }
            with self.lock:
                for tid, t in self.tasks.items():
                    state['tasks'][tid] = {
                        'task_id': t.task_id,
                        'status': t.status.value,
                        'message': t.message,
                        'created_at': t.created_at,
                        'updated_at': t.updated_at,
                        'started_at': t.started_at,
                        'completed_at': t.completed_at,
                        'progress': t.progress,
                        'error': t.error,
                        'metadata': t.metadata,
                    }
            sf = self.logs_base / 'queue' / 'manager_state.json'
            sf.parent.mkdir(parents=True, exist_ok=True)
            with open(sf, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Save state error: {e}")
