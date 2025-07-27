# src/app/queue_manager.py
# -*- coding: utf-8 -*-
"""
Менеджер очереди задач для CallAnnotate

Автор: akoodoy@capitol.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import asyncio
from dataclasses import asdict
import json
import logging
import threading
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
        max_queue_size: int = None,
        max_concurrent_tasks: int = None,
        task_timeout: int = None,
        cleanup_interval: int = None
    ):
        """
        Инициализация менеджера очереди.

        Args:
            config: словарь конфигурации раздела 'queue' из default.yaml.
            volume_path: путь к корневой папке volume, переопределяет config['volume_path'].
            max_queue_size: максимальный размер очереди (переопределяет config).
            max_concurrent_tasks: максимальное число параллельных задач (переопределяет config).
            task_timeout: таймаут выполнения задачи в секундах (переопределяет config).
            cleanup_interval: интервал фоновой очистки в секундах (переопределяет config).
        """
        import yaml

        # Загрузка конфигурации из YAML, если не передан явный словарь
        if config is None:
            base_dir = Path(__file__).resolve().parent.parent
            cfg_file = base_dir.parent / "config" / "default.yaml"
            with open(cfg_file, 'r', encoding='utf-8') as f:
                full_conf = yaml.safe_load(f)
            config = full_conf.get('queue', {})

        # Переопределение конфигурационных параметров, если заданы
        if volume_path is not None:
            config['volume_path'] = volume_path
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
        self.max_concurrent_tasks: int = config.get('max_concurrent_tasks', 2)
        self.max_queue_size: int = config.get('max_queue_size', 100)
        self.task_timeout: int = config.get('task_timeout', 3600)
        self.cleanup_interval: int = config.get('cleanup_interval', 300)

        # Пути для работы с файловой системой
        self.volume_path: Path = Path(config.get('volume_path', '/app/volume'))
        self.queue_path: Path = self.volume_path / 'queue'
        self.logs_path: Path = self.volume_path / 'logs'

        # Создание необходимых директорий
        self._create_directories()

        # Внутренние структуры данных
        self.tasks: Dict[str, TaskResult] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.task_subscribers: Dict[str, List[str]] = {}

        # Воркеры и управление
        self.workers: List[asyncio.Task] = []
        self.is_running: bool = False
        self.shutdown_event: asyncio.Event = asyncio.Event()

        # Thread pool для CPU-интенсивных задач
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)

        # Блокировка для thread-safe операций
        self.lock: threading.RLock = threading.RLock()

    async def get_queue_info(self):
        # Возвращает информацию о длине очереди и т.д.
        return {
            "queue_length": await self.get_queue_size(),
            "processing_jobs": [],
            "queued_jobs": [],
            "average_processing_time": 0
        }

    def _create_directories(self):
        """Создание необходимых директорий"""
        directories = [
            self.queue_path / 'incoming',
            self.queue_path / 'processing',
            self.queue_path / 'completed',
            self.queue_path / 'failed',
            self.queue_path / 'archived',
            self.volume_path / 'intermediate' / 'diarization',
            self.volume_path / 'intermediate' / 'transcription',
            self.volume_path / 'intermediate' / 'recognition',
            self.volume_path / 'intermediate' / 'carddav',
            self.volume_path / 'outputs' / 'pending',
            self.volume_path / 'outputs' / 'delivered',
            self.logs_path / 'tasks',
            self.logs_path / 'system',
            self.logs_path / 'queue'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Запуск менеджера очереди"""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Восстановление состояния после перезапуска
        await self._restore_tasks()
        
        # Запуск воркеров
        for i in range(self.max_concurrent_tasks):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Запуск фонового процесса очистки
        cleanup_task = asyncio.create_task(self._cleanup_task())
        self.workers.append(cleanup_task)
        
        self.logger.info(f"QueueManager запущен с {self.max_concurrent_tasks} воркерами")
    
    async def stop(self):
        """Остановка менеджера очереди"""
        if not self.is_running:
            return
        
        self.logger.info("Остановка QueueManager...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Ожидание завершения всех воркеров
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Остановка executor
        self.executor.shutdown(wait=True)
        
        # Сохранение состояния
        await self._save_state()
        
        self.logger.info("QueueManager остановлен")
    
    async def add_task(self, task_id: str, metadata: Dict[str, Any]) -> bool:
        """Добавление задачи в очередь"""
        try:
            with self.lock:
                if task_id in self.tasks:
                    return False
                
                # Создание TaskResult
                task_result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.QUEUED,
                    message="Задача в очереди на обработку",
                    created_at=datetime.now().isoformat(),
                    metadata=metadata
                )
                
                self.tasks[task_id] = task_result
            
            # Добавление в очередь
            await self.task_queue.put(task_id)
            
            # Сохранение метаданных в файл
            await self._save_task_metadata(task_id, metadata)
            
            self.logger.info(f"Задача {task_id} добавлена в очередь")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении задачи {task_id}: {e}")
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
        """Получение списка задач с фильтрацией"""
        with self.lock:
            filtered_tasks = {}
            
            # Фильтрация по статусу
            tasks_to_process = self.tasks.items()
            if status:
                tasks_to_process = [
                    (tid, result) for tid, result in tasks_to_process
                    if result.status == status
                ]
            
            # Сортировка по времени создания (новые сначала)
            tasks_to_process = sorted(
                tasks_to_process,
                key=lambda x: x[1].created_at,
                reverse=True
            )
            
            # Применение offset и limit
            for task_id, result in tasks_to_process[offset:offset + limit]:
                filtered_tasks[task_id] = result
            
            return filtered_tasks
    
    async def cancel_task(self, job_id: str):
        removed = False
        # Пример: удаляем задачу из всех возможных хранилищ
        if hasattr(self, '_active_tasks') and job_id in self._active_tasks:
            del self._active_tasks[job_id]
            removed = True
        if hasattr(self, '_queue') and job_id in self._queue:
            del self._queue[job_id]
            removed = True
        if hasattr(self, '_done_tasks') and job_id in self._done_tasks:
            del self._done_tasks[job_id]
            removed = True
        if hasattr(self, '_failed_tasks') and job_id in self._failed_tasks:
            del self._failed_tasks[job_id]
            removed = True
        # ... любые другие внутренние хранилища ...
        import logging
        if removed:
            logging.getLogger(__name__).info(f"Задача {job_id} отменена")
        else:
            logging.getLogger(__name__).info(f"Задача {job_id} не найдена для отмены")
        return removed
        
    async def get_queue_size(self) -> int:
        """Получение размера очереди"""
        return self.task_queue.qsize()
    
    async def get_active_tasks_count(self) -> int:
        """Получение количества активных задач"""
        return len(self.processing_tasks)
    
    async def subscribe_to_task(self, task_id: str, client_id: str):
        """Подписка клиента на обновления задачи"""
        if task_id not in self.task_subscribers:
            self.task_subscribers[task_id] = []
        
        if client_id not in self.task_subscribers[task_id]:
            self.task_subscribers[task_id].append(client_id)
    
    async def _worker(self, worker_name: str):
        """Воркер для обработки задач из очереди"""
        self.logger.info(f"Воркер {worker_name} запущен")
        
        while self.is_running:
            try:
                # Получение задачи из очереди с таймаутом
                try:
                    task_id = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Обработка задачи
                processing_task = asyncio.create_task(
                    self._process_task(task_id, worker_name)
                )
                self.processing_tasks[task_id] = processing_task
                
                await processing_task
                
                # Удаление из активных задач
                if task_id in self.processing_tasks:
                    del self.processing_tasks[task_id]
                
                # Отметка выполнения в очереди
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в воркере {worker_name}: {e}")
        
        self.logger.info(f"Воркер {worker_name} остановлен")
    
    async def _process_task(self, task_id: str, worker_name: str):
        """Обработка отдельной задачи"""
        task_log_path = self.logs_path / 'tasks' / f"{task_id}.log"
        
        # Настройка логирования для задачи
        task_logger = logging.getLogger(f"task.{task_id}")
        handler = logging.FileHandler(task_log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        task_logger.addHandler(handler)
        task_logger.setLevel(logging.INFO)
        
        try:
            with self.lock:
                if task_id not in self.tasks:
                    return
                
                task_result = self.tasks[task_id]
                
                # Проверка на отмену
                if task_result.status == TaskStatus.CANCELLED:
                    return
                
                # Обновление статуса
                task_result.status = TaskStatus.PROCESSING
                task_result.message = f"Обрабатывается воркером {worker_name}"
                task_result.started_at = datetime.now().isoformat()
                task_result.updated_at = task_result.started_at
            
            task_logger.info(f"Начата обработка задачи {task_id}")
            
            # Уведомление подписчиков
            await self._notify_subscribers(task_id, {
                "type": "task_update",
                "task_id": task_id,
                "status": TaskStatus.PROCESSING,
                "message": "Обработка начата"
            })
            
            # Получение метаданных
            metadata = task_result.metadata or {}
            file_path = metadata.get('file_path')
            
            if not file_path or not Path(file_path).exists():
                raise FileNotFoundError(f"Аудиофайл не найден: {file_path}")
            
            # Перемещение файла в processing
            processing_path = self.queue_path / 'processing' / Path(file_path).name
            Path(file_path).rename(processing_path)
            
            # Создание сервиса аннотации
            annotation_service = AnnotationService(self.config)
            
            # Обработка аудио с обновлением прогресса
            async def progress_callback(progress: int, message: str):
                with self.lock:
                    if task_id in self.tasks:
                        self.tasks[task_id].progress = progress
                        self.tasks[task_id].message = message
                        self.tasks[task_id].updated_at = datetime.now().isoformat()
                
                await self._notify_subscribers(task_id, {
                    "type": "task_progress",
                    "task_id": task_id,
                    "progress": progress,
                    "message": message
                })
            
            # Выполнение аннотации в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_annotation_sync,
                annotation_service,
                str(processing_path),
                task_id,
                progress_callback
            )
            
            # Сохранение результата
            output_path = self.volume_path / 'outputs' / 'pending' / f"{task_id}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # Перемещение файла в completed
            completed_path = self.queue_path / 'completed' / processing_path.name
            processing_path.rename(completed_path)
            
            # Обновление статуса
            with self.lock:
                if task_id in self.tasks:
                    task_result = self.tasks[task_id]
                    task_result.status = TaskStatus.COMPLETED
                    task_result.message = "Аннотация завершена успешно"
                    task_result.completed_at = datetime.now().isoformat()
                    task_result.updated_at = task_result.completed_at
                    task_result.result = result
                    task_result.progress = 100
            
            task_logger.info(f"Задача {task_id} завершена успешно")
            
            # Уведомление подписчиков
            await self._notify_subscribers(task_id, {
                "type": "task_completed",
                "task_id": task_id,
                "status": TaskStatus.COMPLETED,
                "message": "Аннотация завершена",
                "result": result
            })
            
        except asyncio.CancelledError:
            # Обработка отмены задачи
            with self.lock:
                if task_id in self.tasks:
                    self.tasks[task_id].status = TaskStatus.CANCELLED
                    self.tasks[task_id].message = "Задача отменена"
                    self.tasks[task_id].updated_at = datetime.now().isoformat()
            
            task_logger.info(f"Задача {task_id} отменена")
            
        except Exception as e:
            # Обработка ошибки
            error_msg = str(e)
            
            with self.lock:
                if task_id in self.tasks:
                    task_result = self.tasks[task_id]
                    task_result.status = TaskStatus.FAILED
                    task_result.message = f"Ошибка при обработке: {error_msg}"
                    task_result.error = error_msg
                    task_result.updated_at = datetime.now().isoformat()
            
            task_logger.error(f"Ошибка при обработке задачи {task_id}: {e}")
            
            # Перемещение файла в failed (если существует)
            processing_file = self.queue_path / 'processing' / f"{task_id}_*"
            for file_path in self.queue_path.glob(f'processing/{task_id}_*'):
                failed_path = self.queue_path / 'failed' / file_path.name
                file_path.rename(failed_path)
            
            # Уведомление подписчиков
            await self._notify_subscribers(task_id, {
                "type": "task_failed",
                "task_id": task_id,
                "status": TaskStatus.FAILED,
                "message": f"Ошибка: {error_msg}",
                "error": error_msg
            })
        
        finally:
            # Очистка ресурсов
            task_logger.removeHandler(handler)
            handler.close()
    
    def _run_annotation_sync(
        self,
        annotation_service: AnnotationService,
        file_path: str,
        task_id: str,
        progress_callback: Callable
    ) -> Dict[str, Any]:
        """Синхронный запуск аннотации"""
        # Создание нового event loop для этого потока
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(
                annotation_service.process_audio(file_path, task_id, progress_callback)
            )
        finally:
            loop.close()
    
    async def _notify_subscribers(self, task_id: str, message: Dict[str, Any]):
        """Уведомление подписчиков о изменениях задачи"""
        if task_id not in self.task_subscribers:
            return
        
        # Импорт здесь для избежания циркулярных зависимостей
        from .app import app
        
        websocket_manager = getattr(app.state, 'websocket_manager', None)
        if not websocket_manager:
            return
        
        # Отправка уведомлений всем подписчикам
        for client_id in self.task_subscribers[task_id]:
            await websocket_manager.send_personal_message(message, client_id)
    
    async def _cleanup_task(self):
        """Фоновая задача очистки старых файлов и задач"""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(days=7)  # Удаляем задачи старше 7 дней
                
                # Очистка завершенных задач
                tasks_to_remove = []
                with self.lock:
                    for task_id, result in self.tasks.items():
                        if result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                            task_created = datetime.fromisoformat(result.created_at)
                            if task_created < cleanup_threshold:
                                tasks_to_remove.append(task_id)
                
                # Архивирование старых задач
                for task_id in tasks_to_remove:
                    await self._archive_task(task_id)
                
                if tasks_to_remove:
                    self.logger.info(f"Архивировано {len(tasks_to_remove)} старых задач")
                
            except Exception as e:
                self.logger.error(f"Ошибка в задаче очистки: {e}")
    
    async def _archive_task(self, task_id: str):
        """Архивирование задачи"""
        try:
            # Перемещение файлов в архив
            for status_dir in ['completed', 'failed']:
                status_path = self.queue_path / status_dir
                for file_path in status_path.glob(f'{task_id}_*'):
                    archive_path = self.queue_path / 'archived' / file_path.name
                    file_path.rename(archive_path)
            
            # Перемещение результатов
            result_file = self.volume_path / 'outputs' / 'pending' / f'{task_id}.json'
            if result_file.exists():
                delivered_path = self.volume_path / 'outputs' / 'delivered' / result_file.name
                result_file.rename(delivered_path)
            
            # Удаление из памяти
            with self.lock:
                if task_id in self.tasks:
                    del self.tasks[task_id]
                
                if task_id in self.task_subscribers:
                    del self.task_subscribers[task_id]
            
        except Exception as e:
            self.logger.error(f"Ошибка при архивировании задачи {task_id}: {e}")
    
    async def _save_task_metadata(self, task_id: str, metadata: Dict[str, Any]):
        """Сохранение метаданных задачи в файл"""
        metadata_path = self.logs_path / 'queue' / f'{task_id}_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    async def _restore_tasks(self):
        """Восстановление задач после перезапуска"""
        try:
            # Восстановление из processing - перемещение обратно в incoming
            processing_path = self.queue_path / 'processing'
            for file_path in processing_path.glob('*'):
                if file_path.is_file():
                    incoming_path = self.queue_path / 'incoming' / file_path.name
                    file_path.rename(incoming_path)
                    self.logger.info(f"Файл {file_path.name} перемещен обратно в очередь")
            
            # Восстановление метаданных из логов
            queue_logs_path = self.logs_path / 'queue'
            for metadata_file in queue_logs_path.glob('*_metadata.json'):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    task_id = metadata_file.stem.replace('_metadata', '')
                    
                    # Проверка существования файла
                    file_path = metadata.get('file_path')
                    if file_path and Path(file_path).exists():
                        # Воссоздание TaskResult
                        task_result = TaskResult(
                            task_id=task_id,
                            status=TaskStatus.QUEUED,
                            message="Задача восстановлена после перезапуска",
                            created_at=metadata.get('created_at', datetime.now().isoformat()),
                            metadata=metadata
                        )
                        
                        with self.lock:
                            self.tasks[task_id] = task_result
                        
                        # Добавление в очередь
                        await self.task_queue.put(task_id)
                        
                        self.logger.info(f"Задача {task_id} восстановлена")
                
                except Exception as e:
                    self.logger.error(f"Ошибка при восстановлении задачи из {metadata_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Ошибка при восстановлении задач: {e}")
    
    async def _save_state(self):
        """Сохранение текущего состояния"""
        try:
            state_file = self.logs_path / 'queue' / 'manager_state.json'
            state_data = {
                'timestamp': datetime.now().isoformat(),
                'tasks': {
                    task_id: asdict(result)
                    for task_id, result in self.tasks.items()
                }
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info("Состояние менеджера очереди сохранено")
        
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении состояния: {e}")
