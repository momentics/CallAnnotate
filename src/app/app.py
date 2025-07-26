# -*- coding: utf-8 -*-
"""
FastAPI приложение CallAnnotate с WebSocket и REST/JSON API

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional, Any

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .queue_manager import QueueManager, TaskStatus
from .utils import setup_logging, validate_audio_file, create_task_metadata
from .annotation import AnnotationService


class AudioRequest(BaseModel):
    """Модель запроса для обработки аудио"""
    file_id: str = Field(..., description="Уникальный идентификатор файла")
    priority: int = Field(default=5, ge=1, le=10, description="Приоритет задачи (1-10)")
    callback_url: Optional[str] = Field(None, description="URL для callback уведомлений")
    options: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные опции")


class TaskResponse(BaseModel):
    """Модель ответа с информацией о задаче"""
    task_id: str
    status: TaskStatus
    message: str
    created_at: str
    updated_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WebSocketManager:
    """Менеджер WebSocket соединений"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Подключить WebSocket клиента"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.logger.info(f"WebSocket клиент {client_id} подключен")
    
    def disconnect(self, client_id: str):
        """Отключить WebSocket клиента"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            self.logger.info(f"WebSocket клиент {client_id} отключен")
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Отправить персональное сообщение клиенту"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Ошибка отправки сообщения клиенту {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Отправить сообщение всем подключенным клиентам"""
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Ошибка отправки broadcast сообщения клиенту {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Очистить отключенных клиентов
        for client_id in disconnected_clients:
            self.disconnect(client_id)


def create_app(config_path: str = None) -> FastAPI:
    """Фабрика создания FastAPI приложения"""

    # Определяем путь к default.yaml, если не передан
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent  # src/app/.. -> src/
        config_path = base_dir.parent / "config" / "default.yaml"

    # Загрузка конфигурации
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Настройка логирования
    setup_logging(config.get('logging', {}))
    logger = logging.getLogger(__name__)
    
    # Создание FastAPI приложения
    app = FastAPI(
        title="CallAnnotate API",
        description="API для автоматической аннотации телефонных разговоров",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get('cors', {}).get('origins', ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Инициализация компонентов
    queue_manager = QueueManager(config.get('queue', {}))
    websocket_manager = WebSocketManager()
    annotation_service = AnnotationService(config)
    
    # Сохранение конфигурации в app state
    app.state.config = config
    app.state.queue_manager = queue_manager
    app.state.websocket_manager = websocket_manager
    app.state.annotation_service = annotation_service
    
    @app.on_event("startup")
    async def startup_event():
        """Инициализация при запуске приложения"""
        logger.info("Запуск CallAnnotate API сервера")
        await queue_manager.start()
        logger.info("Менеджер очереди запущен")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Очистка при остановке приложения"""
        logger.info("Остановка CallAnnotate API сервера")
        await queue_manager.stop()
        logger.info("Менеджер очереди остановлен")
    
    # REST API Endpoints
    
    @app.get("/health")
    async def health_check():
        """Проверка работоспособности сервиса"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "queue_size": await queue_manager.get_queue_size(),
            "active_tasks": await queue_manager.get_active_tasks_count()
        }
    
    @app.post("/api/annotate", response_model=TaskResponse)
    async def annotate_audio(
        file: UploadFile = File(...),
        priority: int = 5,
        callback_url: Optional[str] = None
    ):
        """Загрузка аудиофайла для аннотации"""
        try:
            # Валидация файла
            validation_result = validate_audio_file(file)
            if not validation_result.is_valid:
                raise HTTPException(status_code=400, detail=validation_result.error)
            
            # Создание уникального ID задачи
            task_id = str(uuid.uuid4())
            
            # Сохранение файла во временную директорию
            upload_dir = Path("/volume/queue/incoming")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = upload_dir / f"{task_id}_{file.filename}"
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Создание метаданных задачи
            metadata = create_task_metadata(
                task_id=task_id,
                file_path=str(file_path),
                filename=file.filename,
                priority=priority,
                callback_url=callback_url
            )
            
            # Добавление задачи в очередь
            await queue_manager.add_task(task_id, metadata)
            
            logger.info(f"Задача {task_id} добавлена в очередь")
            
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.QUEUED,
                message="Задача добавлена в очередь обработки",
                created_at=metadata["created_at"]
            )
            
        except Exception as e:
            logger.error(f"Ошибка при создании задачи аннотации: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/tasks/{task_id}", response_model=TaskResponse)
    async def get_task_status(task_id: str):
        """Получение статуса задачи по ID"""
        try:
            task_result = await queue_manager.get_task_result(task_id)
            if not task_result:
                raise HTTPException(status_code=404, detail="Задача не найдена")
            
            return TaskResponse(
                task_id=task_id,
                status=task_result.status,
                message=task_result.message,
                created_at=task_result.created_at,
                updated_at=task_result.updated_at,
                result=task_result.result,
                error=task_result.error
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при получении статуса задачи {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/tasks")
    async def list_tasks(
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ):
        """Получение списка задач с фильтрацией"""
        try:
            tasks = await queue_manager.list_tasks(status=status, limit=limit, offset=offset)
            return {
                "tasks": [
                    TaskResponse(
                        task_id=task_id,
                        status=result.status,
                        message=result.message,
                        created_at=result.created_at,
                        updated_at=result.updated_at,
                        result=result.result,
                        error=result.error
                    )
                    for task_id, result in tasks.items()
                ],
                "total": len(tasks),
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error(f"Ошибка при получении списка задач: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/api/tasks/{task_id}")
    async def cancel_task(task_id: str):
        """Отмена задачи"""
        try:
            success = await queue_manager.cancel_task(task_id)
            if not success:
                raise HTTPException(status_code=404, detail="Задача не найдена или уже завершена")
            
            return {"message": f"Задача {task_id} отменена"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при отмене задачи {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/results/{task_id}")
    async def download_result(task_id: str):
        """Скачивание результата аннотации"""
        try:
            result_path = Path(f"/volume/outputs/pending/{task_id}.json")
            if not result_path.exists():
                raise HTTPException(status_code=404, detail="Результат не найден")
            
            return FileResponse(
                path=str(result_path),
                filename=f"annotation_{task_id}.json",
                media_type="application/json"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Ошибка при скачивании результата {task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # WebSocket Endpoints
    
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        """WebSocket endpoint для real-time коммуникации"""
        await websocket_manager.connect(websocket, client_id)
        
        try:
            while True:
                # Получение сообщения от клиента
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Обработка различных типов сообщений
                if message.get("type") == "ping":
                    await websocket_manager.send_personal_message(
                        {"type": "pong", "timestamp": message.get("timestamp")},
                        client_id
                    )
                
                elif message.get("type") == "subscribe_task":
                    task_id = message.get("task_id")
                    if task_id:
                        # Подписка на обновления задачи
                        await queue_manager.subscribe_to_task(task_id, client_id)
                        await websocket_manager.send_personal_message(
                            {
                                "type": "subscribed",
                                "task_id": task_id,
                                "message": f"Подписка на задачу {task_id} активна"
                            },
                            client_id
                        )
                
                elif message.get("type") == "upload_audio":
                    # Обработка загрузки аудио через WebSocket
                    audio_data = message.get("data")
                    filename = message.get("filename", "audio.wav")
                    
                    task_id = str(uuid.uuid4())
                    
                    # Сохранение аудио данных
                    upload_dir = Path("/volume/queue/incoming")
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    file_path = upload_dir / f"{task_id}_{filename}"
                    
                    # Декодирование base64 данных
                    import base64
                    audio_bytes = base64.b64decode(audio_data)
                    with open(file_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Создание задачи
                    metadata = create_task_metadata(
                        task_id=task_id,
                        file_path=str(file_path),
                        filename=filename,
                        priority=message.get("priority", 5),
                        websocket_client_id=client_id
                    )
                    
                    await queue_manager.add_task(task_id, metadata)
                    
                    await websocket_manager.send_personal_message(
                        {
                            "type": "task_created",
                            "task_id": task_id,
                            "status": "queued",
                            "message": "Задача создана и добавлена в очередь"
                        },
                        client_id
                    )
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(client_id)
            logger.info(f"WebSocket клиент {client_id} отключился")
        except Exception as e:
            logger.error(f"Ошибка WebSocket для клиента {client_id}: {e}")
            websocket_manager.disconnect(client_id)
    
    return app


# Создание экземпляра приложения
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
