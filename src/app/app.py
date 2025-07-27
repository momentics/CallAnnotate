# -*- coding: utf8 -*-
"""
FastAPI приложение CallAnnotate с WebSocket и REST/JSON API

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml
from fastapi import (
    FastAPI,
    HTTPException,
    status,
    WebSocket,
    WebSocketDisconnect,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware

from .queue_manager import QueueManager, TaskStatus
from .utils import setup_logging, validate_audio_file_path, create_task_metadata
from .annotation import AnnotationService
from .schemas import CreateJobRequest, CreateJobResponse, JobStatusResponse


class WebSocketManager:
    """Менеджер WebSocket соединений"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.logger.info(f"WebSocket клиент {client_id} подключен")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            self.logger.info(f"WebSocket клиент {client_id} отключен")

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Ошибка отправки сообщения клиенту {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        disconnected = []
        for cid, conn in self.active_connections.items():
            try:
                await conn.send_text(json.dumps(message))
            except Exception:
                disconnected.append(cid)
        for cid in disconnected:
            self.disconnect(cid)


def create_app(config_path: str = None) -> FastAPI:
    """Фабрика создания FastAPI приложения"""
    if config_path is None:
        base = Path(__file__).resolve().parent.parent
        config_path = base.parent / "config" / "default.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)

    app = FastAPI(
        title="CallAnnotate API",
        description="API для автоматической аннотации телефонных разговоров",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors", {}).get("origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    queue_manager = QueueManager(
        None,
        volume_path=os.getenv("VOLUME_PATH"),
        max_queue_size=None,
        max_concurrent_tasks=None,
        task_timeout=None,
        cleanup_interval=None,
    )
    websocket_manager = WebSocketManager()
    annotation_service = AnnotationService(config)

    app.state.queue_manager = queue_manager
    app.state.websocket_manager = websocket_manager
    app.state.annotation_service = annotation_service

    @app.on_event("startup")
    async def startup_event():
        logger.info("Запуск CallAnnotate API сервера")
        await queue_manager.start()
        logger.info("Менеджер очереди запущен")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Остановка CallAnnotate API сервера")
        await queue_manager.stop()
        logger.info("Менеджер очереди остановлен")

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": config.get("server", {}).get("version", "1.0.0"),
            "uptime": int((datetime.now().timestamp() - app.state.start_time) if hasattr(app.state, 'start_time') else 0),
            "queue_length": await queue_manager.get_queue_size(),
            "active_tasks": await queue_manager.get_active_tasks_count(),
            "components": {
                "queue_manager": "healthy",
                "annotation_service": "healthy",
                "volume_access": "healthy" if Path(os.getenv("VOLUME_PATH", "/volume")).exists() else "error"
            }
        }

    @app.get("/info")
    async def info():
        volume_path = os.getenv("VOLUME_PATH", "/volume")
        return {
            "service": "CallAnnotate",
            "version": config.get("server", {}).get("version", "1.0.0"),
            "description": "Сервис автоматической аннотации телефонных разговоров",
            "max_file_size": int(
                os.getenv("MAX_FILE_SIZE", config.get("files", {}).get("max_size", 1073741824))
            ),
            "supported_formats": config.get("files", {}).get("allowed_formats", ["wav", "mp3", "flac"]),
            "processing_mode": "asynchronous",
            "volume_paths": {
                "incoming": f"{volume_path}/incoming",
                "processing": f"{volume_path}/processing",
                "completed": f"{volume_path}/completed",
                "failed": f"{volume_path}/failed"
            },
            "api_endpoints": {
                "rest": "/api/v1",
                "websocket": "/ws"
            }
        }

    @app.post("/api/v1/jobs", status_code=status.HTTP_201_CREATED, response_model=CreateJobResponse)
    async def create_job(request: CreateJobRequest):
        """Создание задачи аннотации по имени файла в volume"""
        
        # Проверка существования файла в volume/incoming
        volume_path = Path(os.getenv("VOLUME_PATH", "/volume"))
        incoming_path = volume_path / "incoming" / request.filename
        
        if not incoming_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{request.filename}' not found in /volume/incoming/"
            )
        
        # Валидация аудиофайла
        validation_result = validate_audio_file_path(str(incoming_path))
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=validation_result.error
            )
        
        # Проверка размера файла
        max_size = int(
            os.getenv("MAX_FILE_SIZE", config.get("files", {}).get("max_size", 524288000))
        )
        file_size = incoming_path.stat().st_size
        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size {file_size} exceeds maximum {max_size} bytes"
            )
        
        # Создание уникального ID задачи
        job_id = str(uuid.uuid4())
        
        # Создание метаданных задачи
        metadata = create_task_metadata(
            task_id=job_id,
            file_path=str(incoming_path),
            filename=request.filename,
            priority=request.priority
        )
        
        # Добавление задачи в очередь
        await queue_manager.add_task(job_id, metadata)
        
        return CreateJobResponse(
            job_id=job_id,
            status=TaskStatus.QUEUED,
            message="Job queued successfully",
            created_at=datetime.now(),
            file_info={
                "filename": request.filename,
                "path": str(incoming_path),
                "size_bytes": file_size
            },
            progress_url=f"/api/v1/jobs/{job_id}",
            result_url=f"/api/v1/jobs/{job_id}/result"
        )

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str):
        """Получение статуса задачи"""
        result = await queue_manager.get_task_result(job_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        # Унифицированная обработка progress
        raw_progress = getattr(result, 'progress', None)
        progress = None
        if isinstance(raw_progress, dict):
            from .schemas import ProgressInfo
            try:
                progress = ProgressInfo(**raw_progress)
            except Exception:
                progress = None
        elif hasattr(raw_progress, "__dict__"):
            progress = raw_progress
        else:
            progress = None

        return JobStatusResponse(
            job_id=job_id,
            status=result.status,
            message=result.message,
            progress=progress,
            timestamps={
                "created_at": getattr(result, 'created_at', None),
                "started_at": getattr(result, 'started_at', None),
                "completed_at": getattr(result, 'completed_at', None)
            },
            file_info=getattr(result, 'file_info', None),
            result=result.result if result.status == TaskStatus.COMPLETED else None,
            error=result.error
        )

    @app.get("/api/v1/jobs/{job_id}/result")
    async def get_job_result(job_id: str):
        """Получение результата аннотации задачи"""
        result = await queue_manager.get_task_result(job_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        if result.status == TaskStatus.FAILED:
            return {
                "job_id": job_id,
                "status": result.status,
                "error": {
                    "code": "PROCESSING_FAILED",
                    "message": result.error or "Unknown error",
                    "timestamp": datetime.now().isoformat()
                },
                "file_info": getattr(result, 'file_info', None)
            }
        
        if result.status != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Job is {result.status}, result not ready"
            )
        
        # Путь к файлам результата
        volume_path = Path(os.getenv("VOLUME_PATH", "/volume"))
        completed_dir = volume_path / "completed" / job_id
        
        return {
            "job_id": job_id,
            "status": result.status,
            "result": result.result,
            "result_files": {
                "annotation_json": str(completed_dir / "result.json"),
                "processed_audio": str(completed_dir / (getattr(result, 'filename', 'audio.wav')))
            }
        }

    @app.delete("/api/v1/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_job(job_id: str):
        """Удаление задачи"""
        success = await queue_manager.cancel_task(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or cannot be cancelled"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/v1/queue/status")
    async def get_queue_status():
        """Получение статуса очереди"""
        queue_info = await queue_manager.get_queue_info()
        return {
            "queue_length": queue_info.get("queue_length", 0),
            "processing_jobs": queue_info.get("processing_jobs", []),
            "queued_jobs": queue_info.get("queued_jobs", []),
            "average_processing_time": queue_info.get("average_processing_time", 0),
            "system_load": {
                "cpu_usage": 0.0,  # TODO: реализовать мониторинг
                "memory_usage": 0.0
            }
        }

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        """WebSocket эндпоинт для реального времени обновлений"""
        await websocket_manager.connect(websocket, client_id)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    await websocket_manager.send_personal_message(
                        {
                            "type": "error",
                            "code": "INVALID_JSON",
                            "message": "Invalid JSON format",
                            "timestamp": datetime.now().isoformat()
                        },
                        client_id,
                    )
                    continue

                await handle_websocket_message(message, client_id, websocket_manager, queue_manager)

        except WebSocketDisconnect:
            websocket_manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
            websocket_manager.disconnect(client_id)

    async def handle_websocket_message(message: dict, client_id: str, ws_manager: WebSocketManager, q_manager: QueueManager):
        """Обработка WebSocket сообщений"""
        msg_type = message.get("type")
        
        if msg_type == "ping":
            await ws_manager.send_personal_message(
                {
                    "type": "pong",
                    "timestamp": message.get("timestamp", datetime.now().isoformat())
                },
                client_id,
            )
        
        elif msg_type == "create_job":
            filename = message.get("filename")
            if not filename:
                await ws_manager.send_personal_message(
                    {
                        "type": "error",
                        "code": "MISSING_FILENAME",
                        "message": "Filename is required",
                        "timestamp": datetime.now().isoformat()
                    },
                    client_id,
                )
                return
            
            try:
                # Проверка файла и создание задачи (аналогично REST API)
                volume_path = Path(os.getenv("VOLUME_PATH", "/volume"))
                incoming_path = volume_path / "incoming" / filename
                
                if not incoming_path.exists():
                    await ws_manager.send_personal_message(
                        {
                            "type": "error",
                            "code": "FILE_NOT_FOUND",
                            "message": f"File '{filename}' not found in /volume/incoming/",
                            "timestamp": datetime.now().isoformat()
                        },
                        client_id,
                    )
                    return
                
                job_id = str(uuid.uuid4())
                metadata = create_task_metadata(
                    task_id=job_id,
                    file_path=str(incoming_path),
                    filename=filename,
                    priority=message.get("priority", 5),
                    websocket_client_id=client_id
                )
                
                await q_manager.add_task(job_id, metadata)
                
                await ws_manager.send_personal_message(
                    {
                        "type": "job_created",
                        "job_id": job_id,
                        "status": "queued",
                        "message": "Job created successfully",
                        "timestamp": datetime.now().isoformat()
                    },
                    client_id,
                )
                
            except Exception as e:
                await ws_manager.send_personal_message(
                    {
                        "type": "error",
                        "code": "JOB_CREATION_FAILED",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    },
                    client_id,
                )
        
        elif msg_type == "subscribe_job":
            job_id = message.get("job_id")
            if job_id:
                await q_manager.subscribe_to_task(job_id, client_id)
                await ws_manager.send_personal_message(
                    {
                        "type": "subscribed",
                        "job_id": job_id,
                        "message": f"Subscribed to job {job_id}",
                        "timestamp": datetime.now().isoformat()
                    },
                    client_id,
                )
        
        else:
            await ws_manager.send_personal_message(
                {
                    "type": "error",
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Unknown message type: {msg_type}",
                    "timestamp": datetime.now().isoformat()
                },
                client_id,
            )

    # Сохранение времени запуска для health check
    app.state.start_time = datetime.now().timestamp()
    
    return app


# Глобальный экземпляр приложения
app = create_app()
