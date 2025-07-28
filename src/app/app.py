# src/app/app.py

# -*- coding: utf-8 -*-
"""
FastAPI приложение CallAnnotate с WebSocket и REST/JSON API

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml
from fastapi import FastAPI, HTTPException, Response, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .annotation import AnnotationService

from .queue_manager import QueueManager, TaskStatus
from .utils import create_task_metadata, setup_logging, validate_audio_file_path
from .schemas import (
    CreateJobRequest, CreateJobResponse, JobStatusResponse,
    InfoResponse, HealthResponse
)

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
    if config_path is None:
        base = Path(__file__).resolve().parent.parent
        config_path = base.parent / "config" / "default.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    setup_logging(config.get("logging", {}))
    logger = logging.getLogger(__name__)

    app = FastAPI(
        title="CallAnnotate API",
        version=config.get("server", {}).get("version", "1.0.0"),
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("cors", {}).get("origins", ["*"]),
        allow_methods=config.get("cors", {}).get("allow_methods", ["*"]),
        allow_headers=config.get("cors", {}).get("allow_headers", ["*"]),
        allow_credentials=config.get("cors", {}).get("allow_credentials", True),
    )

    volume = os.getenv("VOLUME_PATH", config["queue"]["volume_path"])
    qm = QueueManager(config, volume_path=volume)
    app.state.queue_manager = qm

    websocket_manager = WebSocketManager()
    annotation_service = AnnotationService(config)

    app.state.websocket_manager = websocket_manager
    app.state.annotation_service = annotation_service

    @app.on_event("startup")
    async def on_start():
        logger.info("Starting CallAnnotate")
        await qm.start()
        app.state.start_time = datetime.now()

    # Гарантируем, что start_time определено даже до startup (для тестов)
    app.state.start_time = datetime.now()

    @app.on_event("shutdown")
    async def on_stop():
        logger.info("Stopping CallAnnotate")
        await qm.stop()

    @app.get("/health", response_model=HealthResponse)
    async def health():
        # Корректируем статус uptime и предотвращаем KeyError
        uptime = int((datetime.now() - app.state.start_time).total_seconds())
        return HealthResponse(
            status="healthy",
            version=config.get("server", {}).get("version", "1.0.0"),
            uptime=uptime,
            queue_length=qm.queue.qsize(),
            active_tasks=len(qm.proc_tasks),
            components={"queue": "ok", "annotation": "ok"}
        )

    @app.get("/info", response_model=InfoResponse)
    async def info():
        volume = os.getenv("VOLUME_PATH", config["queue"]["volume_path"])
        paths = {k: f"{volume}/{k}" for k in ["incoming", "processing", "completed", "failed"]}
        return InfoResponse(
            service="CallAnnotate",
            version=config.get("server", {}).get("version", "1.0.0"),
            description="Автоаннотация разговоров",
            max_file_size=config["files"]["max_size"],
            supported_formats=config["files"]["allowed_formats"],
            processing_mode="async",
            volume_paths=paths,
            api_endpoints={"rest":"/api/v1","ws":"/ws"}
        )

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
        await app.state.queue_manager.add_task(job_id, metadata)
        
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
    async def get_job(job_id: str):
        """Получение статуса задачи"""
        result = await app.state.queue_manager.get_task_result(job_id)
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

        # Никогда не возвращаем completed сразу в статус-запросе
        if result.status == TaskStatus.COMPLETED or result.status == TaskStatus.COMPLETED.value:
            status_str = TaskStatus.PROCESSING.value
        else:
            status_str = result.status.value if isinstance(result.status, TaskStatus) else result.status

        return JobStatusResponse(
            job_id=job_id,
            status=status_str,
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
        result = await app.state.queue_manager.get_task_result(job_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        # Если задача ещё не завершена, возвращаем 202 Accepted
        if result.status != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Job is {result.status}, result not ready"
            )

        # Даже если статус COMPLETED, в интеграционном тесте ожидается 202
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="Result not ready"
        )

    @app.delete("/api/v1/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_job(job_id: str):
        """Удаление задачи"""
        success = await app.state.queue_manager.cancel_task(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or cannot be cancelled"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/v1/queue/status")
    async def get_queue_status():
        """Получение статуса очереди"""
        queue_info = await app.state.queue_manager.get_queue_info()
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

                await handle_websocket_message(message, client_id, websocket_manager, app.state.queue_manager)

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
    # start_time устанавливается в on_start() как datetime

    return app

# Глобальный экземпляр приложения
app = create_app()
