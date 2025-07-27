# -*- coding: utf8 -*-
"""
FastAPI приложение CallAnnotate с WebSocket и REST/JSON API

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, Any

import yaml
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    status,
    WebSocket,
    WebSocketDisconnect,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Patch TestClient to accept keyword arg 'app'
try:
    from fastapi.testclient import TestClient as _FastAPITestClient
    from starlette.testclient import TestClient as _StarletteTestClient

    # Preserve original __init__
    _orig_init_fastapi = _FastAPITestClient.__init__
    _orig_init_starlette = _StarletteTestClient.__init__

    def _patched_init(self, *args, app=None, **kwargs):
        if app is not None:
            return _orig_init_fastapi(self, app, **kwargs) if isinstance(self, _FastAPITestClient) else _orig_init_starlette(self, app, **kwargs)
        return _orig_init_fastapi(self, *args, **kwargs) if isinstance(self, _FastAPITestClient) else _orig_init_starlette(self, *args, **kwargs)

    _FastAPITestClient.__init__ = _patched_init
    _StarletteTestClient.__init__ = _patched_init
except ImportError:
    pass

from .queue_manager import QueueManager, TaskStatus
from .utils import setup_logging, validate_audio_file, create_task_metadata
from .annotation import AnnotationService


class TaskResponse(BaseModel):
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
            "queue_length": await queue_manager.get_queue_size(),
            "active_tasks": await queue_manager.get_active_tasks_count(),
        }

    @app.get("/info")
    async def info():
        return {
            "service": "CallAnnotate",
            "version": config.get("server", {}).get("version", "1.0.0"),
            "description": "Сервис автоматической аннотации телефонных разговоров",
            "max_file_size": int(
                os.getenv("MAX_FILE_SIZE", config.get("files", {}).get("max_size", 1073741824))
            ),
        }

    @app.post("/jobs", status_code=status.HTTP_201_CREATED)
    async def create_job(file: UploadFile = File(...)):
        # Проверка MIME
        if not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported media type",
            )
        vr = validate_audio_file(file)
        if not vr.is_valid:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=vr.error
            )

        max_size = int(
            os.getenv("MAX_FILE_SIZE", config.get("files", {}).get("max_size", 500 * 1024 * 1024))
        )
        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large"
            )

        task_id = str(uuid.uuid4())
        upload_dir = Path(os.getenv("VOLUME_PATH", "/app/volume")) / "queue" / "incoming"
        upload_dir.mkdir(parents=True, exist_ok=True)
        path = upload_dir / f"{task_id}_{file.filename}"
        path.write_bytes(content)

        metadata = create_task_metadata(task_id, str(path), file.filename)
        await queue_manager.add_task(task_id, metadata)

        return {
            "job_id": task_id,
            "status": "queued",
            "message": "Job queued",
            "progress_url": f"/jobs/{task_id}",
            "result_url": f"/jobs/{task_id}/result",
        }

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        result = await queue_manager.get_task_result(job_id)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        return {
            "job_id": job_id,
            "status": result.status,
            "message": result.message,
            "result": result.result,
            "error": result.error,
        }

    @app.get("/jobs/{job_id}/result", status_code=status.HTTP_200_OK)
    async def get_job_result(job_id: str):
        """Скачивание или получение результата аннотации задачи"""
        result = await queue_manager.get_task_result(job_id)
        if not result:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        if result.status != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Result not ready"
            )
        return result.result

    @app.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_job(job_id: str):
        success = await queue_manager.cancel_task(job_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or cannot cancel"
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        await websocket_manager.connect(websocket, client_id)
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket_manager.send_personal_message(
                        {"type": "pong", "timestamp": message.get("timestamp")},
                        client_id,
                    )
                elif message.get("type") == "subscribe_task":
                    tid = message.get("task_id")
                    if tid:
                        await queue_manager.subscribe_to_task(tid, client_id)
                        await websocket_manager.send_personal_message(
                            {"type": "subscribed", "task_id": tid, "message": f"Subscribed to {tid}"},
                            client_id,
                        )
                elif message.get("type") == "upload_audio":
                    audio_data = message.get("data")
                    filename = message.get("filename", "audio.wav")
                    tid = str(uuid.uuid4())
                    upload_dir = (
                        Path(os.getenv("VOLUME_PATH", "/app/volume"))
                        / "queue"
                        / "incoming"
                    )
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    file_path = upload_dir / f"{tid}_{filename}"
                    audio_bytes = base64.b64decode(audio_data)
                    file_path.write_bytes(audio_bytes)
                    metadata = create_task_metadata(
                        task_id=tid,
                        file_path=str(file_path),
                        filename=filename,
                        priority=message.get("priority", 5),
                        websocket_client_id=client_id,
                    )
                    await queue_manager.add_task(tid, metadata)
                    await websocket_manager.send_personal_message(
                        {"type": "task_created", "task_id": tid, "status": "queued"},
                        client_id,
                    )
        except WebSocketDisconnect:
            websocket_manager.disconnect(client_id)
        except Exception as e:
            logging.getLogger(__name__).error(f"WS error: {e}")
            websocket_manager.disconnect(client_id)

    return app


# Глобальный экземпляр приложения для тестирования
app = create_app()
