# src/app/api/routers/ws.py

# -*- coding: utf-8 -*-
"""
WebSocket роутер для CallAnnotate
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import json
import logging
import os
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException

from ...core.interfaces.queue import QueueService
from ..deps import get_queue
from ...utils import validate_audio_file_path, create_task_metadata
from ...config import load_settings

router = APIRouter()

class WebSocketManager:
    def __init__(self, ping_interval: int):
        self._clients: Dict[str, WebSocket] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
        self._ping_interval = ping_interval

    async def connect(self, ws: WebSocket, client_id: str):
        await ws.accept()
        self._clients[client_id] = ws
        self._logger.info("WS client %s connected", client_id)
        asyncio.create_task(self._heartbeat(client_id))

    def disconnect(self, client_id: str):
        ws = self._clients.pop(client_id, None)
        if ws:
            self._logger.info("WS client %s disconnected", client_id)

    async def send(self, client_id: str, message: dict):
        ws = self._clients.get(client_id)
        if not ws:
            return
        try:
            await ws.send_text(json.dumps(message))
        except Exception as e:
            self._logger.error("WS send error to %s: %s", client_id, e)
            self.disconnect(client_id)

    async def _heartbeat(self, client_id: str):
        """
        Периодическая отправка server-initiated ping сообщения.
        """
        while client_id in self._clients:
            try:
                await asyncio.sleep(self._ping_interval)
                await self.send(client_id, {
                    "type": "ping",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                self._logger.error("Heartbeat error for %s: %s", client_id, e)
                break

# Инициализация WS-менеджера с параметром из конфигурации
cfg = load_settings()
ws_manager = WebSocketManager(ping_interval=cfg.notifications.websocket.ping_interval)

@router.websocket("/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    queue: QueueService = Depends(get_queue),
):
    await ws_manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await ws_manager.send(client_id, {
                    "type": "error",
                    "code": "INVALID_JSON",
                    "message": "Invalid JSON",
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue

            msg_type = message.get("type")

            if msg_type == "ping":
                # Ответ на клиентский ping
                await ws_manager.send(client_id, {
                    "type": "pong",
                    "timestamp": message.get("timestamp")
                })

            elif msg_type == "create_job":
                filename = message.get("filename")
                if not filename:
                    await ws_manager.send(client_id, {
                        "type": "error",
                        "code": "MISSING_FILENAME",
                        "message": "Filename required",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue

                cfg = load_settings()
                vol = Path(os.getenv("VOLUME_PATH", cfg.queue.volume_path)) / "incoming" / filename

                vr = validate_audio_file_path(str(vol))
                if not vr.is_valid:
                    await ws_manager.send(client_id, {
                        "type": "error",
                        "code": "FILE_NOT_FOUND",
                        "message": vr.error,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue

                job_id = str(uuid.uuid4())
                meta = create_task_metadata(
                    job_id, str(vol), filename, message.get("priority", 5), client_id
                )
                await queue.add_task(job_id, meta)
                await ws_manager.send(client_id, {
                    "type": "job_created",
                    "job_id": job_id,
                    "status": "queued",
                    "timestamp": datetime.utcnow().isoformat()
                })

            elif msg_type == "subscribe_job":
                job_id = message.get("job_id")
                if not job_id:
                    raise HTTPException(400, "job_id required")
                await queue.subscribe_to_task(job_id, client_id)
                await ws_manager.send(client_id, {
                    "type": "subscribed",
                    "job_id": job_id,
                    "timestamp": datetime.utcnow().isoformat()
                })

            else:
                await ws_manager.send(client_id, {
                    "type": "error",
                    "code": "UNKNOWN_TYPE",
                    "message": f"Unknown type {msg_type}",
                    "timestamp": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
