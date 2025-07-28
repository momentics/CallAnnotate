# src/app/api/routers/health.py

from datetime import datetime
import os
from fastapi import APIRouter, Depends, Request
from ...schemas import HealthResponse, InfoResponse
from ..deps import get_queue
from ...config import load_settings
from pathlib import Path

router = APIRouter()

CFG = load_settings()

@router.get("/health", response_model=HealthResponse, tags=["Service"])
async def health(request: Request, queue=Depends(get_queue)) -> HealthResponse:
    start_time = getattr(request.app.state, "start_time", None)
    uptime = int((datetime.utcnow() - start_time).total_seconds()) if start_time else 0
    info = await queue.get_queue_info()
    return HealthResponse(
        status="healthy",
        version=CFG.server.version,
        uptime=uptime,
        queue_length=info.get("queue_length", 0),
        active_tasks=len(info.get("processing_jobs", [])),
        components={"queue": "ok", "annotation": "ok"},
    )

@router.get("/info", response_model=InfoResponse, tags=["Service"])
async def info(request: Request) -> InfoResponse:
    # При отсутствии state.volume_path используем переменную окружения VOLUME_PATH
    raw_vol = (
        request.app.state.volume_path
        if hasattr(request.app.state, "volume_path")
        else os.getenv("VOLUME_PATH", CFG.queue.volume_path)
    )
    # Normalize to POSIX-style path for consistent forward slashes
    vol = Path(raw_vol).as_posix()
    return InfoResponse(
        service="CallAnnotate",
        version=CFG.server.version,
        description="Автоматическая диаризация, транскрипция и аннотация разговоров.",
        max_file_size=CFG.files.max_size,
        supported_formats=CFG.files.allowed_formats,
        processing_mode="async",
        volume_paths={
            "incoming": f"{vol}/incoming",
            "processing": f"{vol}/processing",
            "completed": f"{vol}/completed",
            "failed": f"{vol}/failed",
        },
        api_endpoints={"rest": f"{CFG.api.base_path}", "ws": f"/ws/{{client_id}}"},
    )
