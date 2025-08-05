from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
from pathlib import Path

LoggingRouter = APIRouter()

@LoggingRouter.get("/", summary="Скачать текущий системный лог")
async def download_log():
    volume = os.getenv("VOLUME_PATH")
    if not volume:
        raise HTTPException(404, "VOLUME_PATH не задан")
    log_file = os.getenv("LOG_FILE_PATH") or os.path.join(volume, "logs", "callannotate.log")
    log_path = Path(log_file)
    if not log_path.exists():
        raise HTTPException(404, "Лог файл не найден")
    return FileResponse(str(log_path), media_type="text/plain", filename=log_path.name)
