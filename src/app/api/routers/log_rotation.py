from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
from pathlib import Path
import glob

LoggingRouter = APIRouter(prefix="/api/v1/logs", tags=["LogRotation"])

@LoggingRouter.get("/files", summary="Список файлов логов с ротацией")
async def list_log_files():
    volume = os.getenv("VOLUME_PATH")
    if not volume:
        raise HTTPException(404, "VOLUME_PATH не задан")
    patterns = [os.getenv("LOG_FILE_PATH") or str(Path(volume) / "logs" / "callannotate*.log")]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    return JSONResponse([Path(f).name for f in files])
