# src/app/app.py

import os
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_settings
from .utils import setup_logging, ensure_volume_structure
from .api import api_router

# Загрузка настроек и инициализация логирования
CFG = load_settings()
setup_logging(CFG.logging.dict())
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CallAnnotate API",
    version=CFG.server.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.cors.origins,
    allow_credentials=CFG.cors.allow_credentials,
    allow_methods=CFG.cors.allow_methods,
    allow_headers=CFG.cors.allow_headers,
)

app.include_router(api_router)

@app.on_event("startup")
async def on_startup():
    # Время старта
    app.state.start_time = datetime.utcnow()

    # Путь к volume: сначала из переменной окружения, иначе из конфига
    vol = os.getenv("VOLUME_PATH", CFG.queue.volume_path)
    vol_path = Path(vol).expanduser().resolve()
    app.state.volume_path = str(vol_path)

    # Создать всю структуру volume
    try:
        ensure_volume_structure(str(vol_path))
        logger.info(f"Volume structure ensured at startup: {vol_path}")
    except Exception as e:
        logger.error(f"Critical error creating volume structure: {e}")
        raise

    logger.info("Application startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Application shutdown complete")
