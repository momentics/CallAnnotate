# src/app/app.py

# -*- coding: utf-8 -*-
"""
CallAnnotate – основное приложение FastAPI

Перенесена логика WebSocket в отдельный роутер.
"""
from __future__ import annotations

import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_settings
from .utils import setup_logging
from .api import api_router

# Инициализация конфигурации и логирования
CFG = load_settings()
setup_logging(CFG.logging.dict())
logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title="CallAnnotate API",
    version=CFG.server.version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Подключение CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.cors.origins,
    allow_credentials=CFG.cors.allow_credentials,
    allow_methods=CFG.cors.allow_methods,
    allow_headers=CFG.cors.allow_headers,
)

# Подключение всех роутеров
app.include_router(api_router)

# Сохранение времени запуска для health-endpoint
@app.on_event("startup")
async def on_startup():
    app.state.start_time = datetime.utcnow()
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Application shutdown complete")


