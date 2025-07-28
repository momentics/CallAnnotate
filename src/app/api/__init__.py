# src/app/api/__init__.py

from fastapi import APIRouter
from .routers import health, jobs, ws

api_router = APIRouter()
api_router.include_router(health.router, prefix="/api/v1")
api_router.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
api_router.include_router(ws.router, prefix="/ws", tags=["WebSocket"])
