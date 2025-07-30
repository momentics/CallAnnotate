# src/app/api/__init__.py

from fastapi import APIRouter
from .routers import health, jobs, ws, voices, contacts_v2

api_router = APIRouter()
api_router.include_router(health.router, prefix="/api/v1")
api_router.include_router(jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
api_router.include_router(voices.router, prefix="/api/v1/voices", tags=["Voices"])
api_router.include_router(ws.router, prefix="/ws", tags=["WebSocket"])

api_router.include_router(contacts_v2.router)

