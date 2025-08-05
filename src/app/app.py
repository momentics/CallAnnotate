import asyncio
import os
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import load_settings
from .utils import ensure_volume_structure
from .api.routers.logging import LoggingRouter
from .api import api_router
from .api.deps import get_queue

app = FastAPI(
    title="CallAnnotate API",
    version="0.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(LoggingRouter, prefix="/api/v1/logs", tags=["Logs"])
app.include_router(api_router)

@app.on_event("startup")
async def on_startup():
    CFG = load_settings()
    app.version = CFG.server.version

    from .utils import setup_logging
    setup_logging(CFG.model_dump())

    logger = logging.getLogger(__name__)

    app.state.start_time = datetime.utcnow()

    vol = os.getenv("VOLUME_PATH", CFG.queue.volume_path)
    vol_path = Path(vol).expanduser().resolve()
    ensure_volume_structure(str(vol_path))
    logger.info(f"Volume structure ensured at startup: {vol_path}")

    cache_dir = vol_path / "models" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir)
    logger.info(f"HF cache redirected to: {cache_dir}")

    app.state.volume_path = str(vol_path)
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def on_shutdown():
    logger = logging.getLogger(__name__)
    try:
        queue = await get_queue()
        await queue.stop()
    except Exception as e:
        logger.error(f"Error stopping queue on shutdown: {e}")
    finally:
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
