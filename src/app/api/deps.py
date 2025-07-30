# src/app/api/deps.py

from ..queue.manager import AsyncQueueManager
from ..config import load_settings
import os

_settings = load_settings()
_queue: AsyncQueueManager | None = None

async def get_queue() -> AsyncQueueManager:
    global _queue
    if _queue is None:
        # Передаём токен HF_TOKEN из окружения в конфиг
        config_dict = _settings.dict()
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            # прокидываем токен для этапа диаризации
            config_dict["diarization"]["use_auth_token"] = hf_token
        _queue = AsyncQueueManager(config_dict)
        await _queue.start()
    return _queue
