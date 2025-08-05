# src/app/api/deps.py

from ..queue.manager import AsyncQueueManager
from ..config import load_settings
import os

_settings = load_settings()
_queue = None

async def get_queue() -> AsyncQueueManager:
    global _queue
    if _queue is None:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            # inject HF token into settings for diarization
            _settings.diarization.use_auth_token = hf_token
        _queue = AsyncQueueManager(_settings)
        await _queue.start()
    return _queue
