# src/app/api/deps.py

from ..queue.manager import AsyncQueueManager
from ..config import load_settings

_settings = load_settings()
_queue: AsyncQueueManager | None = None

async def get_queue() -> AsyncQueueManager:
    global _queue
    if _queue is None:
        # Инициализация менеджера очереди с обязательным созданием структуры
        _queue = AsyncQueueManager(_settings.dict())
        await _queue.start()
    return _queue
