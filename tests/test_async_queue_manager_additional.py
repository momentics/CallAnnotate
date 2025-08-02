# tests/test_async_queue_manager_additional.py

import asyncio
import pytest
from datetime import datetime, timedelta

import app.queue.manager as qm_mod


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    cfg = {
        "queue": {
            "volume_path": str(vol),
            "max_concurrent_tasks": 2,
            "max_queue_size": 5,
            "task_timeout": 10,
            "cleanup_interval": 1
        }
    }
    return cfg


@pytest.mark.asyncio
async def test_cancel_before_start(cfg):
    """Отмена задачи до начала обработки"""
    q = qm_mod.AsyncQueueManager(cfg)
    # Добавляем задачу, но не запускаем менеджер
    await q.add_task("jobX", {"file_path": "p"})
    # Отмена до старта — успешно
    ok = await q.cancel_task("jobX")
    assert ok
    tr = await q.get_task_result("jobX")
    assert tr.status == qm_mod.TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_update_progress_and_subscribe(cfg):
    """Проверка _update_progress и подписки клиентов"""
    q = qm_mod.AsyncQueueManager(cfg)
    await q.add_task("job1", {"file_path": "p"})
    # Подписываемся
    await q.subscribe_to_task("job1", "clientA")
    # Вызываем внутренний update
    await q._update_progress("job1", 42, "msg")
    tr = await q.get_task_result("job1")
    assert tr.progress == 42
    # Клиент должен быть в подписчиках
    assert "clientA" in tr.subscribers


@pytest.mark.asyncio
async def test_cleanup_once_old_and_recent(cfg):
    """Проверка cleanup_once удаляет только устаревшие задачи"""
    q = qm_mod.AsyncQueueManager(cfg)
    # Новая задача
    await q.add_task("new", {"file_path": "p"})
    # Завершаем её
    await q.cancel_task("new")
    # Старая задача создаем вручную
    await q.add_task("old", {"file_path": "p"})
    tr_old = await q.get_task_result("old")
    tr_old.status = qm_mod.TaskStatus.COMPLETED
    # Помечаем старую задачу устаревшей (>7 дней)
    past = (datetime.utcnow() - timedelta(days=8)).isoformat()
    tr_old.updated_at = past
    # Запускаем cleanup_once
    await q.cleanup_once()
    # Старая задача удалена
    assert await q.get_task_result("old") is None
    # Новая задача по-прежнему существует (не стала устаревшей)
    assert await q.get_task_result("new") is not None
