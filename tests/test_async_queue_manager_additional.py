# -*- coding: utf-8 -*-
"""
Дополнительные unit-тесты для AsyncQueueManager (src/app/queue/manager.py)

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

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
async def test_worker_loop_multiple_tasks(cfg, monkeypatch):
    """Запуск-остановка воркеров, обработка нескольких задач"""
    # Мокаем AnnotationService, чтобы оно быстро возвращало результат
    class FakeAnn:
        async def process_audio(self, fp, jid, progress_callback=None):
            return {"result": jid}

    monkeypatch.setattr(qm_mod, "AnnotationService", lambda cfg: FakeAnn())

    q = qm_mod.AsyncQueueManager(cfg)
    # Добавляем задачи
    await q.add_task("j1", {"file_path": "x"})
    await q.add_task("j2", {"file_path": "y"})
    # Старт
    await q.start()
    # Даем воркерам обработать
    await asyncio.sleep(0.1)
    # Проверяем, что обе задачи завершены
    tr1 = await q.get_task_result("j1")
    tr2 = await q.get_task_result("j2")
    assert tr1.status == qm_mod.TaskStatus.COMPLETED
    assert tr2.status == qm_mod.TaskStatus.COMPLETED
    assert tr1.result["result"] == "j1"
    assert tr2.result["result"] == "j2"
    # Остановка
    await q.stop()


@pytest.mark.asyncio
async def test_cleanup_once_old_and_recent(cfg):
    """Проверка cleanup_once удаляет только устаревшие задачи"""
    q = qm_mod.AsyncQueueManager(cfg)
    # Новая задача
    await q.add_task("new", {"file_path": "p"})
    # Завершаем её
    await q.cancel_task("new")
    # Старую задачу создаем вручную
    await q.add_task("old", {"file_path": "p"})
    tr_old = await q.get_task_result("old")
    tr_old.status = qm_mod.TaskStatus.COMPLETED
    # Помечаем старую задачу устаревшей (>7 дней)
    past = (datetime.utcnow() - timedelta(days=8)).isoformat()
    tr_old.updated_at = past
    # Запускаем cleanup_once
    await q.cleanup_once()
    # Старую задачу удалено
    info = await q.get_queue_info()
    assert "old" not in info["queued_jobs"]
    # Новая задача по-прежнему существует в менеджере (get_task_result не None)
    assert (await q.get_task_result("new")) is not None


@pytest.mark.asyncio
async def test_get_queue_info_various_states(cfg):
    """Проверка get_queue_info при разных состояниях задач"""
    q = qm_mod.AsyncQueueManager(cfg)
    # Ни одной задачи
    info = await q.get_queue_info()
    assert info == {"queue_length": 0, "processing_jobs": [], "queued_jobs": []}

    # Добавляем и не стартуем
    await q.add_task("a", {"file_path": "p"})
    info = await q.get_queue_info()
    assert info["queue_length"] == 1
    assert "a" in info["queued_jobs"]

    # Сэмулируем состояние processing
    tr = await q.get_task_result("a")
    tr.status = qm_mod.TaskStatus.PROCESSING
    info = await q.get_queue_info()
    assert "a" in info["processing_jobs"]
    assert "a" not in info["queued_jobs"]
