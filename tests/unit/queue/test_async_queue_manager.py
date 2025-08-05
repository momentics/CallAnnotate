# tests/unit/queue/test_async_queue_manager.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from app.queue.manager import AsyncQueueManager, TaskResult, TaskStatus

class TestAsyncQueueManager:
    @pytest.fixture
    def config(self):
        config = Mock()
        config.queue.volume_path = "/tmp/test"
        config.queue.max_concurrent_tasks = 2
        config.queue.max_queue_size = 10
        config.queue.task_timeout = 300
        config.queue.cleanup_interval = 60
        return config
    
    @pytest.fixture
    def queue_manager(self, config):
        return AsyncQueueManager(config)
    
    @pytest.mark.asyncio
    async def test_add_task_success(self, queue_manager):
        """Проверяет успешное добавление задачи"""
        metadata = {
            "file_path": "/tmp/test.wav",
            "filename": "test.wav",
            "priority": 5
        }
        
        with patch('shutil.move') as mock_move:
            success = await queue_manager.add_task("job1", metadata)
            
            assert success is True
            assert "job1" in queue_manager._tasks
            assert queue_manager._tasks["job1"].status == TaskStatus.QUEUED
            mock_move.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_task_after_stop(self, queue_manager):
        """Проверяет отклонение задач после остановки"""
        queue_manager._stopped = True
        
        success = await queue_manager.add_task("job1", {})
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_cancel_task_success(self, queue_manager):
        """Проверяет отмену задачи"""
        # Добавляем задачу
        queue_manager._tasks["job1"] = TaskResult("job1", {})
        
        success = await queue_manager.cancel_task("job1")
        
        assert success is True
        assert queue_manager._tasks["job1"].status == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, queue_manager):
        """Проверяет отмену несуществующей задачи"""
        success = await queue_manager.cancel_task("nonexistent")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_get_task_result(self, queue_manager):
        """Проверяет получение результата задачи"""
        task_result = TaskResult("job1", {"test": "data"})
        queue_manager._tasks["job1"] = task_result
        
        result = await queue_manager.get_task_result("job1")
        
        assert result is task_result
    
    @pytest.mark.asyncio
    async def test_get_queue_info(self, queue_manager):
        """Проверяет получение информации об очереди"""
        # Добавляем тестовые задачи
        queue_manager._tasks["job1"] = TaskResult("job1", {})
        queue_manager._tasks["job1"].status = TaskStatus.PROCESSING
        
        queue_manager._tasks["job2"] = TaskResult("job2", {})
        queue_manager._tasks["job2"].status = TaskStatus.QUEUED
        
        info = await queue_manager.get_queue_info()
        
        assert "processing_jobs" in info
        assert "queued_jobs" in info
        assert "job1" in info["processing_jobs"]
        assert "job2" in info["queued_jobs"]
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, queue_manager):
        """Проверяет запуск и остановку менеджера"""
        await queue_manager.start()
        
        assert queue_manager._running.is_set()
        assert len(queue_manager._workers) == 2  # max_concurrent_tasks
        assert queue_manager._cleanup_task is not None
        
        await queue_manager.stop()
        
        assert not queue_manager._running.is_set()
        assert queue_manager._stopped is True
    
    @pytest.mark.asyncio 
    async def test_cleanup_once_removes_stale_tasks(self, queue_manager):
        """Проверяет удаление устаревших задач"""
        from datetime import datetime, timedelta
        
        # Создаём старую задачу
        old_task = TaskResult("old_job", {})
        old_task.status = TaskStatus.COMPLETED
        old_task.updated_at = (datetime.utcnow() - timedelta(days=8)).isoformat()
        
        # Создаём свежую задачу
        fresh_task = TaskResult("fresh_job", {})
        fresh_task.status = TaskStatus.COMPLETED
        fresh_task.updated_at = datetime.utcnow().isoformat()
        
        queue_manager._tasks["old_job"] = old_task
        queue_manager._tasks["fresh_job"] = fresh_task
        
        await queue_manager.cleanup_once()
        
        assert "old_job" not in queue_manager._tasks
        assert "fresh_job" in queue_manager._tasks
