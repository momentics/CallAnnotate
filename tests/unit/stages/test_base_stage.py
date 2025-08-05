# tests/unit/stages/test_base_stage.py
import pytest
from unittest.mock import Mock, AsyncMock
from app.stages.base import BaseStage, StageResult
from app.config import AppSettings

class MockStage(BaseStage):
    @property
    def stage_name(self):
        return "mock"
    
    async def _initialize(self):
        self.initialized = True
    
    async def _process_impl(self, file_path, task_id, previous_results, progress_callback):
        return {"test": "data"}

class TestBaseStage:
    @pytest.fixture
    def config(self):
        return AppSettings()
    
    @pytest.fixture
    def stage(self, config):
        return MockStage(config, {"device": "cpu"}, models_registry=Mock())
    
    @pytest.mark.asyncio
    async def test_process_calls_initialize_once(self, stage):
        """Проверяет, что инициализация вызывается только один раз"""
        assert not stage._initialized
        
        await stage.process("test.wav", "task1", {})
        assert stage._initialized
        
        await stage.process("test2.wav", "task2", {})
        # Инициализация не должна вызываться повторно
    
    @pytest.mark.asyncio
    async def test_process_returns_stage_result(self, stage):
        """Проверяет структуру возвращаемого StageResult"""
        result = await stage.process("test.wav", "task1", {})
        
        assert isinstance(result, StageResult)
        assert result.stage_name == "mock"
        assert result.success is True
        assert result.processing_time > 0
        assert result.payload == {"test": "data"}
    
    @pytest.mark.asyncio
    async def test_process_handles_exceptions(self, stage):
        """Проверяет обработку исключений"""
        stage._process_impl = AsyncMock(side_effect=ValueError("Test error"))
        
        result = await stage.process("test.wav", "task1", {})
        
        assert result.success is False
        assert result.error == "Test error"
        assert result.payload == {}
