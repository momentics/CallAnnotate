# tests/unit/stages/test_diarization_stage.py

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.stages.diarization import DiarizationStage
from app.config import AppSettings

class TestDiarizationStage:
    @pytest.fixture
    def config(self):
        return AppSettings()
    
    @pytest.fixture
    def diar_config(self):
        return {
            "model": "pyannote/speaker-diarization-3.1",
            "device": "cpu",
            "window_enabled": False,
            "batch_size": 16
        }
    
    @pytest.fixture
    def mock_models_registry(self):
        registry = Mock()
        registry.get_model.return_value = Mock()
        return registry
    
    @pytest.fixture
    def stage(self, config, diar_config, mock_models_registry):
        return DiarizationStage(config, diar_config, mock_models_registry)
    
    @pytest.mark.asyncio
    async def test_initialize_loads_pipeline(self, stage, mock_models_registry):
        """Проверяет, что при наличии models_registry используется его get_model"""
        with patch('app.stages.diarization.Pipeline.from_pretrained') as mock_from_pretrained:
            await stage._initialize()
            # Pipeline.from_pretrained should NOT be called when models_registry is provided
            mock_from_pretrained.assert_not_called()
            # Instead, models_registry.get_model must be called once
            mock_models_registry.get_model.assert_called_once()
    
