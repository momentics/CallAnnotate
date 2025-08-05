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
    
    @pytest.mark.asyncio
    async def test_process_monolithic_mode(self, stage):
        """Тестирует monolithic режим диаризации"""
        # Настройка моков
        mock_pipeline = Mock()
        mock_annotation = Mock()
        mock_annotation.itertracks.return_value = [
            (Mock(start=0.0, end=3.0), None, "speaker_1"),
            (Mock(start=3.0, end=6.0), None, "speaker_2")
        ]
        mock_pipeline.return_value = mock_annotation
        stage.pipeline = mock_pipeline
        stage._initialized = True
        
        # Выполнение
        result = await stage._process_impl("test.wav", "task1", {})
        
        # Проверки
        assert "segments" in result
        assert "speakers" in result
        assert len(result["segments"]) == 2
        assert result["total_speakers"] == 2
    
    @pytest.mark.asyncio
    async def test_process_windowed_mode(self, stage):
        """Тестирует windowed режим диаризации"""
        stage.cfg.window_enabled = True
        stage.cfg.window_size = 30.0
        stage.cfg.hop_size = 10.0
        
        mock_pipeline = Mock()
        # mock get_audio_duration to avoid entering fallback
        mock_pipeline.get_audio_duration.return_value = 60.0
        # simulate pipeline.crop and result for windowed
        from pyannote.core import Segment, Annotation
        ann = Annotation()
        ann[Segment(0, 30)] = "speaker_1"
        mock_pipeline.crop.return_value = ann
        stage.pipeline = mock_pipeline
        stage._initialized = True
        
        result = await stage._process_impl("test.wav", "task1", {})
        
        assert "segments" in result
        assert "total_speakers" in result
