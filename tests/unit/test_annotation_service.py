# tests/unit/test_annotation_service.py

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.annotation import AnnotationService
from app.config import AppSettings
from app.schemas import AudioMetadata

class TestAnnotationService:
    @pytest.fixture
    def config(self):
        return AppSettings()
    
    @pytest.fixture
    def annotation_service(self, config):
        return AnnotationService(config)
    
    @pytest.mark.asyncio
    async def test_process_audio_full_pipeline(self, annotation_service):
        """Integration test for the full annotation pipeline with mocked stages."""
        # Mock each stage
        mock_stages = []
        for i in range(5):
            stage = Mock()
            stage.stage_name = f"stage_{i}"
            stage.process = AsyncMock(return_value=Mock(
                stage_name=f"stage_{i}",
                processing_time=1.0,
                model_info={"test": "info"},
                payload={"test": "data"},
                success=True,
                error=None
            ))
            mock_stages.append(stage)
        annotation_service.stages = mock_stages
        
        # Prepare a real AudioMetadata instance for extract_audio_metadata to return
        audio_meta = AudioMetadata(
            filename="test.wav",
            duration=10.0,
            sample_rate=16000,
            channels=1,
            format="wav",
            bitrate=128000,
            size_bytes=1000
        )

        with patch('app.annotation.extract_audio_metadata', return_value=audio_meta), \
             patch('app.annotation.ensure_directory'), \
             patch('shutil.copy'), \
             patch('shutil.move'):
            result = await annotation_service.process_audio(
                "test.wav",
                "task1",
                progress_callback=None
            )
        
        # Ensure all stages were invoked
        for stage in mock_stages:
            stage.process.assert_called_once()
        
        # Check that result contains expected keys
        assert result["task_id"] == "task1"
        assert "version" in result
        assert "audio_metadata" in result
        assert isinstance(result["audio_metadata"], dict)
        assert "processing_info" in result
    
    @pytest.mark.asyncio
    async def test_update_progress_callback(self, annotation_service):
        """Test that the progress callback is invoked correctly."""
        callback = AsyncMock()
        await annotation_service._update_progress(callback, 50, "Test message")
        callback.assert_called_once_with(50, "Test message")
    
    def test_build_final_annotation(self, annotation_service):
        """Test assembling of final annotation with proper word payload."""
        # Mock stage results
        context = {
            "diarization": Mock(
                payload={"segments": [{"start": 0, "end": 3, "speaker": "speaker_1"}]},
                model_info={"test": "info"},
                processing_time=1.0
            ),
            "transcription": Mock(
                payload={
                    "words": [
                        {"start": 0, "end": 1, "word": "test", "probability": 1.0, "speaker": "speaker_1"}
                    ],
                    "confidence": 0.9,
                    "language": "en"
                },
                model_info={"test": "info"},
                processing_time=2.0
            ),
            "recognition": Mock(
                payload={"speakers": {}},
                model_info={"test": "info"},
                processing_time=0.5
            ),
            "carddav": Mock(
                payload={"speakers": {}},
                model_info={"test": "info"},
                processing_time=0.1
            )
        }
        audio_metadata = AudioMetadata(
            filename="test.wav",
            duration=10.0,
            sample_rate=16000,
            channels=1,
            format="wav",
            bitrate=None,
            size_bytes=1000
        )
        
        result = annotation_service._build_final_annotation("task1", audio_metadata, context)
        
        # Validate final annotation structure
        assert result.task_id == "task1"
        assert len(result.speakers) == 1
        assert len(result.segments) == 1
        assert result.segments[0].text == "test"
        assert result.statistics.total_words == 1
