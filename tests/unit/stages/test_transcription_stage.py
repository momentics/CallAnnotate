# tests/unit/stages/test_transcription_stage.py

import pytest
from unittest.mock import Mock, patch
from app.stages.transcription import TranscriptionStage

@pytest.fixture(autouse=True)
def disable_setup_logging(monkeypatch):
    # Disable setup_logging to avoid filesystem interactions and Mock path errors
    monkeypatch.setattr("app.utils.setup_logging", lambda cfg: None)

class TestTranscriptionStage:
    @pytest.fixture
    def config(self):
        return {
            "model": "openai/whisper-base",
            "device": "cpu",
            "language": "ru",
            "task": "transcribe",
            "min_segment_duration": 0.2,
            "max_silence_between": 0.0,
            "min_overlap": 0.3
        }

    @pytest.fixture
    def stage(self, config):
        app_config = Mock()
        app_config.queue = Mock()
        app_config.queue.volume_path = "/tmp/test"
        return TranscriptionStage(app_config, config, Mock())

    @pytest.mark.asyncio
    async def test_initialize_loads_whisper_model(self, stage):
        """Проверяет загрузку модели Whisper"""
        with patch('app.stages.transcription.whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            await stage._initialize()

            # load_model should be called at least once
            assert mock_load.call_count >= 1

    @pytest.mark.asyncio
    async def test_process_impl_returns_segments_and_words(self, stage):
        """Проверяет возврат сегментов и слов"""
        # ensure initialization to set segment filtering params
        with patch('app.stages.transcription.whisper.load_model'):
            await stage._initialize()

        # Подготовка мока модели Whisper
        mock_transcription_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Привет мир",
                    "words": [
                        {"start": 0.0, "end": 1.0, "word": "Привет", "probability": 0.99},
                        {"start": 1.0, "end": 3.0, "word": "мир", "probability": 0.95}
                    ],
                    "no_speech_prob": 0.01,
                    "avg_logprob": -0.1
                }
            ],
            "confidence": 0.97
        }

        stage.model = Mock()
        stage.model.transcribe.return_value = mock_transcription_result
        stage._initialized = True

        result = await stage._process_impl("test.wav", "task1", {})

        # Проверки
        assert "segments" in result
        assert "words" in result
        assert "confidence" in result
        assert len(result["segments"]) == 1
        assert len(result["words"]) == 2
        assert result["confidence"] == 0.97

    @pytest.mark.asyncio
    async def test_segment_speaker_assignment(self, stage):
        """Проверяет привязку сегментов к спикерам из диаризации"""
        # ensure initialization
        with patch('app.stages.transcription.whisper.load_model'):
            await stage._initialize()

        # Мок результатов диаризации
        diarization_results = {
            "segments": [
                {"start": 0.0, "end": 3.5, "speaker": "speaker_1"},
                {"start": 3.5, "end": 7.0, "speaker": "speaker_2"}
            ]
        }

        # Мок результатов Whisper
        whisper_result = {
            "segments": [
                {"start": 0.5, "end": 3.0, "text": "Hello", "words": [], "no_speech_prob": 0.01, "avg_logprob": -0.1},
                {"start": 4.0, "end": 6.0, "text": "World", "words": [], "no_speech_prob": 0.02, "avg_logprob": -0.15}
            ],
            "confidence": 0.95
        }

        stage.model = Mock()
        stage.model.transcribe.return_value = whisper_result
        stage._initialized = True

        result = await stage._process_impl("test.wav", "task1", diarization_results)

        # Проверяем привязку спикеров
        assert result["segments"][0]["speaker"] == "speaker_1"
        assert result["segments"][1]["speaker"] == "speaker_2"
