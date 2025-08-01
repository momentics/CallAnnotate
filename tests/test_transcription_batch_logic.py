# tests/test_transcription_batch_logic.py

import pytest
import whisper
from unittest.mock import MagicMock

from app.stages.transcription import TranscriptionStage

@pytest.mark.asyncio
async def test_batch_logic(monkeypatch):
    # Мокаем whisper.load_model и метод transcribe
    fake_model = MagicMock(
        transcribe=lambda batch, **opts: {
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "text": "batch",
                    "no_speech_prob": 0.05,
                    "avg_logprob": -0.1,
                    "words": [{"start": 0, "end": 1, "word": "batch", "probability": 0.95}]
                }
            ],
            "confidence": 0.9
        }
    )
    monkeypatch.setattr(whisper, "load_model", lambda size, device=None: fake_model)

    cfg = {
        "model": "whisper-base",
        "device": "cpu",
        "language": "auto",
        "task": "transcribe",
        "batch_size": 5,
        "min_segment_duration": 0.1,
        "max_silence_between": 0.2,
        "min_overlap": 0.3,
    }
    stage = TranscriptionStage(cfg, MagicMock())
    await stage._initialize()

    # Подав сегмент для привязки спикера, чтобы текстовые поля формировались
    aligned = await stage._process_impl(
        "audio.wav",
        "jid",
        {"segments": [{"start": 0, "end": 1, "speaker": "spk"}]}
    )

    # Проверяем ключи и значения
    assert aligned["confidence"] == pytest.approx(0.9)
    assert aligned["segments"][0]["speaker"] == "spk"
    assert aligned["words"][0]["word"] == "batch"
    # Уверенность точно передалась
    assert "confidence" in aligned
    # Метрики avg_logprob и no_speech_prob присутствуют
    assert "avg_logprob" in aligned and "no_speech_prob" in aligned
