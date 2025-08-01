# tests/test_stages_transcription.py

import pytest
import whisper
import numpy as np
from unittest.mock import MagicMock

from app.stages.transcription import TranscriptionStage

@pytest.mark.asyncio
async def test_transcription_align(monkeypatch):
    # Мокаем whisper.load_model и метод transcribe
    fake_model = MagicMock(
        transcribe=lambda batch, **opts: {
            "segments": [
                {
                    "start": 0,
                    "end": 1,
                    "text": "hi",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.2,
                    "words": [{"start": 0, "end": 1, "word": "hi", "probability": 0.9}]
                }
            ],
            "confidence": 0.8
        }
    )
    monkeypatch.setattr(whisper, "load_model", lambda m, device=None: fake_model)

    cfg = {
        "model": "whisper-base",
        "device": "cpu",
        "language": "auto",
        "task": "transcribe",
        "batch_size": 10,
        "min_segment_duration": 0.2,
        "max_silence_between": 0.3,
        "min_overlap": 0.3,
    }
    stage = TranscriptionStage(cfg, MagicMock())
    await stage._initialize()

    diar_segments = [{"start": 0, "end": 1, "speaker": "spk"}]
    output = await stage._process_impl("dummy.wav", "job", {"segments": diar_segments})

    # Проверяем, что слово получило спикера из диаризации
    assert output["words"][0]["word"] == "hi"
    assert output["segments"][0]["speaker"] == "spk"
    assert output["confidence"] == pytest.approx(0.8)
    # Убедимся, что список слов не пуст
    assert len(output["words"]) == 1
