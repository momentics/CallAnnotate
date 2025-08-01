# tests/test_stages_transcription_extended.py

import pytest
import whisper
import numpy as np
from unittest.mock import MagicMock

from app.stages.transcription import TranscriptionStage

@pytest.mark.asyncio
async def test_empty_audio_file(monkeypatch):
    # Мокаем whisper.load_model для транскрипции
    fake_model = MagicMock(
        transcribe=lambda batch, **opts: {"segments": [], "confidence": 0.0}
    )
    monkeypatch.setattr(whisper, "load_model", lambda m, device=None: fake_model)

    cfg = {
        "model": "whisper-base",
        "device": "cpu",
        "language": "ru",
        "task": "transcribe",
        "batch_size": 4,
        "min_segment_duration": 0.2,
        "max_silence_between": 0.1,
        "min_overlap": 0.3,
    }
    stage = TranscriptionStage(cfg, MagicMock())
    await stage._initialize()

    result = await stage._process_impl("corrupt.wav", "jid_empty", {"segments": []})
    # При пустом входе должны быть пустые списки
    assert result["segments"] == []
    assert result["words"] == []
    assert result["confidence"] == 0.0

@pytest.mark.asyncio
async def test_min_duration_filter(monkeypatch):
    # Мокаем модель с двумя сегментами разной длины
    fake_model = MagicMock(
        transcribe=lambda batch, **opts: {
            "segments": [
                {"start": 0, "end": 0.1, "text": "a", "no_speech_prob": 0.0,
                 "avg_logprob": -0.1,
                 "words": [{"start": 0, "end": 0.1, "word": "a", "probability": 0.5}]},
                {"start": 0.2, "end": 0.5, "text": "b", "no_speech_prob": 0.0,
                 "avg_logprob": -0.2,
                 "words": [{"start": 0.2, "end": 0.5, "word": "b", "probability": 0.6}]}
            ],
            "confidence": 0.7
        }
    )
    monkeypatch.setattr(whisper, "load_model", lambda m, device=None: fake_model)

    cfg = {
        "model": "whisper-base",
        "device": "cpu",
        "language": "ru",
        "task": "transcribe",
        "batch_size": 2,
        "min_segment_duration": 0.299,
        "max_silence_between": 0.1,
        "min_overlap": 0.3,
    }
    stage = TranscriptionStage(cfg, MagicMock())
    await stage._initialize()

    diar = [{"start": 0, "end": 1, "speaker": "spk"}]
    res = await stage._process_impl("dummy.wav", "jid_min", {"segments": diar})

    # Останется только сегмент длиной >= 0.299
    segments = res["segments"]
    assert len(segments) == 1
    seg = segments[0]
    seg_dur = seg["end"] - seg["start"]
    assert seg_dur + 1e-3 >= 0.299
    assert seg["text"] == "b"
