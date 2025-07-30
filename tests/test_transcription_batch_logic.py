import pytest
from app.stages.transcription import TranscriptionStage
from app.config import AppSettings
from unittest.mock import MagicMock

@pytest.mark.asyncio
async def test_batch_logic(monkeypatch, tmp_path):
    # мокаем whisper.load_model и модель
    fake_model = MagicMock(transcribe=lambda batch, **opts: {
        "segments": [{"start":0,"end":1,"text":"hi","no_speech_prob":0.1,"avg_logprob":-0.2,"words":[{"start":0,"end":1,"word":"hi","probability":0.9}]}],
        "confidence":0.8
    })
    monkeypatch.setattr("whisper.load_model", lambda size, device=None: fake_model)

    cfg = AppSettings().transcription.dict()
    cfg.update({"batch_size": 10})
    stage = TranscriptionStage(cfg, models_registry=None)
    await stage._initialize()
    result = await stage._process_impl("audio.wav", "jid", {"segments":[{"start":0,"end":1,"speaker":"spk"}]})

    assert result["confidence"] == pytest.approx(0.8)
    assert isinstance(result["avg_logprob"], float)
    assert isinstance(result["no_speech_prob"], float)
    assert "processing_time" in result
    assert result["segments"][0]["speaker"] == "spk"

