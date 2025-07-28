import pytest
import whisper
from unittest.mock import MagicMock
from app.stages.transcription import TranscriptionStage

@pytest.mark.asyncio
async def test_transcription_align(monkeypatch):
    # Мокаем whisper.load_model и transcribe
    fake_model = MagicMock(transcribe=lambda fp, **opts: {
        "segments":[{"start":0,"end":1,"text":"hi","words":[{"start":0,"end":1,"word":"hi","probability":0.9}]}],
        "language":"en","text":"hi"
    })
    monkeypatch.setattr(whisper, "load_model", lambda m, device=None: fake_model)
    cfg = {"model":"whisper-base","device":"cpu","language":"auto","task":"transcribe"}
    stage = TranscriptionStage(cfg, MagicMock())
    await stage._initialize()
    out = await stage._process_impl("file", "job", {"segments":[{"start":0,"end":1,"speaker":"spk"}]})
    assert out["words"][0]["word"] == "hi"
    assert out["segments"][0]["speaker"] == "spk"
