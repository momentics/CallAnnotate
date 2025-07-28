import pytest
import torch
from unittest.mock import MagicMock
from app.stages.diarization import DiarizationStage

@pytest.mark.asyncio
async def test_diarization_empty(monkeypatch):
    # Мокаем Pipeline.from_pretrained и его вызов
    fake_pipe = lambda fp: MagicMock(itertracks=lambda yield_label: [])
    monkeypatch.setattr("app.stages.diarization.Pipeline.from_pretrained", lambda *a, **k: fake_pipe)
    cfg = {"model": "m", "device": "cpu"}
    stage = DiarizationStage(cfg, MagicMock())
    await stage._initialize()
    out = await stage._process_impl("file.wav", "job", {})
    assert out["segments"] == []
    assert out["total_segments"] == 0
