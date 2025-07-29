# -*- coding: utf-8 -*-
"""
Unit-тесты для DiarizationStage с режимом скользящего окна
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import pytest
from unittest.mock import MagicMock
from pyannote.core import Annotation, Segment
from app.stages.diarization import DiarizationStage

@pytest.mark.asyncio
async def test_windowed_aggregation(monkeypatch):
    fake_pipeline = MagicMock()
    def crop_side_effect(context, segment):
        ann = Annotation()
        # возвращаем сегмент в координатах окна (1–2), не накапливая start
        ann[Segment(1.0, 2.0), "spk"] = True
        return ann
    fake_pipeline.crop.side_effect = crop_side_effect
    fake_pipeline.get_audio_duration.return_value = 50.0
    monkeypatch.setattr(
        "app.stages.diarization.Pipeline.from_pretrained",
        lambda *a, **k: fake_pipeline
    )

    cfg = {
        "model": "m",
        "device": "cpu",
        "window_size": 20.0,
        "hop_size": 10.0
    }
    stage = DiarizationStage(cfg, models_registry=None)
    await stage._initialize()
    out = await stage._process_impl("any.wav", "jid", {})
    # После агрегации остается единый сегмент от 1.0 до 42.0
    assert out["total_segments"] == 1
    assert out["total_speakers"] == 1
    seg = out["segments"][0]
    assert pytest.approx(1.0, rel=1e-3) == seg["start"]
    assert pytest.approx(42.0, rel=1e-3) == seg["end"]

@pytest.mark.asyncio
async def test_model_info_includes_window_params(monkeypatch):
    fake_pipeline = MagicMock()
    fake_pipeline.get_audio_duration.return_value = 5.0
    monkeypatch.setattr(
        "app.stages.diarization.Pipeline.from_pretrained",
        lambda *a, **k: fake_pipeline
    )

    cfg = {"model": "m", "device": "cpu", "window_size": 5.0, "hop_size": 2.5}
    stage = DiarizationStage(cfg, models_registry=None)
    await stage._initialize()
    info = stage._get_model_info()
    assert info["window_size"] == 5.0
    assert info["hop_size"] == 2.5
