# -*- coding: utf-8 -*-
"""
Расширенные unit-тесты для этапа DiarizationStage (src/app/stages/diarization.py)

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import pytest
from unittest.mock import MagicMock

from app.stages.diarization import DiarizationStage
from app.stages.base import StageResult


@pytest.mark.asyncio
async def test_multiple_speakers(monkeypatch):
    """Проверка корректной обработки нескольких сегментов и спикеров"""
    # Мокируем Annotation.itertracks
    class Segment:
        def __init__(self, start, end, speaker, confidence=0.8):
            self.start = start
            self.end = end
            self.confidence = confidence

    fake_annotation = MagicMock()
    fake_annotation.itertracks.return_value = [
        (Segment(0.0, 1.0, 'spk1'), None, 'spk1'),
        (Segment(0.5, 2.0, 'spk2'), None, 'spk2'),
        (Segment(2.0, 3.0, 'spk1'), None, 'spk1'),
    ]
    fake_pipeline = MagicMock(return_value=fake_annotation)
    monkeypatch.setattr("app.stages.diarization.Pipeline.from_pretrained",
                        lambda *args, **kwargs: fake_pipeline)

    cfg = {"model": "m", "device": "cpu", "use_auth_token": None}
    stage = DiarizationStage(cfg, models_registry=None)
    await stage._initialize()
    payload = await stage._process_impl("file.wav", "jid", {})
    # три сегмента, два спикера
    assert payload["total_segments"] == 3
    assert payload["total_speakers"] == 2
    # speakers list содержит spk1 и spk2
    assert set(payload["speakers"]) == {"spk1", "spk2"}
    # сегменты отсортированы по start
    starts = [seg["start"] for seg in payload["segments"]]
    assert starts == sorted(starts)


@pytest.mark.asyncio
async def test_pipeline_error(monkeypatch):
    """Проверка обработки исключения внутри pipeline"""
    fake_pipeline = MagicMock(side_effect=RuntimeError("fail"))
    monkeypatch.setattr("app.stages.diarization.Pipeline.from_pretrained",
                        lambda *a, **k: fake_pipeline)

    cfg = {"model": "m", "device": "cpu", "use_auth_token": None}
    stage = DiarizationStage(cfg, models_registry=None)
    res: StageResult = await stage.process("file.wav", "jid")
    assert not res.success
    assert "fail" in res.error.lower()
