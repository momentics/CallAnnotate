# -*- coding: utf-8 -*-
"""
Unit-тесты для этапа DiarizationStage

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import pytest
from unittest.mock import MagicMock

from app.stages.diarization import DiarizationStage
from app.stages.base import StageResult


@pytest.mark.asyncio
async def test_diarization_empty(monkeypatch):
    """Проверка, что при отсутствии сегментов возвращается пустой список и total_segments == 0"""
    # Мокаем Pipeline.from_pretrained и его вызов
    fake_pipeline = MagicMock()
    # при вызове fake_pipeline(file) вернёт Annotation с itertracks пустым
    fake_annotation = MagicMock()
    fake_annotation.itertracks.return_value = []
    monkeypatch.setattr(fake_pipeline, "__call__", lambda self, x: fake_annotation)
    monkeypatch.setattr("app.stages.diarization.Pipeline.from_pretrained",
                        lambda *a, **k: fake_pipeline)

    cfg = {"model": "test-model", "device": "cpu", "use_auth_token": None}
    stage = DiarizationStage(cfg, models_registry=None)
    await stage._initialize()

    # Выполняем тестовый прогресс-калбэк, проверяем payload
    result_payload = await stage._process_impl("file.wav", "job", {})
    assert result_payload["segments"] == []
    assert result_payload["total_segments"] == 0
    assert result_payload["total_speakers"] == 0


@pytest.mark.asyncio
async def test_process_returns_stage_result(monkeypatch):
    """Проверка, что метод process оборачивает _process_impl в StageResult"""
    fake_pipeline = MagicMock()
    fake_annotation = MagicMock()
    fake_annotation.itertracks.return_value = []
    monkeypatch.setattr("app.stages.diarization.Pipeline.from_pretrained",
                        lambda *a, **k: fake_pipeline)
    cfg = {"model": "m", "device": "cpu", "use_auth_token": None}
    stage = DiarizationStage(cfg, models_registry=None)

    # process вызывает initialize и process_impl
    res: StageResult = await stage.process("path.wav", "jobid")
    assert isinstance(res, StageResult)
    assert res.stage_name == "diarization"
    assert res.payload["segments"] == []
    assert res.payload["total_segments"] == 0
