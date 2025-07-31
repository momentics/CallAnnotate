# tests/test_diarization_stage_filters.py
# -*- coding: utf-8 -*-
"""
Дополнительные тесты для повышения покрытия src/app/stages/diarization.py.

Проверяются:
1. Логика post-processing при указании min_speakers / max_speakers.
2. Автоматическое включение window_enabled, если задан window_size (или hop_size).
3. Корректный выброс исключения при некорректных окнах (hop_size > window_size).

Автор: CallAnnotate tests team
Лицензия: Apache-2.0
"""

import pytest
from app.stages.diarization import DiarizationStage, DiarizationCfg


# ---------------------------------------------------------------------------
#  1. Проверка фильтрации спикеров по min_speakers / max_speakers
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_postprocess_min_max_filters():
    """
    Если указаны min_speakers и max_speakers, _postprocess должен
    отфильтровать лишних спикеров, оставив не менее min_ и не более max_.
    """
    # 4 спикера, разная длительность речи
    speaker_stats = {
        "s1": {"total_duration": 10.0, "segments_count": 1},
        "s2": {"total_duration": 8.0, "segments_count": 1},
        "s3": {"total_duration": 6.0, "segments_count": 1},
        "s4": {"total_duration": 4.0, "segments_count": 1},
    }
    segments = [
        {"start": 0.0, "end": 10.0, "duration": 10.0, "speaker": "s1", "confidence": 0.9},
        {"start": 10.0, "end": 18.0, "duration": 8.0, "speaker": "s2", "confidence": 0.8},
        {"start": 18.0, "end": 24.0, "duration": 6.0, "speaker": "s3", "confidence": 0.7},
        {"start": 24.0, "end": 28.0, "duration": 4.0, "speaker": "s4", "confidence": 0.6},
    ]
    payload = {
        "segments": segments,
        "speakers": list(speaker_stats.keys()),
        "speaker_stats": speaker_stats,
        "total_segments": len(segments),
        "total_speakers": len(speaker_stats),
    }

    cfg = {
        "model": "dummy-model",
        "device": "cpu",
        "batch_size": 8,
        "min_speakers": 2,
        "max_speakers": 3,
    }
    stage = DiarizationStage(cfg, models_registry=None)

    # _postprocess изменяет payload in-place и возвращает его
    result = stage._postprocess(payload)

    assert result["total_speakers"] == 3
    assert set(result["speakers"]) == {"s1", "s2", "s3"}
    # Все сегменты должны относиться только к оставшимся спикерам
    assert all(seg["speaker"] in {"s1", "s2", "s3"} for seg in result["segments"])


# ---------------------------------------------------------------------------
#  2. Автоматическое включение оконного режима
# ---------------------------------------------------------------------------
def test_window_auto_enable():
    """
    При явном задании window_size / hop_size режим window_enabled
    должен автоматически активироваться, даже если в конфиге False.
    """
    data = {
        "model": "m",
        "device": "cpu",
        "batch_size": 16,
        "window_enabled": False,
        "window_size": 25.0,   # наличие параметра должно включить оконный режим
        "hop_size": 10.0,
    }
    cfg = DiarizationCfg.from_dict(data)
    assert cfg.window_enabled is True
    assert cfg.window_size == 25.0
    assert cfg.hop_size == 10.0


# ---------------------------------------------------------------------------
#  3. Неверное отношение hop_size / window_size вызывает ValueError
# ---------------------------------------------------------------------------
def test_invalid_window_params():
    """
    Если hop_size больше window_size, должен выбрасываться ValueError.
    """
    bad_cfg = {
        "model": "m",
        "device": "cpu",
        "window_enabled": True,
        "window_size": 5.0,
        "hop_size": 10.0,  # больше, чем окно
    }
    with pytest.raises(ValueError):
        _ = DiarizationCfg.from_dict(bad_cfg)
