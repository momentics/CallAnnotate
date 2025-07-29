# -*- coding: utf-8 -*-
"""
Unit-тесты для проверок крайних случаев в PreprocessingStage (src/app/stages/preprocessing.py)

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import subprocess
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.stages.preprocessing import PreprocessingStage, init_df, enhance
from app.config import PreprocessingConfig


@pytest.mark.asyncio
async def test_missing_deepfilternet(monkeypatch):
    """Без init_df/enhance этап инициализации падает"""
    monkeypatch.setattr("app.stages.preprocessing.init_df", None)
    monkeypatch.setattr("app.stages.preprocessing.enhance", None)
    cfg = PreprocessingConfig().dict()
    stage = PreprocessingStage(cfg, None)
    with pytest.raises(RuntimeError):
        await stage._initialize()


@pytest.mark.asyncio
async def test_sox_failure(monkeypatch, tmp_path):
    """Если SoX бросает CalledProcessError, используем оригинальный файл"""
    # создаём dummy файл
    wav = tmp_path / "f.wav"
    wav.write_bytes(b"1234")
    # инициализация DeepFilterNet и RNNoise
    monkeypatch.setattr("app.stages.preprocessing.init_df", lambda **kw: (None, None, None))
    monkeypatch.setattr("app.stages.preprocessing.enhance", lambda m, s, d: d)
    # форсим sox_available=True
    monkeypatch.setenv("SOX_SKIP", "0")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/sox")
    # subprocess.run будет падать
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, a[0])))
    cfg = PreprocessingConfig().dict()
    stage = PreprocessingStage(cfg, None)
    await stage._initialize()
    result = await stage._apply_sox(str(wav), "jid", target_rms_db=-20.0)
    # при ошибке возвращается путь оригинального файла
    assert result == str(wav)


@pytest.mark.asyncio
async def test_rnnoise_skip(monkeypatch, tmp_path):
    """Если RNNoise падает на этапе фильтрации, возвращаем оригинальный AudioSegment"""
    from pydub import AudioSegment
    # создаём WAV
    sr = 8000
    samples = (0.01 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))).astype(np.float32)
    wav_path = tmp_path / "t.wav"
    AudioSegment.from_file = lambda p: AudioSegment(
        samples.tobytes(), frame_rate=sr, sample_width=4, channels=1
    )
    # DeepFilterNet OK
    monkeypatch.setattr("app.stages.preprocessing.init_df", lambda **kw: (None, None, None))
    monkeypatch.setattr("app.stages.preprocessing.enhance", lambda m, s, d: d)
    # RNNoise.__init__ работает, но filter падает
    class RN:
        def __init__(self): pass
        def filter(self, seg): raise Exception("rn fail")
    monkeypatch.setattr("app.stages.preprocessing.RNNoise", RN)
    cfg = PreprocessingConfig(rnnoise_enabled=True).dict()
    stage = PreprocessingStage(cfg, None)
    await stage._initialize()
    # вызов _apply_rnnoise не должен упасть
    seg = AudioSegment.from_file(str(wav_path))
    out = await stage._apply_rnnoise(seg, idx=0)
    assert isinstance(out, AudioSegment)


@pytest.mark.asyncio
async def test_merge_chunks_methods():
    """Проверка склейки чанков и оконного метода"""
    cfg = PreprocessingConfig().dict()
    stage = PreprocessingStage(cfg, None)
    # два массива длиной 100, overlap 20
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32) * 2
    merged_linear = await stage._merge_chunks([a.copy(), b.copy()], overlap_ms=20, sample_rate=1000, method="linear")
    # без windowed первые 80 из b добавляются
    assert merged_linear.shape[0] == 100 + (100 - 20)
    merged_windowed = await stage._merge_chunks([a.copy(), b.copy()], overlap_ms=20, sample_rate=1000, method="windowed")
    assert merged_windowed.shape[0] == merged_linear.shape[0]


@pytest.mark.asyncio
async def test_final_normalization_zero_audio():
    """Если RMS == 0, возвращаем без изменений"""
    cfg = PreprocessingConfig().dict()
    stage = PreprocessingStage(cfg, None)
    silent = np.zeros(100, dtype=np.float32)
    out = await stage._apply_final_normalization(silent, target_rms_db=-10.0)
    assert out is silent
