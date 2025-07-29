# tests/test_preprocessing_rnnoise.py
# -*- coding: utf-8 -*-
"""
Тесты для проверки работы RNNoise-шумоподавления в PreprocessingStage
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import pytest
import numpy as np
import soundfile as sf
import os
import tempfile

from app.stages.preprocessing import PreprocessingStage
from app.config import PreprocessingConfig

class DummyRNNoise:
    def __init__(self):
        pass

    def filter(self, segment):
        # искусственно добавляем уменьшение RMS на 50%
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples *= 0.5
        return segment._spawn(samples.astype(segment.array_type))

@pytest.mark.asyncio
async def test_rnnoise_reduction(tmp_path, monkeypatch):
    # создаём WAV с шумом: синус + белый шум
    sr = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), False)
    sine = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = 0.1 * np.random.RandomState(0).randn(len(t)).astype(np.float32)
    samples = sine + noise
    wav_path = tmp_path / "noisy.wav"
    sf.write(str(wav_path), samples, sr)

    # конфиг с одним чанком без overlap
    cfg = PreprocessingConfig(chunk_duration=1.0, overlap=0.0, target_rms=-20.0)
    stage = PreprocessingStage(cfg.dict(), models_registry=None)

    # мокаем init_df/enhance, а RNNoise
    monkeypatch.setattr("app.stages.preprocessing.init_df", lambda **kwargs: (None, None, None))
    monkeypatch.setattr("app.stages.preprocessing.enhance", lambda model, state, data: data)
    monkeypatch.setattr("app.stages.preprocessing.RNNoise", lambda: DummyRNNoise())

    await stage._initialize()
    result = await stage._process_impl(str(wav_path), "testid", {})
    out_path = result["processed_path"]
    assert os.path.exists(out_path)

    # проверяем RMS до и после подавления на выходном файле sox-шаг пропускаем,
    # но RNNoise снизило шум на ~50%
    orig_rms = np.sqrt(np.mean(samples**2))
    processed, _ = sf.read(out_path)
    proc_rms = np.sqrt(np.mean(processed**2))
    assert proc_rms < orig_rms * 0.6  # с учётом других этапов
