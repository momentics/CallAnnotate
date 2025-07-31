# tests/test_preprocessing_chunk_overlap.py
# -*- coding: utf-8 -*-
"""
Тесты для проверки корректности чанковой обработки с перекрытием
и итоговой длительности без искажений
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

@pytest.mark.asyncio
async def test_chunking_overlap_duration(tmp_path, monkeypatch):
    # создаём синусоиду длительностью 3 секунды
    sr = 8000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), False)
    samples = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "tone.wav"
    sf.write(str(wav_path), samples, sr)

    # конфиг: chunk 2s, overlap 1s => ожидаемые чанки [0–2], [1–3]
    cfg = PreprocessingConfig(deepfilter_enabled=True, 
                              rnnoise_enabled=True, 
                              sox_enabled=False,
                              chunk_duration=2.0, 
                              overlap=1.0,
                              chunk_overlap_method="linear",
                              target_rms=-20.0)
    stage = PreprocessingStage(cfg.dict(), models_registry=None)

    # мокаем все этапы: sox, RNNoise, DeepFilterNet2
    monkeypatch.setattr("app.stages.preprocessing.init_df", lambda **kwargs: (None, None, None))
    monkeypatch.setattr("app.stages.preprocessing.enhance", lambda model, state, data: data)
    # мокаем RNNoise.filter
    class NoOp:
        def filter(self, seg): return seg
    monkeypatch.setattr("app.stages.preprocessing.RNNoise", lambda: NoOp())

    await stage._initialize()
    result = await stage._process_impl(str(wav_path), "testid2", {})
    out_path = result["processed_path"]
    assert os.path.exists(out_path)

    # проверяем длительность выходного файла ≈ 3s
    processed, _ = sf.read(out_path)
    out_dur = len(processed) / sr
    assert pytest.approx(duration, rel=1e-3) == out_dur

    # проверяем, что форма сигнала не исказилась: сравнение RMS
    orig_rms = np.sqrt(np.mean(samples**2))
    proc_rms = np.sqrt(np.mean(processed**2))
    assert pytest.approx(orig_rms, rel=0.05) == proc_rms
