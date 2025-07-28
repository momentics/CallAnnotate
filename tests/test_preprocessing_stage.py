# tests/test_preprocessing_stage.py

import pytest
from pathlib import Path
from unittest.mock import patch
from app.stages.preprocessing import PreprocessingStage
from app.config import PreprocessingConfig

@pytest.mark.asyncio
async def test_preprocessing_normalization(tmp_path, monkeypatch):
    # создаём файл с синусоидой
    sr = 8000
    t = 1.0
    import numpy as np
    import soundfile as sf
    samples = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, t, int(sr*t)))).astype(np.float32)
    wav = tmp_path / "in.wav"
    sf.write(str(wav), samples, sr)

    # конфигурация
    cfg = PreprocessingConfig(chunk_duration=1.0, overlap=0.0, target_rms=-10.0)
    stage = PreprocessingStage(cfg.dict(), None)

    # мокаем df.enhance модуль целиком
    class MockDF:
        @staticmethod
        def init_df(**kwargs):
            return None, None, None
        
        @staticmethod
        def enhance(model, state, samples):
            return samples

    monkeypatch.setattr("app.stages.preprocessing.init_df", MockDF.init_df)
    monkeypatch.setattr("app.stages.preprocessing.enhance", MockDF.enhance)

    # инициализация должна пройти успешно
    await stage._initialize()
    
    result = await stage._process_impl(str(wav), "jid", {})
    assert "processed_path" in result
    assert Path(result["processed_path"]).exists()

@pytest.mark.asyncio
async def test_preprocessing_failure(monkeypatch):
    # Эмулируем отсутствие deepfilternet (init_df, enhance = None)
    monkeypatch.setattr("app.stages.preprocessing.init_df", None)
    monkeypatch.setattr("app.stages.preprocessing.enhance", None)
    cfg = PreprocessingConfig()
    stage = PreprocessingStage(cfg.dict(), None)
    with pytest.raises(RuntimeError, match="Не установлена зависимость deepfilternet"):
        await stage._initialize()
