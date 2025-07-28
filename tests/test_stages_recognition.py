import pytest
import numpy as np
from unittest.mock import MagicMock
from app.stages.recognition import RecognitionStage

@pytest.mark.asyncio
async def test_recognition_no_db(monkeypatch, tmp_path):
    # Инициализация без базы
    cfg = {"model": "m", "device": "cpu", "embeddings_path": None, "threshold": 0.5}
    stage = RecognitionStage(cfg, MagicMock())
    await stage._initialize()
    out = await stage._process_impl("file", "job", {"segments":[{"speaker":"s","duration":2.0,"start":0,"end":2}]})
    assert out["speakers"]["s"]["reason"].startswith("База голосов")
