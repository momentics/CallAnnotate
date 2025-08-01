import numpy as np
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.stages.recognition import RecognitionStage


class DummyClassifier:
    """Заглушка для EncoderClassifier."""
    def __init__(self, embedding_size=192):
        self.embedding_size = embedding_size

    def encode_batch(self, tensor):
        # возвращаем вектор единиц нужного размера
        batch_size = tensor.shape[0]
        return torch.ones(batch_size, self.embedding_size)


@pytest.mark.asyncio
async def test_initialize_without_database(tmp_path):
    """Инициализация без папки эмбеддингов."""
    cfg = {
        "model": "dummy-model",
        "device": "cpu",
        "embeddings_path": None,
        "threshold": 0.5
    }
    stage = RecognitionStage(cfg, models_registry=MagicMock())
    # При отсутствии каталога эмбеддингов индекс должен быть None
    await stage._initialize()
    assert stage.index is None
    assert stage.speaker_labels == {}


@pytest.mark.asyncio
async def test_initialize_with_database(tmp_path, monkeypatch):
    """Инициализация с папкой эмбеддингов (.vec)."""
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    # создаём два файла эмбеддингов спикеров
    vec1 = emb_dir / "alice.vec"
    np.savetxt(str(vec1), np.array([[0.1, 0.2, 0.3]]))
    vec2 = emb_dir / "bob.vec"
    np.savetxt(str(vec2), np.array([[0.4, 0.5, 0.6]]))

    cfg = {
        "model": "dummy-model",
        "device": "cpu",
        "embeddings_path": str(emb_dir),
        "threshold": 0.7
    }

    # Подменяем EncoderClassifier и faiss.IndexFlatIP
    monkeypatch.setattr(
        "app.stages.recognition.EncoderClassifier.from_hparams",
        lambda source, run_opts: DummyClassifier(3)
    )
    fake_index = MagicMock()
    fake_index.ntotal = 2
    fake_index.search.return_value = (np.array([[0.8]]), np.array([[1]]))
    monkeypatch.setattr(
        "app.stages.recognition.faiss.IndexFlatIP",
        lambda dim: fake_index
    )

    stage = RecognitionStage(cfg, models_registry=MagicMock())
    await stage._initialize()

    # Проверяем загрузку эмбеддингов и заполнение speaker_labels
    assert stage.index is fake_index
    assert stage.speaker_labels[0] == "alice"
    assert stage.speaker_labels[1] == "bob"

    # Проверяем метод _get_model_info
    info = stage._get_model_info()
    assert info["stage"] == "recognition"
    assert info["threshold"] == 0.7
    assert info["db_size"] == 2
    assert info["framework"] == "SpeechBrain + FAISS"


@pytest.mark.asyncio
async def test_process_impl_no_segments(tmp_path, monkeypatch):
    """Если нет сегментов диаризации, возвращается пустой результат."""
    cfg = {
        "model": "dummy-model",
        "device": "cpu",
        "embeddings_path": None,
        "threshold": 0.6
    }
    stage = RecognitionStage(cfg, models_registry=MagicMock())
    await stage._initialize()
    payload = await stage._process_impl(
        file_path="dummy.wav",
        task_id="tid",
        previous_results={"segments": []}
    )
    # Переименованные ключи:
    assert payload["speakers"] == {}
    assert payload.get("processed", 0) == 0
    assert payload.get("identified", 0) == 0


@pytest.mark.asyncio
async def test_process_impl_with_segments(tmp_path, monkeypatch):
    """Распознавание спикеров с помощью FAISS-индекса."""
    # создаём временный каталог эмбеддингов
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    vec = emb_dir / "spk1.vec"
    np.savetxt(str(vec), np.array([[1.0, 0.0, 0.0]]))

    cfg = {
        "model": "dummy-model",
        "device": "cpu",
        "embeddings_path": str(emb_dir),
        "threshold": 0.5,
        "index_path": None
    }

    # Подменяем EncoderClassifier и faiss.IndexFlatIP
    monkeypatch.setattr(
        "app.stages.recognition.EncoderClassifier.from_hparams",
        lambda source, run_opts: DummyClassifier(3)
    )
    fake_index = MagicMock()
    fake_index.ntotal = 1
    # возвращаем близкий сосед с индексом 0
    fake_index.search.return_value = (np.array([[0.8]]), np.array([[0]]))
    monkeypatch.setattr(
        "app.stages.recognition.faiss.IndexFlatIP",
        lambda dim: fake_index
    )

    stage = RecognitionStage(cfg, models_registry=MagicMock())
    await stage._initialize()

    # готовим тестовые сегменты
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "spk1"},
        {"start": 1.0, "end": 2.0, "speaker": "spk2"},
    ]
    payload = await stage._process_impl(
        file_path="dummy.wav",
        task_id="tid",
        previous_results={"segments": segments}
    )

    # первый сегмент идентифицируется
    assert "spk1" in payload["speakers"]
    result1 = payload["speakers"]["spk1"]
    assert result1.identified is True
    assert result1.name == "spk1"
    assert result1.confidence == pytest.approx(0.8, rel=1e-3)

    # второй сегмент остаётся неизвестным
    assert "spk2" in payload["speakers"]
    result2 = payload["speakers"]["spk2"]
    assert result2.identified is False
    assert result2.name is None
