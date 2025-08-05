# tests/unit/stages/test_recognition_stage.py
import pytest
from unittest.mock import Mock, patch
import numpy as np

from app.stages.recognition import RecognitionStage


@pytest.fixture(autouse=True)
def disable_setup_logging(monkeypatch):
    # Disable setup_logging to avoid filesystem interactions
    monkeypatch.setattr("app.utils.setup_logging", lambda cfg: None)


class TestRecognitionStage:
    @pytest.fixture
    def config(self, tmp_path):
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()
        return {
            "model": "speechbrain/spkrec-ecapa-voxceleb",
            "device": "cpu",
            "threshold": 0.7,
            "embeddings_path": str(emb_dir),
        }

    @pytest.fixture
    def stage(self, config):
        app_config = Mock()
        app_config.queue = Mock()
        app_config.queue.volume_path = "/tmp/test"
        return RecognitionStage(app_config, config, Mock())

    @pytest.mark.asyncio
    async def test_initialize_without_embeddings_path(self, stage):
        """Проверяет инициализацию без пути к эмбеддингам"""
        stage.config["embeddings_path"] = None

        await stage._initialize()

        assert stage.classifier is None
        assert stage.index is None

    @pytest.mark.asyncio
    async def test_initialize_with_embeddings_and_faiss(self, stage, tmp_path):
        """Проверяет инициализацию с существующей базой эмбеддингов и FAISS"""
        # prepare fake embedding files
        emb_path = tmp_path / "emb"
        emb_path.mkdir()
        vec1 = emb_path / "sp1.vec"
        vec2 = emb_path / "sp2.vec"
        np.savetxt(vec1, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        np.savetxt(vec2, np.array([0.4, 0.5, 0.6], dtype=np.float32))

        stage.config["embeddings_path"] = str(emb_path)

        with patch("app.stages.recognition.faiss") as mock_faiss:
            # stub FAISS IndexFlatIP and normalize_L2
            mock_index = Mock()
            mock_faiss.IndexFlatIP.return_value = mock_index
            mock_faiss.normalize_L2 = lambda x: None

            await stage._initialize()

            # Ensure two embeddings were indexed independent of ordering
            assert len(stage.speaker_labels) == 2
            assert set(stage.speaker_labels.values()) == {"sp1", "sp2"}
            mock_index.add.assert_called_once()

    def test_match_speaker_above_threshold(self, stage):
        """Проверяет распознавание спикера выше порога"""
        stage.index = Mock()
        stage.index.search.return_value = (
            np.array([[0.8]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        stage.speaker_labels = {0: "john_doe"}
        stage.threshold = 0.7

        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = stage._match_speaker(embedding)

        assert result["identified"] is True
        assert result["name"] == "john_doe"
        assert result["confidence"] == 0.8

    def test_match_speaker_below_threshold(self, stage):
        """Проверяет случай когда similarity ниже порога"""
        stage.index = Mock()
        stage.index.search.return_value = (
            np.array([[0.6]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        stage.speaker_labels = {0: "john_doe"}
        stage.threshold = 0.7

        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = stage._match_speaker(embedding)

        assert result["identified"] is False
        assert result["name"] is None
        assert result["confidence"] == 0.6
