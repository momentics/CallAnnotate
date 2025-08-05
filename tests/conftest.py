# tests/conftest.py

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from app.config import AppSettings
import warnings

# Suppress known warnings during tests
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.custom_fwd")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@pytest.fixture
def temp_volume():
    """Создаёт временную директорию volume для тестов"""
    with tempfile.TemporaryDirectory() as tmpdir:
        volume = Path(tmpdir)
        (volume / "incoming").mkdir()
        (volume / "processing").mkdir()
        (volume / "completed").mkdir()
        (volume / "failed").mkdir()
        (volume / "models" / "embeddings").mkdir(parents=True)
        yield str(volume)

@pytest.fixture
def mock_models_registry():
    """Мок реестра моделей"""
    registry = Mock()
    registry.get_model.return_value = Mock()
    return registry

@pytest.fixture
def app_config(temp_volume):
    """Базовая конфигурация приложения для тестов"""
    config = AppSettings()
    config.queue.volume_path = temp_volume
    return config
