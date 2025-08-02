# -*- coding: utf-8 -*-
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import pathlib
import yaml


from app.config import (
    DiarizationConfig,
    TranscriptionConfig,
    load_settings,
    RecognitionConfig,
    load_settings_from_yaml,
)

def test_recognition_validator(tmp_path):
    """
    Убедиться, что модель валидатор создаёт директорию embeddings_path.
    """
    emb = tmp_path / "emb"
    cfg = RecognitionConfig(embeddings_path=str(emb))
    # Директория должна быть создана валидатором
    assert cfg.embeddings_path is not None
    assert pathlib.Path(cfg.embeddings_path).exists()
    
def write_yaml(tmp_path, data):
    f = tmp_path / "cfg.yaml"
    f.write_text(yaml.safe_dump(data))
    return str(f)

def test_load_basic_yaml(tmp_path, monkeypatch):
    data = {
        "diarization": {"model": "m", "device": "cpu"},
        "transcription": {"model": "t", "device": "cpu"},
        "recognition": {"model": "r", "device": "cpu"},
        "carddav": {"enabled": False},
        "queue": {"max_concurrent_tasks": 1, "max_queue_size": 10},
        "server": {"host": "127.0.0.1", "port": 9000},
        "files": {"max_size": 123},
        "logging": {"level": "DEBUG"},
        "cors": {"origins": ["https://a"]},
        "voices": [],
        "notifications": {},
        "security": {},
        "monitoring": {},
        "features": {},
    }
    path = write_yaml(tmp_path, data)
    settings = load_settings_from_yaml(path)
    # Server settings
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 9000
    # Logging settings
    assert settings.logging.level == "DEBUG"
    # Components types
    assert isinstance(settings.diarization, DiarizationConfig)
    assert settings.diarization.model == "m"
    assert isinstance(settings.transcription, TranscriptionConfig)
    assert settings.transcription.model == "t"
    assert isinstance(settings.recognition, RecognitionConfig)
    assert settings.recognition.model == "r"
    # CardDAV toggle
    assert settings.carddav.enabled is False
    # Queue settings
    assert settings.queue.max_concurrent_tasks == 1
    assert settings.queue.max_queue_size == 10
    # Files settings
    assert settings.files.max_size == 123
    # CORS settings
    assert settings.cors.origins == ["https://a"]
    # Defaults for others
    assert settings.notifications.webhooks.enabled is True
    assert settings.security.rate_limiting.enabled is True
    assert settings.features.batch_processing is True

def test_load_settings_env_override(monkeypatch):
    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
    s = load_settings(config_path="nonexistent.yaml")
    assert s.server.host == "0.0.0.0"

