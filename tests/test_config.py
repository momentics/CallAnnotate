# -*- coding: utf-8 -*-
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

from fastapi import Path
import yaml

from app.config import (
    DiarizationConfig,
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
    # Проверяем, что значение действительно было создано и путь корректен
    assert cfg.embeddings_path is not None
    assert Path(cfg.embeddings_path).exists()

    
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
        "performance": {},
        "security": {},
        "monitoring": {},
        "features": {},
    }
    path = write_yaml(tmp_path, data)
    settings = load_settings_from_yaml(path)
    assert settings.server.port == 9000
    assert settings.logging.level == "DEBUG"
    assert isinstance(settings.diarization, DiarizationConfig)

def test_load_settings_env_override(monkeypatch):
    monkeypatch.setenv("SERVER_HOST", "0.0.0.0")
    s = load_settings(config_path="nonexistent.yaml")
    assert s.server.host == "0.0.0.0"

