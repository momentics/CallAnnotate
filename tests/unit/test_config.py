import os
import tempfile
import shutil
import yaml
import pytest
from pathlib import Path

from app.config import load_settings, load_settings_from_yaml, AppSettings


@pytest.fixture
def tmp_config_file(tmp_path):
    # Создаём временный каталог и копируем в него оригинальный default.yaml
    project_root = Path(__file__).parents[2]
    default_yaml = project_root / "config" / "default.yaml"
    dst = tmp_path / "default.yaml"
    shutil.copy(default_yaml, dst)
    return dst


def test_load_settings_from_default(tmp_config_file, monkeypatch):
    # Гарантируем, что без указания переменных окружения берётся default.yaml
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.setenv("CONFIG_PATH", str(tmp_config_file))
    settings = load_settings()
    # Проверяем, что поля совпадают с теми, что в default.yaml
    with open(tmp_config_file, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    assert settings.server.host == cfg_dict["server"]["host"]
    assert settings.server.port == cfg_dict["server"]["port"]
    assert settings.queue.volume_path == cfg_dict["queue"]["volume_path"]
    assert settings.preprocess.chunk_duration == cfg_dict["preprocess"]["chunk_duration"]
    # В отсутствие override из окружения должны остаться значения из YAML
    assert settings.security.rate_limiting.requests_per_minute == cfg_dict["security"]["rate_limiting"]["requests_per_minute"]


def test_load_settings_from_yaml_function(tmp_config_file):
    # Проверяем прямой вызов load_settings_from_yaml
    settings: AppSettings = load_settings_from_yaml(str(tmp_config_file))
    data = yaml.safe_load(tmp_config_file.read_text(encoding="utf-8"))
    # Несколько ключевых проверок
    assert settings.files.max_size == data["files"]["max_size"]
    assert settings.logging.level.lower() == data["logging"]["level"].lower()
    assert settings.diarization.model == data["diarization"]["model"]


def test_override_via_env(monkeypatch, tmp_config_file):
    # Проверяем, что можно указать CONFIG_PATH через окружение
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.setenv("CONFIG_PATH", str(tmp_config_file))
    # Подменим один параметр в YAML
    text = tmp_config_file.read_text(encoding="utf-8")
    text = text.replace("workers: 1", "workers: 5")
    tmp_config_file.write_text(text, encoding="utf-8")
    settings = load_settings()
    assert settings.server.workers == 5


def test_fallback_to_defaults_if_no_yaml(monkeypatch):
    # Если CONFIG_PATH указывает несуществующий файл, load_settings возвращает дефолтные настройки
    monkeypatch.setenv("CONFIG_PATH", "/nonexistent/path.yaml")
    settings = load_settings()
    # Значения из AppSettings().model_dump()
    default = AppSettings()
    assert settings.server.host == default.server.host
    assert settings.files.max_size == default.files.max_size
    assert settings.preprocess.model == default.preprocess.model
