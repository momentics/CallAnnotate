# -*- coding: utf-8 -*-
"""
Конфигурация приложения CallAnnotate с использованием Pydantic

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from pydantic import BaseSettings, Field, validator
import yaml


class DiarizationConfig(BaseSettings):
    """Конфигурация этапа диаризации"""
    model_name: str = Field("pyannote/speaker-diarization-3.1", description="Модель pyannote.audio")
    auth_token: Optional[str] = Field(None, description="HuggingFace токен")
    device: str = Field("cpu", description="Устройство для вычислений")
    
    class Config:
        env_prefix = "DIARIZATION_"


class TranscriptionConfig(BaseSettings):
    """Конфигурация этапа транскрипции"""
    model_size: str = Field("base", description="Размер модели Whisper")
    device: str = Field("cpu", description="Устройство для вычислений")
    options: Dict[str, Any] = Field(
        default_factory=lambda: {
            "language": None,
            "task": "transcribe",
            "temperature": 0.0,
            "best_of": 5,
            "beam_size": 5
        },
        description="Опции транскрипции"
    )
    
    class Config:
        env_prefix = "TRANSCRIPTION_"


class RecognitionConfig(BaseSettings):
    """Конфигурация этапа распознавания голосов"""
    model_name: str = Field(
        "speechbrain/spkrec-ecapa-voxceleb", 
        description="Модель SpeechBrain"
    )
    device: str = Field("cpu", description="Устройство для вычислений")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Порог распознавания")
    embeddings_path: Optional[str] = Field(None, description="Путь к базе эмбеддингов")
    index_path: Optional[str] = Field(None, description="Путь к FAISS индексу")
    
    class Config:
        env_prefix = "RECOGNITION_"


class CardDAVConfig(BaseSettings):
    """Конфигурация CardDAV"""
    server_url: Optional[str] = Field(None, description="URL CardDAV сервера")
    username: Optional[str] = Field(None, description="Имя пользователя")
    password: Optional[str] = Field(None, description="Пароль")
    timeout: int = Field(10, gt=0, description="Таймаут запросов в секундах")
    
    class Config:
        env_prefix = "CARDDAV_"


class QueueConfig(BaseSettings):
    """Конфигурация очереди задач"""
    max_concurrent_tasks: int = Field(2, gt=0, description="Максимум параллельных задач")
    max_queue_size: int = Field(100, gt=0, description="Максимальный размер очереди")
    task_timeout: int = Field(3600, gt=0, description="Таймаут задачи в секундах")
    cleanup_interval: int = Field(300, gt=0, description="Интервал очистки в секундах")
    volume_path: str = Field("/app/volume", description="Путь к volume")
    
    class Config:
        env_prefix = "QUEUE_"


class ServerConfig(BaseSettings):
    """Конфигурация сервера"""
    host: str = Field("0.0.0.0", description="Хост сервера")
    port: int = Field(8000, gt=0, le=65535, description="Порт сервера")
    version: str = Field("1.0.0", description="Версия API")
    debug: bool = Field(False, description="Режим отладки")
    
    class Config:
        env_prefix = "SERVER_"


class FilesConfig(BaseSettings):
    """Конфигурация файлов"""
    max_size: int = Field(500 * 1024 * 1024, gt=0, description="Максимальный размер файла")
    allowed_formats: List[str] = Field(
        default_factory=lambda: ["wav", "mp3", "ogg", "flac", "aac", "m4a"],
        description="Разрешенные форматы"
    )
    
    class Config:
        env_prefix = "FILES_"


class LoggingConfig(BaseSettings):
    """Конфигурация логирования"""
    level: str = Field("INFO", description="Уровень логирования")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Формат логов"
    )
    file: Optional[str] = Field(None, description="Файл для логов")
    external_levels: Dict[str, str] = Field(
        default_factory=lambda: {},
        description="Уровни для внешних библиотек"
    )
    
    class Config:
        env_prefix = "LOGGING_"


class CORSConfig(BaseSettings):
    """Конфигурация CORS"""
    origins: List[str] = Field(default_factory=lambda: ["*"], description="Разрешенные origins")
    
    class Config:
        env_prefix = "CORS_"


class AppSettings(BaseSettings):
    """Основные настройки приложения"""
    
    # Подконфигурации
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    carddav: CardDAVConfig = Field(default_factory=CardDAVConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    
    @validator('recognition')
    def validate_recognition_paths(cls, v):
        """Валидация путей для распознавания"""
        if v.embeddings_path and not Path(v.embeddings_path).exists():
            raise ValueError(f"Embeddings path не существует: {v.embeddings_path}")
        if v.index_path and not Path(v.index_path).exists():
            raise ValueError(f"Index path не существует: {v.index_path}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_settings_from_yaml(yaml_path: str) -> AppSettings:
    """Загрузка настроек из YAML файла"""
    
    if not Path(yaml_path).exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    # Преобразование YAML в формат для Pydantic
    settings_data = {}
    
    for section, config in yaml_data.items():
        if isinstance(config, dict):
            settings_data[section] = config
    
    return AppSettings(**settings_data)


def load_settings(config_path: Optional[str] = None) -> AppSettings:
    """Загрузка настроек с приоритетом: переменные окружения > YAML > значения по умолчанию"""
    
    # Определение пути к конфигурации
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/default.yaml")
    
    # Загрузка из YAML если файл существует
    if Path(config_path).exists():
        return load_settings_from_yaml(config_path)
    else:
        # Использование значений по умолчанию и переменных окружения
        return AppSettings()
