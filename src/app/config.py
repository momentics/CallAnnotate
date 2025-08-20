# src/app/config.py
# -*- coding: utf-8 -*-
#  Apache-2.0
#  Автор: akoodoy@capilot.ru
#  Ссылка: https://github.com/momentics/CallAnnotate

"""
Конфигурация приложения CallAnnotate с использованием Pydantic v2 и pydantic-settings.

Правки для этапа предобработки (PreprocessingConfig):
- Параметры приведены к тем, которые реально используются в PreprocessingStage.
- Дефолтные значения синхронизированы с config/default.yaml.
- Добавлены русскоязычные комментарии по смыслу каждого поля.
- Явно валидируются допустимые значения (audio_format, channels, chunk_overlap_method).
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, field_validator, validator

class APIConfig(BaseModel):
    base_path: str = Field("/api/v1", description="Базовый путь для REST API")
    class Config:
        env_prefix = "API_"

class ModelsConfig(BaseModel):
    cache_dir: str = "./volume/models"

class RecognitionConfig(BaseSettings):
    model: str = Field("speechbrain/spkrec-ecapa-voxceleb", description="Модель SpeechBrain")
    device: str = Field("cpu", description="Устройство для вычислений")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Порог распознавания")
    embeddings_path: Optional[str] = Field(None, description="Путь к базе эмбеддингов")
    index_path: Optional[str] = Field(None, description="Путь к FAISS индексу")

    @field_validator("embeddings_path", "index_path", mode="after")
    def _ensure_dir(cls, v: Optional[str]) -> Optional[str]:
        if v:
            p = Path(v).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return str(p)
        return v

    class Config:
        env_prefix = "RECOGNITION_"

class DiarizationConfig(BaseSettings):
    model: str = Field("pyannote/speaker-diarization-3.1", description="Модель pyannote.audio")
    use_auth_token: Optional[str] = Field(None, description="HuggingFace токен")
    device: str = Field("cpu", description="Устройство для вычислений")
    batch_size: int = Field(32, gt=0, description="Размер пакета")
    window_enabled: bool = Field(True, description="Включить оконный режим диаризации")
    window_size: float = Field(30.0, gt=0, description="Длина окна (сек)")
    hop_size: float = Field(10.0, ge=0, description="Шаг окна (сек)")

    class Config:
        env_prefix = "DIARIZATION_"

class MetricsConfig(BaseModel):
    confidence: bool = Field(True, description="Сбор avg confidence")
    avg_logprob: bool = Field(True, description="Сбор avg logprob")
    no_speech_prob: bool = Field(True, description="Сбор avg no_speech_prob")
    timing: bool = Field(True, description="Сбор времени транскрипции")

class TranscriptionConfig(BaseSettings):
    model: str = Field("openai/whisper-base", description="Модель Whisper (например, openai/whisper-small)")
    device: str = Field("cpu", description="Устройство для вычислений")
    language: str = Field("ru", description="Язык транскрипции или 'auto'")
    batch_size: int = Field(16, gt=0, description="Размер пакета")
    task: str = Field("transcribe", description="Задача: transcribe или translate")

    # Привязка к диаризации
    min_segment_duration: float = Field(0.2, gt=0.0, description="Мин. длительность сегмента (сек)")
    max_silence_between: float = Field(0.0, ge=0.0, description="Макс. пауза между сегментом и диаризацией (сек)")
    min_overlap: float = Field(0.3, ge=0.0, le=1.0, description="Мин. доля перекрытия при привязке сегмента к спикеру")

    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="Настройки метрик") # type: ignore

    class Config:
        env_prefix = "TRANSCRIPTION_"

class CardDAVConfig(BaseSettings):
    enabled: bool = Field(True, description="Включить CardDAV")
    url: Optional[str] = Field(None, description="URL CardDAV сервера")
    username: Optional[str] = Field(None, description="Имя пользователя")
    password: Optional[str] = Field(None, description="Пароль")
    timeout: int = Field(30, gt=0, description="Таймаут запросов (сек)")
    verify_ssl: bool = Field(True, description="Проверка SSL сертификатов")

    class Config:
        env_prefix = "CARDDAV_"

class QueueConfig(BaseSettings):
    max_concurrent_tasks: int = Field(2, gt=0, description="Максимум параллельных задач")
    max_queue_size: int = Field(100, gt=0, description="Максимальный размер очереди")
    task_timeout: int = Field(3600, gt=0, description="Таймаут задачи (сек)")
    cleanup_interval: int = Field(300, gt=0, description="Интервал очистки (сек)")
    volume_path: str = Field("./volume", description="Путь к volume")

    class Config:
        env_prefix = "QUEUE_"

class ServerConfig(BaseSettings):
    host: str = Field("0.0.0.0", description="Хост сервера")
    port: int = Field(8000, gt=0, le=65535, description="Порт сервера")
    workers: int = Field(1, gt=0, description="Количество воркеров")
    reload: bool = Field(False, description="Автоперезагрузка")
    log_level: str = Field("info", description="Уровень логирования uvicorn")
    version: Optional[str] = Field("1.0.0", description="Версия сервиса")

    class Config:
        env_prefix = "SERVER_"

class FilesConfig(BaseSettings):
    max_size: int = Field(524288000, gt=0, description="Максимальный размер файла")
    allowed_formats: List[str] = Field(
        default_factory=lambda: ["wav", "mp3", "ogg", "flac", "aac", "m4a", "mp4"],
        description="Разрешенные форматы"
    )
    temp_cleanup_hours: int = Field(24, gt=0, description="Часы до очистки временных файлов")
    class Config:
        env_prefix = "FILES_"

class RotationConfig(BaseModel):
    enabled: bool = Field(False, description="Включить ротацию логов")
    when: str = Field("midnight", description="Когда ротировать (sec/min/hour/daily/midnight)")
    interval: int = Field(1, description="Интервал ротации")
    backup_count: int = Field(7, description="Количество удерживаемых файлов")
    class Config:
        extra = "ignore"

class LoggingConfig(BaseSettings):
    level: str = Field("INFO", description="Уровень логирования")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Формат логов")
    file: Optional[str] = Field(None, description="Путь к лог-файлу")
    rotation: RotationConfig = Field(default_factory=RotationConfig, description="Параметры ротации логов") # type: ignore
    external_levels: Dict[str, str] = Field(
        default_factory=lambda: {"uvicorn": "INFO", "fastapi": "INFO", "asyncio": "WARNING"},
        description="Уровни для внешних библиотек"
    )
    class Config:
        env_prefix = "LOGGING_"
        extra = "ignore"

class CORSConfig(BaseSettings):
    origins: List[str] = Field(default_factory=lambda: ["*"], description="Разрешенные origins")
    allow_credentials: bool = Field(True, description="Разрешить credentials")
    allow_methods: List[str] = Field(default_factory=lambda: ["*"], description="Разрешенные методы")
    allow_headers: List[str] = Field(default_factory=lambda: ["*"], description="Разрешенные заголовки")
    class Config:
        env_prefix = "CORS_"

class VoiceInfo(BaseModel):
    name: str = Field(..., description="Имя владельца голоса")
    embedding: str = Field(..., description="Путь к файлу эмбеддинга")
    phone: Optional[str] = Field(None, description="Номер телефона")
    description: Optional[str] = Field(None, description="Описание")

class VoiceInfoConfig(BaseModel):
    name: str
    embedding: str
    phone: Optional[str] = None
    description: Optional[str] = None

class WebhookConfig(BaseModel):
    enabled: bool = Field(True, description="Включить веб-хуки")
    timeout: int = Field(30, gt=0, description="Таймаут запроса")
    retry_count: int = Field(3, ge=0, description="Количество повторов")
    retry_delay: int = Field(5, gt=0, description="Задержка между повторами")

class WebSocketConfig(BaseModel):
    ping_interval: int = Field(30, gt=0, description="Интервал пинга")
    ping_timeout: int = Field(10, gt=0, description="Таймаут пинга")
    close_timeout: int = Field(10, gt=0, description="Таймаут закрытия")

class NotificationsConfig(BaseModel):
    webhooks: WebhookConfig = Field(default_factory=WebhookConfig) # type: ignore
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig) # type: ignore

class RateLimitingConfig(BaseModel):
    enabled: bool = Field(True, description="Включить ограничение скорости")
    requests_per_minute: int = Field(60, gt=0, description="Запросов в минуту")

class FileUploadConfig(BaseModel):
    virus_scan: bool = Field(False, description="Проверка на вирусы")
    content_validation: bool = Field(True, description="Валидация содержимого")

class SecurityConfig(BaseModel):
    api_key_required: bool = Field(False, description="Требовать API ключ")
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig) # type: ignore
    file_upload: FileUploadConfig = Field(default_factory=FileUploadConfig) # type: ignore

class MonitoringConfig(BaseModel):
    metrics_enabled: bool = Field(True, description="Включить метрики")
    health_check_interval: int = Field(60, gt=0, description="Интервал проверки здоровья")
    performance_logging: bool = Field(True, description="Логирование производительности")

class FeaturesConfig(BaseModel):
    real_time_processing: bool = Field(True, description="Обработка в реальном времени")
    batch_processing: bool = Field(True, description="Пакетная обработка")
    webhook_callbacks: bool = Field(True, description="Колбеки через веб-хуки")
    file_download: bool = Field(True, description="Скачивание файлов")
    task_cancellation: bool = Field(True, description="Отмена задач")
    progress_tracking: bool = Field(True, description="Отслеживание прогресса")

class PreprocessingConfig(BaseSettings):
    """
    Конфигурация этапа предобработки (preprocess).

    Логика:
      1) (опционально) SoX noisered — если sox доступен в системе.
      2) RNNoise — работает покадрово 10мс при 48kHz; «мягко» отключается при отсутствии либ.
      3) DeepFilterNet — DFN2/3; «мягко» отключается, если пакет df.* не установлен.
      4) Чанкование, обработка и склейка (linear/windowed).
      5) RMS-ceiling до целевого уровня (не усиливает шум).
    """
    # Основные параметры
    model: str = Field("DeepFilterNet2", description="Идентификатор DFN-модели (для логов/индикации)")
    device: str = Field("cpu", description="Устройство вычислений (cpu/cuda) для DFN")
    chunk_duration: float = Field(30.0, gt=0, description="Длительность чанка, сек")
    overlap: float = Field(0.2, ge=0.0, description="Перекрытие чанков, сек")
    target_rms: float = Field(-20.0, description="Целевой RMS (дБFS), потолок: уменьшает только при превышении")

    # SoX (опционально)
    sox_enabled: bool = Field(False, description="Включить статическое шумоподавление SoX (если доступен)")
    sox_noise_profile_duration: float = Field(2.0, gt=0, description="Длительность профиля шума, сек")
    sox_noise_reduction: float = Field(0.3, ge=0.0, le=1.0, description="Сила подавления шума (0.0-1.0)")
    sox_gain_normalization: bool = Field(True, description="Применять 'sox gain -n' (нормализация пиков)")

    # RNNoise
    rnnoise_enabled: bool = Field(True, description="Включить RNNoise (мягкое отключение при отсутствии lib)")
    rnnoise_sample_rate: int = Field(48000, description="Рабочая частота RNNoise (фиксированно 48000 Гц)")

    # DeepFilterNet
    deepfilter_enabled: bool = Field(True, description="Включить DeepFilterNet (DFN2/3)")
    deepfilter_sample_rate: int = Field(48000, description="Рабочая частота DFN-процесса (Гц)")

    # Параметры выходного файла
    output_suffix: str = Field("_processed", description="Суффикс имени выходного файла")
    audio_format: str = Field("wav", description="Формат результата (wav/flac/ogg/mp3)")
    bit_depth: int = Field(16, gt=0, description="Битовость PCM при записи WAV")

    # Приведение канальности/частоты перед обработкой
    channels: str = Field("mono", description="Режим каналов: mono|stereo|original")
    sample_rate_target: Optional[int] = Field(16000, description="Целевая частота дискретизации Гц; null — не менять")

    # Склейка чанков
    chunk_overlap_method: str = Field("windowed", description="Метод склейки: linear|windowed")

    # Служебное/ресурсы
    processing_threads: int = Field(1, gt=0, description="(Зарезервировано) Количество потоков обработки")
    memory_limit_mb: int = Field(1024, gt=0, description="(Зарезервировано) Лимит памяти в МБ")
    temp_cleanup: bool = Field(True, description="Очищать временные файлы")
    preserve_original: bool = Field(True, description="Сохранять оригинальный файл")
    debug_mode: bool = Field(False, description="Режим расширенного логирования")
    save_intermediate: bool = Field(False, description="Сохранять промежуточные файлы")
    progress_interval: int = Field(10, gt=0, le=100, description="Шаг отчёта прогресса, %")

    @field_validator("audio_format")
    def validate_audio_format(cls, v):
        allowed = ["wav", "flac", "ogg", "mp3"]
        if v not in allowed:
            raise ValueError(f"Формат {v} не поддерживается. Доступные: {allowed}")
        return v

    @field_validator("channels")
    def validate_channels(cls, v):
        allowed = ["mono", "stereo", "original"]
        if v not in allowed:
            raise ValueError(f"Режим каналов {v} не поддерживается. Доступные: {allowed}")
        return v

    @field_validator("chunk_overlap_method")
    def validate_overlap_method(cls, v):
        allowed = ["linear", "windowed"]
        if v not in allowed:
            raise ValueError(f"Метод склейки {v} не поддерживается. Доступные: {allowed}")
        return v

    class Config:
        env_prefix = "PREPROCESS_"

class AppSettings(BaseSettings): 
    api: APIConfig = APIConfig() # type: ignore
    server: ServerConfig = ServerConfig() # type: ignore
    queue: QueueConfig = QueueConfig() # type: ignore
    files: FilesConfig = FilesConfig() # type: ignore
    logging: LoggingConfig = LoggingConfig() # type: ignore
    cors: CORSConfig = CORSConfig() # type: ignore
    models: ModelsConfig = ModelsConfig() # type: ignore
    diarization: DiarizationConfig = DiarizationConfig() # type: ignore
    transcription: TranscriptionConfig = TranscriptionConfig() # type: ignore
    recognition: RecognitionConfig = RecognitionConfig() # type: ignore
    carddav: CardDAVConfig = CardDAVConfig() # type: ignore
    voices: List['VoiceInfoConfig'] = []
    notifications: NotificationsConfig = NotificationsConfig() # type: ignore
    security: SecurityConfig = SecurityConfig() # type: ignore
    monitoring: MonitoringConfig = MonitoringConfig() # type: ignore
    features: FeaturesConfig = FeaturesConfig() # type: ignore
    preprocess: PreprocessingConfig = PreprocessingConfig() # type: ignore

    class Config:
        env_file = ".env"
        case_sensitive = False

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Рекурсивно сливает словари:
    - override переопределяет base,
    - вложенные dict объединяются рекурсивно.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_settings_from_yaml(yaml_path: str) -> AppSettings:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f) or {}
    defaults = AppSettings().model_dump()
    merged = _deep_merge(defaults, yaml_data)
    settings = AppSettings(**merged)
    return settings

def load_settings(config_path: Optional[str] = None) -> AppSettings:
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/default.yaml")
    if Path(config_path).exists():
        settings = load_settings_from_yaml(config_path)
    else:
        settings = AppSettings()

    vol = Path(settings.queue.volume_path).expanduser().resolve()
    vol.mkdir(parents=True, exist_ok=True)

    models_cache = Path(settings.models.cache_dir).expanduser().resolve()
    models_cache.mkdir(parents=True, exist_ok=True)

    return settings
