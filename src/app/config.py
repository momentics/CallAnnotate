# -*- coding: utf-8 -*-
#  Apache-2.0
#  Автор: akoodoy@capilot.ru
#  Ссылка: https://github.com/momentics/CallAnnotate

"""
Конфигурация приложения CallAnnotate с использованием Pydantic v2 и pydantic-settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# BaseSettings берём из pydantic-settings
from pydantic_settings import BaseSettings
# Field и валидаторы — из pydantic
from pydantic import BaseModel, Field, field_validator, validator

class APIConfig(BaseModel):
    """Конфигурация HTTP API"""
    base_path: str = Field("/api/v1", description="Базовый путь для REST API")

    class Config:
        env_prefix = "API_"

class RecognitionConfig(BaseSettings):
    """Конфигурация этапа распознавания голосов"""
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
    """Конфигурация этапа диаризации"""
    model: str = Field("pyannote/speaker-diarization-3.1", description="Модель pyannote.audio")
    use_auth_token: Optional[str] = Field(None, description="HuggingFace токен")
    device: str = Field("cpu", description="Устройство для вычислений")
    batch_size: int = Field(32, gt=0, description="Размер пакета")
    window_enabled: bool = Field(True, description="Включить оконный режим диаризации")
    window_size: float = Field(30.0, gt=0, description="Длина окна для скользящего батча в секундах")
    hop_size: float = Field(10.0, ge=0, description="Шаг окна для скользящего батча в секундах")

    class Config:
        env_prefix = "DIARIZATION_"


class MetricsConfig(BaseModel):
    """Конфигурация метрик для транскрипции"""
    confidence: bool = Field(True, description="Сбор avg confidence")
    avg_logprob: bool = Field(True, description="Сбор avg logprob")
    no_speech_prob: bool = Field(True, description="Сбор avg no_speech_prob")
    timing: bool = Field(True, description="Сбор времени транскрипции")


class TranscriptionConfig(BaseSettings):
    """Конфигурация этапа транскрипции"""
    model: str = Field("openai/whisper-base", description="Модель Whisper")
    device: str = Field("cpu", description="Устройство для вычислений")
    language: str = Field("ru", description="Язык транскрипции или auto")
    batch_size: int = Field(16, gt=0, description="Размер пакета")
    task: str = Field("transcribe", description="Задача: transcribe или translate")

    # Новые параметры для точного выравнивания сегментов
    min_segment_duration: float = Field(
        0.2, gt=0.0, description="Минимальная длительность сегмента (сек)"
    )
    max_silence_between: float = Field(
        0.0, ge=0.0, description="Максимальная пауза между сегментом и диаризацией (сек)"
    )
    min_overlap: float = Field(
        0.3, ge=0.0, le=1.0, description="Минимальное отношение перекрытия при привязке сегмента к спикеру"
    )

    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="Настройки сбора метрик")
    
    class Config:
        env_prefix = "TRANSCRIPTION_"



class CardDAVConfig(BaseSettings):
    """Конфигурация CardDAV"""
    enabled: bool = Field(True, description="Включить CardDAV")
    url: Optional[str] = Field(None, description="URL CardDAV сервера")
    username: Optional[str] = Field(None, description="Имя пользователя")
    password: Optional[str] = Field(None, description="Пароль")
    timeout: int = Field(30, gt=0, description="Таймаут запросов в секундах")
    verify_ssl: bool = Field(True, description="Проверка SSL сертификатов")
    
    class Config:
        env_prefix = "CARDDAV_"


class QueueConfig(BaseSettings):
    """Конфигурация очереди задач"""
    max_concurrent_tasks: int = Field(2, gt=0, description="Максимум параллельных задач")
    max_queue_size: int = Field(100, gt=0, description="Максимальный размер очереди")
    task_timeout: int = Field(3600, gt=0, description="Таймаут задачи в секундах")
    cleanup_interval: int = Field(300, gt=0, description="Интервал очистки в секундах")
    volume_path: str = Field("./volume", description="Путь к volume")

    class Config:
        env_prefix = "QUEUE_"


class ServerConfig(BaseSettings):
    """Конфигурация сервера"""
    host: str = Field("0.0.0.0", description="Хост сервера")
    port: int = Field(8000, gt=0, le=65535, description="Порт сервера")
    workers: int = Field(1, gt=0, description="Количество воркеров")
    reload: bool = Field(False, description="Автоперезагрузка")
    log_level: str = Field("info", description="Уровень логирования uvicorn")
    version: Optional[str] = Field("1.0.0", description="Версия сервиса")    

    class Config:
        env_prefix = "SERVER_"


class FilesConfig(BaseSettings):
    """Конфигурация файлов"""
    max_size: int = Field(524288000, gt=0, description="Максимальный размер файла")
    allowed_formats: List[str] = Field(
        default_factory=lambda: ["wav", "mp3", "ogg", "flac", "aac", "m4a", "mp4"],
        description="Разрешенные форматы"
    )
    temp_cleanup_hours: int = Field(24, gt=0, description="Часы до очистки временных файлов")
    
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
        default_factory=lambda: {
            "uvicorn": "INFO",
            "fastapi": "INFO",
            "asyncio": "WARNING"
        },
        description="Уровни для внешних библиотек"
    )
    
    class Config:
        env_prefix = "LOGGING_"


class CORSConfig(BaseSettings):
    """Конфигурация CORS"""
    origins: List[str] = Field(default_factory=lambda: ["*"], description="Разрешенные origins")
    allow_credentials: bool = Field(True, description="Разрешить credentials")
    allow_methods: List[str] = Field(default_factory=lambda: ["*"], description="Разрешенные методы")
    allow_headers: List[str] = Field(default_factory=lambda: ["*"], description="Разрешенные заголовки")
    
    class Config:
        env_prefix = "CORS_"


class VoiceInfo(BaseModel):
    """Информация о голосе"""
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
    """Конфигурация веб-хуков"""
    enabled: bool = Field(True, description="Включить веб-хуки")
    timeout: int = Field(30, gt=0, description="Таймаут запроса")
    retry_count: int = Field(3, ge=0, description="Количество повторов")
    retry_delay: int = Field(5, gt=0, description="Задержка между повторами")


class WebSocketConfig(BaseModel):
    """Конфигурация WebSocket"""
    ping_interval: int = Field(30, gt=0, description="Интервал пинга")
    ping_timeout: int = Field(10, gt=0, description="Таймаут пинга")
    close_timeout: int = Field(10, gt=0, description="Таймаут закрытия")


class NotificationsConfig(BaseModel):
    """Конфигурация уведомлений"""
    webhooks: WebhookConfig = Field(default_factory=WebhookConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)



class RateLimitingConfig(BaseModel):
    """Конфигурация ограничения скорости"""
    enabled: bool = Field(True, description="Включить ограничение скорости")
    requests_per_minute: int = Field(60, gt=0, description="Запросов в минуту")


class FileUploadConfig(BaseModel):
    """Конфигурация загрузки файлов"""
    virus_scan: bool = Field(False, description="Проверка на вирусы")
    content_validation: bool = Field(True, description="Валидация содержимого")


class SecurityConfig(BaseModel):
    """Конфигурация безопасности"""
    api_key_required: bool = Field(False, description="Требовать API ключ")
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    file_upload: FileUploadConfig = Field(default_factory=FileUploadConfig)


class MonitoringConfig(BaseModel):
    """Конфигурация мониторинга"""
    metrics_enabled: bool = Field(True, description="Включить метрики")
    health_check_interval: int = Field(60, gt=0, description="Интервал проверки здоровья")
    performance_logging: bool = Field(True, description="Логирование производительности")


class FeaturesConfig(BaseModel):
    """Конфигурация функций"""
    real_time_processing: bool = Field(True, description="Обработка в реальном времени")
    batch_processing: bool = Field(True, description="Пакетная обработка")
    webhook_callbacks: bool = Field(True, description="Колбеки через веб-хуки")
    file_download: bool = Field(True, description="Скачивание файлов")
    task_cancellation: bool = Field(True, description="Отмена задач")
    progress_tracking: bool = Field(True, description="Отслеживание прогресса")

class PreprocessingConfig(BaseSettings):
    """Конфигурация этапа предобработки (preprocess)"""
    # === Основные параметры обработки ===
    model: str = Field("DeepFilterNet2", description="Модель для улучшения речи (DeepFilterNet2)")
    device: str = Field("cpu", description="Устройство вычислений (cpu/cuda)")
    chunk_duration: float = Field(10.0, gt=0, description="Длительность чанка в секундах")
    overlap: float = Field(0.5, ge=0, description="Перекрытие между чанками в секундах")
    target_rms: float = Field(-20.0, description="Целевой RMS уровень в дБFS")
    
    # === Конфигурация SoX (системный аудиопроцессор) ===
    sox_enabled: bool = Field(False, description="Включить предварительную обработку SoX")
    sox_noise_profile_duration: float = Field(5.0, gt=0, description="Длительность для создания профиля шума (сек)")
    sox_noise_reduction: float = Field(0.4, ge=0.0, le=1.0, description="Коэффициент подавления шума (0.0-1.0)")
    sox_gain_normalization: bool = Field(True, description="Применять автоматическую нормализацию усиления")
    
    # === Конфигурация RNNoise (нейронное шумоподавление) ===
    rnnoise_enabled: bool = Field(True, description="Включить RNNoise шумоподавление")
    rnnoise_sample_rate: int = Field(48000, description="Частота дискретизации для RNNoise (только 48kHz)")
    
    # === Конфигурация DeepFilterNet (улучшение речи) ===
    deepfilter_enabled: bool = Field(True, description="Включить DeepFilterNet обработку")
    deepfilter_sample_rate: int = Field(48000, description="Частота дискретизации для DeepFilterNet (похоже 48kHz)")
    
    # === Параметры выходных файлов ===
    output_suffix: str = Field("_processed", description="Суффикс для обработанного файла")
    audio_format: str = Field("wav", description="Формат выходного аудио (wav/flac/ogg)")
    bit_depth: int = Field(16, gt=0, description="Глубина битов PCM (16/24/32)")
    channels: str = Field("mono", description="Количество каналов (mono/stereo/original)")
    sample_rate_target: Optional[int] = Field(16000, description="Целевая частота дискретизации (null = сохранить исходную)")
    
    # === Параметры обработки ===
    chunk_overlap_method: str = Field("linear", description="Метод склеивания чанков (linear/windowed)")
    processing_threads: int = Field(1, gt=0, description="Количество потоков для параллельной обработки")
    memory_limit_mb: int = Field(1024, gt=0, description="Лимит использования памяти в МБ")
    
    # === Служебные параметры ===
    temp_cleanup: bool = Field(True, description="Автоматическая очистка временных файлов")
    preserve_original: bool = Field(True, description="Сохранять оригинальный файл")
    debug_mode: bool = Field(False, description="Режим отладки с дополнительным логированием")
    save_intermediate: bool = Field(False, description="Сохранять промежуточные результаты")
    progress_interval: int = Field(5, gt=0, le=100, description="Интервал отчета о прогрессе (%)")

    @field_validator("audio_format")
    def validate_audio_format(cls, v):
        allowed_formats = ["wav", "flac", "ogg", "mp3"]
        if v not in allowed_formats:
            raise ValueError(f"Формат {v} не поддерживается. Доступные: {allowed_formats}")
        return v

    @field_validator("channels")
    def validate_channels(cls, v):
        allowed_channels = ["mono", "stereo", "original"]
        if v not in allowed_channels:
            raise ValueError(f"Режим каналов {v} не поддерживается. Доступные: {allowed_channels}")
        return v

    @field_validator("chunk_overlap_method")
    def validate_overlap_method(cls, v):
        allowed_methods = ["linear", "windowed"]
        if v not in allowed_methods:
            raise ValueError(f"Метод склеивания {v} не поддерживается. Доступные: {allowed_methods}")
        return v

    class Config:
        env_prefix = "PREPROCESS_"


class AppSettings(BaseSettings):
    """Основные настройки приложения"""
    
    # Подконфигурации
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    transcription: TranscriptionConfig = Field(
        default_factory=TranscriptionConfig,
        description="Конфигурация этапа транскрипции"
    )
    recognition: RecognitionConfig = Field(default_factory=RecognitionConfig)
    carddav: CardDAVConfig = Field(default_factory=CardDAVConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)

    api: APIConfig = Field(default_factory=APIConfig)

    # Список известных голосов: имя + путь к файлу эмбеддинга
    voices: List[VoiceInfoConfig] = Field(default_factory=list, description="Известные голоса")
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    
    preprocess: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


    @validator('recognition')
    def validate_recognition_paths(cls, v):
        """Валидация путей для распознавания"""
        if v.embeddings_path and not Path(v.embeddings_path).exists():
            # Создаем директорию если её нет  
            Path(v.embeddings_path).mkdir(parents=True, exist_ok=True)
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
    return AppSettings(**yaml_data)


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
