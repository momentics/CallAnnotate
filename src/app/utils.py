# -*- coding: utf-8 -*-
"""
Вспомогательные функции для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import logging
import logging.config
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Any, NamedTuple

import librosa
import soundfile as sf
from fastapi import UploadFile

from .schemas import AudioMetadata



class ValidationResult:
    def __init__(self, is_valid: bool, error: str = None):
        self.is_valid = is_valid
        self.error = error

def validate_audio_file_path(file_path: str) -> ValidationResult:
    """
    Примитивная проверка валидности аудиофайла для CallAnnotate
    - Существование файла
    - Проверка расширения на допустимые (['.wav', '.mp3', '.ogg', '.flac'])
    - Файл не пустой
    """
    if not os.path.exists(file_path):
        return ValidationResult(False, f"Файл '{file_path}' не существует")
    if not os.path.isfile(file_path):
        return ValidationResult(False, f"Путь '{file_path}' не является файлом")
    ext = Path(file_path).suffix.lower()
    allowed = {".wav", ".mp3", ".ogg", ".flac"}
    if ext not in allowed:
        return ValidationResult(False, f"Формат '{ext}' не поддерживается")
    if os.path.getsize(file_path) == 0:
        return ValidationResult(False, f"Файл '{file_path}' пустой")
    return ValidationResult(True)

def setup_logging(config: Dict[str, Any]):
    """Настройка системы логирования"""
    
    log_level = config.get("level", "INFO").upper()
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = config.get("file")
    external_levels = config.get("external_levels", {})
    
    # Базовая конфигурация
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": log_level
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"]
        },
        "loggers": {}
    }
    
    # Добавление файлового логгера если указан
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging_config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "filename": str(log_path),
            "formatter": "default",
            "level": log_level,
            "encoding": "utf-8"
        }
        
        logging_config["root"]["handlers"].append("file")
    
    # Настройка уровней для внешних библиотек
    for logger_name, level in external_levels.items():
        logging_config["loggers"][logger_name] = {
            "level": level.upper(),
            "propagate": True
        }
    
    logging.config.dictConfig(logging_config)
    
    logger = logging.getLogger(__name__)
    logger.info("Система логирования настроена")


def validate_audio_file(file: UploadFile) -> ValidationResult:
    """Валидация аудиофайла"""
    
    # Проверка MIME-типа
    if not file.content_type:
        return ValidationResult(False, "Не указан тип содержимого")
    
    if not file.content_type.startswith("audio/"):
        return ValidationResult(False, f"Неподдерживаемый тип файла: {file.content_type}")
    
    # Проверка расширения файла
    if not file.filename:
        return ValidationResult(False, "Не указано имя файла")
    
    file_ext = Path(file.filename).suffix.lower()
    supported_extensions = [".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".mp4"]
    
    if file_ext not in supported_extensions:
        return ValidationResult(False, f"Неподдерживаемое расширение файла: {file_ext}")
    
    return ValidationResult(True)


def extract_audio_metadata(file_path: str) -> AudioMetadata:
    """Извлечение метаданных из аудиофайла"""
    
    try:
        file_path_obj = Path(file_path)
        
        # Получение базовой информации о файле
        file_size = file_path_obj.stat().st_size
        filename = file_path_obj.name
        
        # Загрузка аудио для анализа
        try:
            # Попытка использовать librosa
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            duration = len(audio_data) / sample_rate
            channels = 1  # librosa загружает моно по умолчанию
            
        except Exception:
            # Fallback на soundfile
            try:
                audio_info = sf.info(file_path)
                duration = audio_info.duration
                sample_rate = audio_info.samplerate
                channels = audio_info.channels
            except Exception as e:
                # Если не удается получить метаданные, используем значения по умолчанию
                logging.getLogger(__name__).warning(f"Не удалось извлечь метаданные аудио: {e}")
                duration = 0.0
                sample_rate = 16000
                channels = 1
        
        # Определение формата
        file_format = file_path_obj.suffix.lower().lstrip('.')
        
        # Примерная оценка битрейта
        if duration > 0:
            bitrate = int((file_size * 8) / duration)
        else:
            bitrate = None
        
        return AudioMetadata(
            filename=filename,
            duration=round(duration, 3),
            sample_rate=sample_rate,
            channels=channels,
            format=file_format,
            bitrate=bitrate,
            size_bytes=file_size
        )
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка извлечения метаданных из {file_path}: {e}")
        
        # Возвращаем базовые метаданные
        return AudioMetadata(
            filename=Path(file_path).name,
            duration=0.0,
            sample_rate=16000,
            channels=1,
            format="unknown",
            bitrate=None,
            size_bytes=Path(file_path).stat().st_size if Path(file_path).exists() else 0
        )


def create_task_metadata(
    task_id: str,
    file_path: str,
    filename: str,
    priority: int = 5,
    websocket_client_id: str = None
) -> Dict[str, Any]:
    """Создание метаданных задачи"""
    
    return {
        "task_id": task_id,
        "file_path": file_path,
        "filename": filename,
        "priority": priority,
        "created_at": datetime.now().isoformat(),
        "websocket_client_id": websocket_client_id,
        "status": "queued"
    }


def ensure_directory(path: str) -> Path:
    """Создание директории если она не существует"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def cleanup_temp_files(directory: str, max_age_hours: int = 24):
    """Очистка временных файлов старше указанного времени"""
    
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in directory_path.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        logging.getLogger(__name__).debug(f"Удален временный файл: {file_path}")
                    except Exception as e:
                        logging.getLogger(__name__).error(f"Ошибка удаления файла {file_path}: {e}")
    
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка очистки временных файлов: {e}")


def format_duration(seconds: float) -> str:
    """Форматирование длительности в читаемый вид"""
    
    if seconds < 60:
        return f"{seconds:.1f}с"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}м {secs}с"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}ч {minutes}м {secs}с"


def get_supported_audio_formats() -> Dict[str, str]:
    """Получение списка поддерживаемых аудио форматов"""
    
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4"
    }
