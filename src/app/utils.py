"""
Вспомогательные функции для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import shutil
import logging.config
from pathlib import Path
from typing import Dict, Any, Optional, Union
from fastapi import UploadFile
from datetime import datetime, timedelta

import librosa

from  .config import AppSettings

from .schemas import AudioMetadata
from .schemas import VoiceInfo


class ValidationResult:
    def __init__(self, ok: bool, error: str = None):
        self.is_valid = ok
        self.error = error


def get_supported_audio_formats() -> Dict[str, str]:
    """
    Возвращает словарь поддерживаемых аудио-расширений.
    """
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
    }

def validate_audio_file(upload: UploadFile) -> ValidationResult:
    """
    Проверка UploadFile на корректный аудиофайл по расширению.
    """
    filename = upload.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in get_supported_audio_formats():
        return ValidationResult(False, f"Unsupported audio format '{ext}'")
    return ValidationResult(True)

def validate_audio_file_path(path: str) -> ValidationResult:
    """
    Проверка файла по пути на существование, тип и расширение.
    """
    if not os.path.exists(path):
        return ValidationResult(False, f"File '{path}' not found")
    if not os.path.isfile(path):
        return ValidationResult(False, f"'{path}' is not a file")
    ext = Path(path).suffix.lower()
    if ext not in get_supported_audio_formats():
        return ValidationResult(False, f"Unsupported audio format '{ext}'")
    if os.path.getsize(path) == 0:
        return ValidationResult(False, "File is empty")
    return ValidationResult(True)

def ensure_directory(path: str) -> Path:
    """
    Убеждается в наличии директории, создаёт её при отсутствии.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def cleanup_temp_files(directory: str, max_age_hours: Union[int, float]) -> None:
    """
    Удаляет из директории файлы и поддиректории старше max_age_hours.
    """
    now = datetime.now()
    cutoff = now - timedelta(hours=max_age_hours)
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        return
    for f in d.iterdir():
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f, ignore_errors=True)
        except Exception:
            continue

def create_task_metadata(
    task_id: str,
    file_path: str,
    filename: str,
    priority: int,
    client_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Формирует метаданные задачи для QueueManager.
    """
    meta: Dict[str, Any] = {
        "task_id": task_id,
        "file_path": file_path,
        "filename": filename,
        "priority": priority,
    }
    if client_id is not None:
        meta["client_id"] = client_id
    return meta

def remove_intermediate_files(temp_dir: str):
    """
    Удаляет все файлы предобработки (*_processed.wav) и содержимое temp_dir.
    """
    for root, _, files in os.walk(temp_dir, topdown=False):
        for f in files:
            if f.endswith("_processed.wav"):
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

def setup_logging(cfg: AppSettings):
    """
    Настройка логирования через dictConfig.
    Если cfg['file'] задан (не None), логируется по этому пути и имени файла.
    Иначе — в <VOLUME_PATH>/logs/callannotate.log.
    Поддерживается ротация.
    """
    #volume = os.getenv("VOLUME_PATH", "./volume")

    volume = Path(cfg.queue.volume_path).expanduser().resolve()

    # выбор файла для логов
    if cfg.logging.file:
        log_file = cfg.logging.file
    else:
        # лог в едином файле callannotate.log под volume/logs
        log_file = os.path.join(volume, "logs", "callannotate.log")
        cfg.logging.file = log_file

    # создаём директорию
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    handlers = ["console"]
    cfg_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": cfg.logging.format}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": cfg.logging.level.upper()
            }
        },
        "root": {
            "level": cfg.logging.level.upper(),
            "handlers": handlers.copy()
        }
    }

    # File handler с ротацией
    if cfg.logging.rotation.enabled:
        # TimedRotatingFileHandler
        cfg_dict["handlers"]["file"] = {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "default",
            "level": cfg.logging.level.upper(),
            "filename": log_file,
            "when": cfg.logging.rotation.when,
            "interval": cfg.logging.rotation.interval,
            "backupCount": cfg.logging.rotation.backup_count,
            "encoding": "utf-8"
        }
        cfg_dict["root"]["handlers"].append("file")
    else:
        # обычный FileHandler
        cfg_dict["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": cfg.logging.level.upper(),
            "filename": log_file,
            "encoding": "utf-8"
        }
        cfg_dict["root"]["handlers"].append("file")

    logging.config.dictConfig(cfg_dict)

def extract_audio_metadata(path: str) -> AudioMetadata:
    from .schemas import AudioMetadata
    p = Path(path)
    stat = p.stat()
    try:
        audio, sample_rate = librosa.load(path, sr=None)
        duration = len(audio) / sample_rate
        channels = 1
    except Exception:
        sample_rate = 16000
        channels = 1
        duration = stat.st_size / (16000 * 1 * 2)

    bitrate = int(stat.st_size * 8 / duration) if duration > 0 else None
    
    return AudioMetadata(
        filename=p.name,
        duration=round(duration, 2),
        sample_rate=int(sample_rate),
        channels=channels,
        format=p.suffix.lstrip("."),
        bitrate=bitrate,
        size_bytes=stat.st_size,
    )

def ensure_volume_structure(volume_path: str) -> None:
    logger = logging.getLogger(__name__)
    base = Path(volume_path).expanduser().resolve()
    subdirs = [
        "incoming",
        "processing",
        "completed",
        "failed",
        "logs",
        "models/embeddings",
        "temp",
    ]
    base.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured volume base directory: {base}")
    for sub in subdirs:
        path = base / sub
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created subdirectory: {path}")