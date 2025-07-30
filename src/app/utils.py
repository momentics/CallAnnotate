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

from .schemas import AudioMetadata


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
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
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


def extract_audio_metadata(path: str) -> AudioMetadata:
    """
    Извлечение базовых метаданных аудио: размер и реальная длительность через librosa.
    """
    v = validate_audio_file_path(path)
    if not v.is_valid:
        raise FileNotFoundError(v.error)
    p = Path(path)
    stat = p.stat()

    try:
        audio, sample_rate = librosa.load(path, sr=None)
        length = len(audio)
        channels = 1
        duration = length / sample_rate
    except Exception:
        sample_rate = 16000
        channels = 1
        duration = stat.st_size / (16000 * 1 * 2)

    return AudioMetadata(
        filename=p.name,
        duration=round(duration, 2),
        sample_rate=int(sample_rate),
        channels=channels,
        format=p.suffix.lstrip('.'),
        bitrate=None,
        size_bytes=stat.st_size
    )


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


def format_duration(seconds: Union[int, float]) -> str:
    """
    Форматирует длительность: если меньше минуты — "Z.Zс", иначе "Xч Yм Zс".
    """
    total_sec = float(seconds)
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    secs = total_sec - hours * 3600 - minutes * 60
    parts = []
    if hours:
        parts.append(f"{hours}ч")
    if minutes:
        parts.append(f"{minutes}м")
    if hours or minutes:
        parts.append(f"{int(secs)}с")
    else:
        parts.append(f"{secs:.1f}с")
    return " ".join(parts)


def get_human_readable_size(num: int) -> str:
    """Возвращает размер в удобочитаемом формате (KB/MB/GB)"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def setup_logging(cfg: Dict[str, Any]):
    """
    Настройка логирования через dictConfig.
    Создаёт каталоги для файловых хендлеров перед применением конфигурации.
    """
    log_file = cfg.get("file")
    if log_file:
        Path(log_file).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    handlers = ["console"]
    cfg_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": cfg.get("format")}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": cfg.get("level", "INFO").upper()
            }
        },
        "root": {
            "level": cfg.get("level", "INFO").upper(),
            "handlers": handlers
        }
    }
    if log_file:
        cfg_dict["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": cfg.get("level", "INFO").upper(),
            "filename": log_file
        }
        cfg_dict["root"]["handlers"].append("file")

    logging.config.dictConfig(cfg_dict)


def ensure_volume_structure(volume_path: str) -> None:
    """
    Создаёт полную структуру каталогов volume для CallAnnotate.

    Структура:
      volume/
        incoming/
        processing/
        completed/
        failed/
        archived/
        logs/
          system/
          tasks/
        models/
          embeddings/
        temp/
    """
    logger = logging.getLogger(__name__)
    base = Path(volume_path).expanduser().resolve()
    subdirs = [
        "incoming",
        "processing",
        "completed",
        "failed",
        "archived",
        "logs/system",
        "logs/tasks",
        "models/embeddings",
        "temp",
    ]

    try:
        base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Volume base directory ensured: {base}")
        for sub in subdirs:
            path = base / sub
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created subdirectory: {path}")
        logger.info(f"Volume structure created at {base}")
    except Exception as e:
        logger.error(f"Error creating volume structure at {base}: {e}")
        raise


def create_task_metadata(
    task_id: str,
    file_path: str,
    filename: str,
    priority: int,
    websocket_client_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Формирует метаданные для задачи очереди.

    Args:
        task_id: уникальный идентификатор задачи
        file_path: полный путь к исходному аудиофайлу
        filename: имя файла (без пути)
        priority: приоритет задачи (1–10)
        websocket_client_id: идентификатор WS-клиента (если есть)

    Returns:
        Словарь с метаданными задачи.
    """
    meta = {
        "task_id": task_id,
        "file_path": file_path,
        "filename": filename,
        "priority": priority,
        "created_at": datetime.now().isoformat(),
    }
    if websocket_client_id:
        meta["websocket_client_id"] = websocket_client_id
    return meta

