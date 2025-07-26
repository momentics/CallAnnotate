# -*- coding: utf-8 -*-
"""
Утилиты для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from fastapi import UploadFile
import librosa


@dataclass
class AudioMetadata:
    """Метаданные аудиофайла"""
    filename: str
    duration: float
    sample_rate: int
    channels: int
    format: str
    bitrate: Optional[int] = None
    size_bytes: int = 0


@dataclass
class ValidationResult:
    """Результат валидации файла"""
    is_valid: bool
    error: Optional[str] = None
    metadata: Optional[AudioMetadata] = None


def setup_logging(config: Dict[str, Any]):
    """Настройка системы логирования"""
    log_level = config.get('level', 'INFO').upper()
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Базовая настройка
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # Настройка файлового логирования
    log_file = config.get('file')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Добавление к root logger
        logging.getLogger().addHandler(file_handler)
    
    # Настройка уровней для сторонних библиотек
    external_loggers = config.get('external_levels', {})
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))


def validate_audio_file(file: UploadFile) -> ValidationResult:
    """
    Валидация загруженного аудиофайла
    
    Args:
        file: Загруженный файл
        
    Returns:
        Результат валидации
    """
    try:
        # Проверка MIME типа
        allowed_mime_types = [
            'audio/wav', 'audio/wave', 'audio/x-wav',
            'audio/mpeg', 'audio/mp3',
            'audio/ogg', 'audio/x-ogg-audio',
            'audio/flac', 'audio/x-flac',
            'audio/aac', 'audio/x-aac',
            'audio/m4a', 'audio/mp4'
        ]
        
        mime_type, _ = mimetypes.guess_type(file.filename)
        if mime_type not in allowed_mime_types:
            return ValidationResult(
                is_valid=False,
                error=f"Неподдерживаемый тип файла: {mime_type}"
            )
        
        # Проверка размера файла (максимум 500MB)
        max_size = 500 * 1024 * 1024  # 500MB
        if hasattr(file, 'size') and file.size > max_size:
            return ValidationResult(
                is_valid=False,
                error=f"Файл слишком большой: {file.size} байт (максимум {max_size})"
            )
        
        # Проверка расширения файла
        allowed_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a', '.mp4'}
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            return ValidationResult(
                is_valid=False,
                error=f"Неподдерживаемое расширение файла: {file_extension}"
            )
        
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error=f"Ошибка при валидации файла: {str(e)}"
        )


def extract_audio_metadata(file_path: str) -> AudioMetadata:
    """
    Извлечение метаданных из аудиофайла
    
    Args:
        file_path: Путь к аудиофайлу
        
    Returns:
        Метаданные аудиофайла
    """
    try:
        file_path_obj = Path(file_path)
        
        # Загрузка аудио с помощью librosa
        y, sr = librosa.load(file_path, sr=None, mono=False)
        
        # Определение количества каналов
        if y.ndim == 1:
            channels = 1
        else:
            channels = y.shape[0]
        
        # Длительность в секундах
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Размер файла
        size_bytes = file_path_obj.stat().st_size
        
        # Формат файла
        file_format = file_path_obj.suffix.lower().lstrip('.')
        
        # Приблизительный битрейт
        bitrate = None
        if duration > 0:
            bitrate = int((size_bytes * 8) / duration)
        
        return AudioMetadata(
            filename=file_path_obj.name,
            duration=duration,
            sample_rate=sr,
            channels=channels,
            format=file_format,
            bitrate=bitrate,
            size_bytes=size_bytes
        )
        
    except Exception as e:
        # Возвращаем базовые метаданные при ошибке
        file_path_obj = Path(file_path)
        return AudioMetadata(
            filename=file_path_obj.name,
            duration=0.0,
            sample_rate=0,
            channels=0,
            format=file_path_obj.suffix.lower().lstrip('.'),
            size_bytes=file_path_obj.stat().st_size if file_path_obj.exists() else 0
        )


def create_task_metadata(
    task_id: str,
    file_path: str,
    filename: str,
    priority: int = 5,
    callback_url: Optional[str] = None,
    websocket_client_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Создание метаданных для задачи
    
    Args:
        task_id: Идентификатор задачи
        file_path: Путь к файлу
        filename: Имя файла
        priority: Приоритет задачи
        callback_url: URL для callback уведомлений
        websocket_client_id: ID WebSocket клиента
        **kwargs: Дополнительные параметры
        
    Returns:
        Словарь с метаданными задачи
    """
    metadata = {
        "task_id": task_id,
        "file_path": file_path,
        "filename": filename,
        "priority": priority,
        "created_at": datetime.now().isoformat(),
        "callback_url": callback_url,
        "websocket_client_id": websocket_client_id,
        "options": kwargs
    }
    
    return metadata


def format_duration(seconds: float) -> str:
    """
    Форматирование длительности в человекочитаемый вид
    
    Args:
        seconds: Длительность в секундах
        
    Returns:
        Форматированная строка
    """
    if seconds < 60:
        return f"{seconds:.1f}с"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}м {secs:.1f}с"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}ч {minutes}м {secs:.1f}с"


def format_file_size(size_bytes: int) -> str:
    """
    Форматирование размера файла в человекочитаемый вид
    
    Args:
        size_bytes: Размер в байтах
        
    Returns:
        Форматированная строка
    """
    if size_bytes < 1024:
        return f"{size_bytes} Б"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} КБ"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} МБ"
    else:
        return f"{size_bytes / (1024 ** 3):.1f} ГБ"


def ensure_directory(path: str) -> Path:
    """
    Обеспечение существования директории
    
    Args:
        path: Путь к директории
        
    Returns:
        Объект Path
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def cleanup_temp_files(directory: str, older_than_hours: int = 24):
    """
    Очистка временных файлов старше определенного возраста
    
    Args:
        directory: Директория для очистки
        older_than_hours: Возраст файлов в часах
    """
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    logging.info(f"Удален временный файл: {file_path}")
                except Exception as e:
                    logging.error(f"Ошибка при удалении файла {file_path}: {e}")
                    
    except Exception as e:
        logging.error(f"Ошибка при очистке временных файлов в {directory}: {e}")


class ProgressTracker:
    """Отслеживание прогресса выполнения задач"""
    
    def __init__(self, total_steps: int = 100):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = datetime.now()
        self.callbacks = []
    
    def add_callback(self, callback):
        """Добавление callback функции"""
        self.callbacks.append(callback)
    
    async def update(self, step: int, message: str = ""):
        """Обновление прогресса"""
        self.current_step = step
        progress_percent = min(100, int((step / self.total_steps) * 100))
        
        for callback in self.callbacks:
            try:
                await callback(progress_percent, message)
            except Exception as e:
                logging.error(f"Ошибка в callback прогресса: {e}")
    
    def get_elapsed_time(self) -> float:
        """Получение времени выполнения"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def estimate_remaining_time(self) -> Optional[float]:
        """Оценка оставшегося времени"""
        if self.current_step == 0:
            return None
        
        elapsed = self.get_elapsed_time()
        progress_ratio = self.current_step / self.total_steps
        
        if progress_ratio > 0:
            total_estimated = elapsed / progress_ratio
            return total_estimated - elapsed
        
        return None
