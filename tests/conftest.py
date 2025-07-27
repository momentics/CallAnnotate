# tests/conftest.py
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from fastapi.testclient import TestClient

# Добавляем src в PATH для корректного импорта приложения
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from app.app import app

GLOBAL_ACTIVE_JOB_IDS = set()

@pytest.fixture(scope="session")
def mock_queue_manager_and_deps():
    mock_task_result = MagicMock()
    mock_task_result.status = "completed"
    mock_task_result.message = "Task completed successfully"
    mock_task_result.result = {"test": "data"}
    mock_task_result.error = None
    mock_task_result.created_at = "2025-07-27T20:00:00Z"
    mock_task_result.started_at = "2025-07-27T20:00:05Z"
    mock_task_result.completed_at = "2025-07-27T20:02:30Z"
    mock_task_result.file_info = {"filename": "test.wav", "path": "/test/path"}
    mock_task_result.progress = {
        "percentage": 0,
        "current_stage": "queued",
        "stage_progress": 0,
        "stages_completed": [],
        "stages_remaining": [],
        "estimated_completion": None
    }

    async def add_task(job_id, metadata):
        GLOBAL_ACTIVE_JOB_IDS.add(job_id)

    async def cancel_task(job_id):
        if job_id in GLOBAL_ACTIVE_JOB_IDS:
            GLOBAL_ACTIVE_JOB_IDS.remove(job_id)
            return True
        return False

    async def get_task_result(job_id):
        if job_id in GLOBAL_ACTIVE_JOB_IDS:
            return mock_task_result
        return None

    mock_queue_manager = AsyncMock()
    mock_queue_manager.get_queue_size.return_value = 0
    mock_queue_manager.get_active_tasks_count.return_value = 0
    mock_queue_manager.subscribe_to_task.return_value = None
    mock_queue_manager.start.return_value = None
    mock_queue_manager.stop.return_value = None
    mock_queue_manager.get_queue_info.return_value = {
        "queue_length": 0,
        "processing_jobs": [],
        "queued_jobs": [],
        "average_processing_time": 0
    }
    mock_queue_manager.add_task.side_effect = add_task
    mock_queue_manager.cancel_task.side_effect = cancel_task
    mock_queue_manager.get_task_result.side_effect = get_task_result

    mock_annotation_service = AsyncMock()
    mock_annotation_service.process_audio.return_value = {"test": "annotation_result"}

    from app.app import app
    app.state.queue_manager = mock_queue_manager
    app.state.annotation_service = mock_annotation_service
    return app

@pytest.fixture
def client():
    # Один set для текущего теста
    active_job_ids = set()
    mock_task_result = MagicMock()
    mock_task_result.status = "completed"
    mock_task_result.message = "Task completed successfully"
    mock_task_result.result = {"test": "data"}
    mock_task_result.error = None
    mock_task_result.created_at = "2025-07-27T20:00:00Z"
    mock_task_result.started_at = "2025-07-27T20:00:05Z"
    mock_task_result.completed_at = "2025-07-27T20:02:30Z"
    mock_task_result.file_info = {"filename": "test.wav", "path": "/test/path"}
    mock_task_result.progress = {
        "percentage": 0,
        "current_stage": "queued",
        "stage_progress": 0,
        "stages_completed": [],
        "stages_remaining": [],
        "estimated_completion": None
    }

    async def add_task(job_id, metadata):
        print(f"[mock] add_task {job_id}")
        active_job_ids.add(job_id)

    async def cancel_task(job_id):
        print(f"[mock] cancel_task {job_id} in {active_job_ids}")
        if job_id in active_job_ids:
            active_job_ids.remove(job_id)
            return True
        return False

    async def get_task_result(job_id):
        print(f"[mock] get_task_result {job_id} in {active_job_ids}")
        if job_id in active_job_ids:
            return mock_task_result
        return None

    mock_queue_manager = AsyncMock()
    mock_queue_manager.get_queue_size.return_value = 0
    mock_queue_manager.get_active_tasks_count.return_value = 0
    mock_queue_manager.subscribe_to_task.return_value = None
    mock_queue_manager.start.return_value = None
    mock_queue_manager.stop.return_value = None
    mock_queue_manager.get_queue_info.return_value = {
        "queue_length": 0,
        "processing_jobs": [],
        "queued_jobs": [],
        "average_processing_time": 0
    }
    mock_queue_manager.add_task.side_effect = add_task
    mock_queue_manager.cancel_task.side_effect = cancel_task
    mock_queue_manager.get_task_result.side_effect = get_task_result

    mock_annotation_service = AsyncMock()
    mock_annotation_service.process_audio.return_value = {"test": "annotation_result"}

    # КРИТИЧЕСКОЕ МЕСТО: патчим состояние приложения до клиента!
    app.state.queue_manager = mock_queue_manager
    app.state.annotation_service = mock_annotation_service

    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Настройка тестового окружения"""
    # Создаем временные директории для тестов
    test_volume_path = Path(__file__).parent / "test_volume"
    test_volume_path.mkdir(exist_ok=True)
    
    # Создаем необходимые поддиректории
    for subdir in ["incoming", "processing", "completed", "failed", "logs/system"]:
        (test_volume_path / subdir).mkdir(parents=True, exist_ok=True)
    
    # Устанавливаем переменные окружения для тестов
    os.environ["VOLUME_PATH"] = str(test_volume_path)
    os.environ["MAX_FILE_SIZE"] = "10485760"  # 10MB для тестов
    os.environ["LOG_LEVEL"] = "DEBUG" 
    
    yield
    
    # Очистка после тестов
    import shutil
    if test_volume_path.exists():
        shutil.rmtree(test_volume_path, ignore_errors=True)




@pytest.fixture
def sample_audio_file(tmp_path):
    """Создание тестового аудиофайла"""
    audio_file = tmp_path / "sample.wav"
    # Минимальный валидный WAV заголовок
    wav_header = (
        b'RIFF' +
        (44 - 8).to_bytes(4, 'little') +  # Размер файла - 8
        b'WAVE' +
        b'fmt ' +
        (16).to_bytes(4, 'little') +      # Размер fmt chunk
        (1).to_bytes(2, 'little') +       # Audio format (PCM)
        (1).to_bytes(2, 'little') +       # Num channels
        (44100).to_bytes(4, 'little') +   # Sample rate
        (88200).to_bytes(4, 'little') +   # Byte rate
        (2).to_bytes(2, 'little') +       # Block align
        (16).to_bytes(2, 'little') +      # Bits per sample
        b'data' +
        (0).to_bytes(4, 'little')         # Data size
    )
    audio_file.write_bytes(wav_header)
    return audio_file


@pytest.fixture
def mock_successful_task_result():
    """Фикстура успешного результата задачи"""
    from app.queue_manager import TaskStatus
    
    result = MagicMock()
    result.status = TaskStatus.COMPLETED
    result.message = "Task completed successfully"
    result.result = {
        "task_id": "test-job-id",
        "speakers": [{"id": "speaker_01", "name": "Test Speaker"}],
        "segments": [{"start": 0.0, "end": 5.0, "text": "Test transcription"}],
        "statistics": {"total_speakers": 1, "total_segments": 1}
    }
    result.error = None
    return result


@pytest.fixture
def mock_failed_task_result():
    """Фикстура неудачного результата задачи"""
    from app.queue_manager import TaskStatus
    
    result = MagicMock()
    result.status = TaskStatus.FAILED
    result.message = "Task failed"
    result.result = None
    result.error = "Processing error occurred"
    return result


@pytest.fixture
def mock_processing_task_result():
    """Фикстура задачи в процессе обработки"""
    from app.queue_manager import TaskStatus
    
    result = MagicMock()
    result.status = TaskStatus.PROCESSING
    result.message = "Task is being processed"
    result.result = None
    result.error = None
    result.progress = {
        "percentage": 45,
        "current_stage": "transcription"
    }
    return result
