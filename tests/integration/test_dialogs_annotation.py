# tests/test_dialogs_annotation.py

import asyncio
import json
from pathlib import Path
import time

import pytest

from app.config import load_settings
from app.queue.manager import AsyncQueueManager
from app.utils import ensure_volume_structure
from app.schemas import AnnotationResult
from app.queue.manager import TaskStatus

# --------------------------------------------------------------------------- #
#  Autouse fixture: создаём тестовый volume и подключаем реальные fixtures    #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def ensure_fixtures_volume(tmp_path_factory, monkeypatch):
    """
    Создание изолированного каталога /volume для тестов и настройка пути,
    чтобы реальная директория с fixtures (tests/fixtures) копировалась внутрь.
    """
    # создаём временную директорию volume
    vol_dir = tmp_path_factory.mktemp("volume")
    # монтируем fixtures внутрь volume/incoming
    fixtures_src = Path(__file__).parent.parent / "fixtures"
    incoming = vol_dir / "incoming"

    incoming.mkdir(parents=True, exist_ok=True)
    # копируем все WAV из fixtures
    for wav in fixtures_src.glob("*.wav"):
        dest = incoming / wav.name
        dest.write_bytes(wav.read_bytes())
    monkeypatch.setenv("VOLUME_PATH", str(vol_dir))

    ensure_volume_structure(str(vol_dir))
    return vol_dir

@pytest.mark.asyncio
async def test_dialog_01_annotation(ensure_fixtures_volume):
    """
    End-to-end annotation для реального WAV из fixtures/Dialog_01.wav.
    Проверяем, что после обработки:
      - В volume/incoming появился processed файл
      - В volume/completed находится оригинальный WAV
      - Рядом в volume/completed лежит JSON аннотации
    """
    settings = load_settings()
    vol = ensure_fixtures_volume
    settings.queue.volume_path = str(vol)

    # Запускаем очередь и воркеры
    queue_mgr = AsyncQueueManager(settings)
    await queue_mgr.start()

    #---------------------------------------------    

    wav_name = "Dialog_01.wav"
    incoming = vol / "incoming" / wav_name
    assert incoming.exists(), f"{incoming} не найден"

    # Добавляем задачу на обработку файла
    job_id = "job-dialog-01"
    await queue_mgr.add_task(job_id, {
        "file_path": str(incoming),
        "filename": wav_name,
        "priority": 5
    })

    # Ждём, чтобы очередь подобрала задачу и сгенерировала processed файл
    # Ждём завершения задачи с таймаутом 60 секунд
    timeout = 60
    start_time = time.time()
    while True:
        tr = await queue_mgr.get_task_result(job_id)
        if tr and tr.status == TaskStatus.COMPLETED:
            break
        if tr and tr.status == TaskStatus.FAILED:
            pytest.fail(f"Task {job_id} failed with error {tr.error}")   # type: ignore[attr-defined]
        if time.time() - start_time > timeout:
            pytest.fail("Timeout waiting for task to complete")
        await asyncio.sleep(0.5)

    # Проверяем, что в incoming нет файла
    assert not (vol / "incoming" / "Dialog_01_processed.wav").exists(), \
        "Файл остался в incoming"

    # Проверяем, что в incoming нет файла
    assert not (vol / "incoming" / "Dialog_01.wav").exists(), \
        "Файл остался в incoming"

    # Проверяем, что оригинал перемещён в completed
    orig_completed = vol / "completed" / "Dialog_01.wav"
    assert orig_completed.exists(), "Оригинальный WAV не перемещён в completed"

    # Проверяем наличие JSON-аннотации рядом с оригиналом
    json_path = vol / "completed" / "Dialog_01.json"
    assert json_path.exists(), "JSON аннотации не найден в completed"

    # Загружаем результат и проверяем содержимое
    result_dict = json.loads(json_path.read_text(encoding="utf-8"))
    ann = AnnotationResult(**result_dict)

    #assert ann.task_id == job_id
    #assert ann.statistics.total_segments > 0
    #assert ann.statistics.total_words > 0
    #assert ann.transcription.full_text



    #-----------------------------------------------------









    wav_name = "DialogLong_01.wav"
    incoming = vol / "incoming" / wav_name
    assert incoming.exists(), f"{incoming} не найден"

    # Добавляем задачу на обработку файла
    job_id = "job-dialog-02"
    await queue_mgr.add_task(job_id, {
        "file_path": str(incoming),
        "filename": wav_name,
        "priority": 5
    })

    # Ждём, чтобы очередь подобрала задачу и сгенерировала processed файл
    # Ждём завершения задачи с таймаутом 60 секунд
    timeout = 60
    start_time = time.time()
    while True:
        tr = await queue_mgr.get_task_result(job_id)
        if tr and tr.status == TaskStatus.COMPLETED:
            break
        if tr and tr.status == TaskStatus.FAILED:
            pytest.fail(f"Task {job_id} failed with error {tr.error}")    # type: ignore[attr-defined]
        if time.time() - start_time > timeout:
            pytest.fail("Timeout waiting for task to complete")
        await asyncio.sleep(0.5)

    # Проверяем, что в incoming нет файла
    assert not (vol / "incoming" / "DialogLong_01_processed.wav").exists(), \
        "Файл остался в incoming"

    # Проверяем, что в incoming нет файла
    assert not (vol / "incoming" / "DialogLong_01.wav").exists(), \
        "Файл остался в incoming"

    # Проверяем, что оригинал перемещён в completed
    orig_completed = vol / "completed" / "DialogLong_01.wav"
    assert orig_completed.exists(), "Оригинальный WAV не перемещён в completed"

    # Проверяем наличие JSON-аннотации рядом с оригиналом
    json_path = vol / "completed" / "DialogLong_01.json"
    assert json_path.exists(), "JSON аннотации не найден в completed"

    # Загружаем результат и проверяем содержимое
    result_dict = json.loads(json_path.read_text(encoding="utf-8"))
    ann = AnnotationResult(**result_dict)

    #assert ann.task_id == job_id
    #assert ann.statistics.total_segments > 0
    #assert ann.statistics.total_words > 0
    #assert ann.transcription.full_text

    # Graceful shutdown of the queue manager
    await queue_mgr.stop()
