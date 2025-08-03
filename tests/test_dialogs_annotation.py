# tests/test_dialogs_annotation.py

import asyncio
import os
import pytest
from pathlib import Path

from app.annotation import AnnotationService
from app.config import load_settings
from app.schemas import AnnotationResult

@pytest.fixture(autouse=True)
def ensure_volume(tmp_path_factory, monkeypatch):
    """
    Override VOLUME_PATH to use a temporary directory for any queue operations
    and ensure the volume structure is created.
    """
    vol_dir = tmp_path_factory.mktemp("volume")
    monkeypatch.setenv("VOLUME_PATH", str(vol_dir))
    # ensure volume structure on startup
    from app.utils import ensure_volume_structure
    ensure_volume_structure(str(vol_dir))
    return vol_dir

@pytest.mark.asyncio
async def test_dialog_01_annotation():
    """
    End-to-end annotation for Dialog_01.wav using AnnotationService directly.
    """
    # Path to fixture audio
    fixture_path = Path(__file__).parent / "fixtures" / "Dialog_01.wav"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    # Load settings and instantiate service
    settings = load_settings()
    service = AnnotationService(settings.dict())

    # Process audio
    result_dict = await service.process_audio(
        str(fixture_path),
        task_id="test-dialog-01",
    )
    ann = AnnotationResult(**result_dict)

    # Basic assertions
    assert ann.task_id == "test-dialog-01"
    assert ann.statistics.total_segments > 0
    assert ann.statistics.total_words > 0
    # Ensure transcription exists
    assert ann.transcription.full_text

@pytest.mark.asyncio
async def test_dialoglong_01_annotation():
    """
    End-to-end annotation for DialogLong_01.wav using AnnotationService directly.
    """

    fixture_path = Path(__file__).parent / "fixtures" / "DialogLong_01.wav"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    settings = load_settings()
    service = AnnotationService(settings.dict())

    result_dict = await service.process_audio(
        str(fixture_path),
        task_id="test-dialoglong-01",
    )
    ann = AnnotationResult(**result_dict)

    assert ann.task_id == "test-dialoglong-01"
    assert ann.statistics.total_segments > 0
    assert ann.statistics.total_words > 0
    assert ann.transcription.full_text
