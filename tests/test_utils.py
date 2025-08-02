# -*- coding: utf-8 -*-
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import os
import pytest
from fastapi import UploadFile
from io import BytesIO

from app.utils import (
    validate_audio_file,
    extract_audio_metadata,
    ensure_directory,
    cleanup_temp_files,
    get_supported_audio_formats,
)


def make_upload(filename: str, content_type: str) -> UploadFile:
    """
    Создаёт UploadFile с указанием заголовков, чтобы content_type 
    был корректно рассчитан из headers.
    """
    # Параметр headers ожидает list of tuples или Mapping
    headers = {"content-type": content_type}
    return UploadFile(file=BytesIO(b"data"), filename=filename, headers=headers)


def test_validate_audio_file_ok():
    upload = make_upload("test.wav", "audio/wav")
    result = validate_audio_file(upload)
    assert result.is_valid


def test_validate_audio_file_bad_ext():
    upload = make_upload("test.txt", "text/plain")
    result = validate_audio_file(upload)
    assert not result.is_valid
    assert "unsupported" in result.error.lower() or "неподдерж" in result.error.lower()


def test_extract_audio_metadata(tmp_path, monkeypatch):
    # создаём пустой WAV файл
    path = tmp_path / "test.wav"
    path.write_bytes(b"\x00" * 1000)
    import app.utils as utils
    # мокаем librosa.load
    def fake_load(fp, sr=None):
        return [0.0, 0.0], 2
    monkeypatch.setattr(utils.librosa, "load", fake_load)
    md = extract_audio_metadata(str(path))
    assert md.filename == "test.wav"
    assert md.sample_rate == 2
    assert md.duration == pytest.approx(1.0)
    assert md.channels == 1

def test_ensure_directory_and_cleanup(tmp_path):
    d = tmp_path / "subdir"
    p = ensure_directory(str(d))
    assert p.exists()
    old = d / "old.txt"
    old.write_text("x")
    # делаем файл «старым»
    old_time = old.stat().st_mtime - 3600 * 25
    os.utime(old, (old_time, old_time))
    cleanup_temp_files(str(d), max_age_hours=24)
    assert not old.exists()


def test_get_supported_audio_formats():
    fm = get_supported_audio_formats()
    assert ".wav" in fm
    assert fm[".wav"] == "audio/wav"

