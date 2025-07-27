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
    format_duration,
    get_supported_audio_formats,
)


def make_upload(filename, content_type):
    u = UploadFile(file=BytesIO(b"x"*10), filename=filename)
    object.__setattr__(u, "content_type", content_type)
    return u

def test_validate_audio_file_ok():
    assert validate_audio_file(make_upload("ok.wav", "audio/wav")).is_valid

def test_validate_audio_file_bad_ext():
    res = validate_audio_file(make_upload("bad.txt", "text/plain"))
    assert not res.is_valid
    assert "Unsupported" in res.error or "неподдерж" in res.error.lower()


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

def test_format_duration():
    assert format_duration(30) == "30.0с"
    assert format_duration(90) == "1м 30с"
    assert format_duration(3661) == "1ч 1м 1с"

def test_get_supported_audio_formats():
    fm = get_supported_audio_formats()
    assert ".wav" in fm
    assert fm[".wav"] == "audio/wav"

