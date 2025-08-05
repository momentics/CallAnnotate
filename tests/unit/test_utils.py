# tests/unit/test_utils.py

import os
import shutil
import pytest
import numpy as np
from pathlib import Path
from app.utils import (
    get_supported_audio_formats,
    validate_audio_file_path,
    ensure_directory,
    cleanup_temp_files,
    extract_audio_metadata,
)

def test_get_supported_formats():
    fmts = get_supported_audio_formats()
    assert ".wav" in fmts and fmts[".wav"] == "audio/wav"
    assert all(k.startswith(".") for k in fmts.keys())

def test_validate_audio_file_path(tmp_path):
    f = tmp_path / "test.mp3"
    f.write_bytes(b"dummy")
    res = validate_audio_file_path(str(f))
    assert res.is_valid
    bad = validate_audio_file_path(str(tmp_path / "nope.wav"))
    assert not bad.is_valid

def test_ensure_directory(tmp_path):
    d = tmp_path / "subdir"
    result = ensure_directory(str(d))
    assert d.exists() and d.is_dir()
    # idempotent
    result2 = ensure_directory(str(d))
    assert result2 == d

def test_cleanup_temp_files(tmp_path):
    d = tmp_path / "temp"
    d.mkdir()
    old = d / "old.txt"
    recent = d / "new.txt"
    old.write_text("x")
    recent.write_text("y")
    # set old mtime to 48 hours ago
    old_time = (tmp_path.stat().st_mtime - 48*3600)
    os.utime(old, (old_time, old_time))
    cleanup_temp_files(str(d), max_age_hours=24)
    assert not old.exists()
    assert recent.exists()

def test_extract_audio_metadata(tmp_path, monkeypatch):
    # create a small WAV via numpy
    path = tmp_path / "tone.wav"
    sr = 8000
    t = np.linspace(0, 0.1, int(0.1*sr), False)
    data = (0.5*np.sin(2*np.pi*440*t)).astype(np.float32)
    import soundfile as sf
    sf.write(str(path), data, sr)
    meta = extract_audio_metadata(str(path))
    assert meta.filename == "tone.wav"
    assert meta.duration > 0
    assert meta.sample_rate == sr
    assert meta.channels == 1
    assert meta.size_bytes > 0
