# -*- coding: utf-8 -*-
"""
io.py — функции ввода/вывода и конвертации аудио для предобработки.

Основные принципы:
- Опора на pydub для унификации чтения различных форматов.
- Внутренний формат для DSP — numpy float32 в диапазоне [-1.0..1.0], моно.
- Аккуратные преобразования частоты дискретизации (scipy.signal.resample_poly).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from pydub import AudioSegment
from scipy.signal import resample_poly


def load_audio_segment(path: str) -> AudioSegment:
    """
    Безопасная загрузка файла аудио. Pydub сам выберет backend (ffmpeg).
    В случае проблем — возбуждается исключение.
    """
    return AudioSegment.from_file(path)


def to_mono_float32(seg: AudioSegment) -> np.ndarray:
    """
    Преобразование pydub.AudioSegment -> np.ndarray float32 моно [-1..1].
    """
    s = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels > 1:
        s = s.reshape(-1, seg.channels).mean(axis=1)
    max_i = float(np.iinfo(seg.array_type).max)
    if max_i <= 0:
        max_i = 1.0
    s = np.clip(s / max_i, -1.0, 1.0)
    return s.astype(np.float32, copy=False)


def from_mono_float32(x: np.ndarray, like: AudioSegment) -> AudioSegment:
    """
    Обратная конвертация float32 моно [-1..1] в AudioSegment «как like».
    """
    x = np.clip(x, -1.0, 1.0)
    max_i = float(np.iinfo(like.array_type).max)
    pcm = (x * max_i).astype(like.array_type, copy=False)
    seg = AudioSegment(
        pcm.tobytes(),
        frame_rate=like.frame_rate,
        sample_width=like.sample_width,
        channels=1,
    )
    if like.channels > 1:
        seg = seg.set_channels(like.channels)
    return seg


def ensure_channels_and_rate(seg: AudioSegment, channels: str, sample_rate_target: Optional[int]) -> AudioSegment:
    """
    Нормализует канальность (mono|stereo|original) и частоту дискретизации.
    """
    # Каналы
    if channels == "mono" and seg.channels > 1:
        seg = seg.set_channels(1)
    elif channels == "stereo" and seg.channels == 1:
        seg = seg.set_channels(2)
    # Частота дискретизации
    if isinstance(sample_rate_target, int) and sample_rate_target > 0:
        seg = seg.set_frame_rate(sample_rate_target)
    return seg


def resample_float(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Пересэмплинг через resample_poly (высокое качество, разумная скорость).
    Ничего не делает, если target_sr == orig_sr.
    """
    if target_sr == orig_sr:
        return x
    if target_sr <= 0 or orig_sr <= 0 or x.size == 0:
        return x
    g = np.gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    y = resample_poly(x, up, down)
    return y.astype(np.float32, copy=False)
