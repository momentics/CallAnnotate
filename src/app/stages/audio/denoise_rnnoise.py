# -*- coding: utf-8 -*-
"""
denoise_rnnoise.py — интеграция RNNoise с мягким fallback.
Работает на 48kHz/mono. При несовпадении частоты делается up/downsample.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from pydub import AudioSegment

from app.rnnoise_wrapper import RNNoise
from .io import to_mono_float32, from_mono_float32, resample_float


RN_SR = 48_000


class RNNoiseWrapper:
    """
    Обёртка, безопасная к отсутствию librnnoise:
    - если библиотека недоступна — фильтр становится passthrough.
    """
    def __init__(self, enabled: bool, allow_passthrough: bool = True):
        self.enabled = enabled
        self._rn: Optional[RNNoise] = None
        if not self.enabled:
            return
        try:
            self._rn = RNNoise(sample_rate=RN_SR, allow_passthrough=allow_passthrough)
        except Exception:
            # fallback
            self._rn = None
            self.enabled = False

    def filter_segment(self, seg: AudioSegment) -> AudioSegment:
        """Фильтр RNNoise на одном сегменте. При недоступности — passthrough."""
        if not self.enabled or self._rn is None:
            return seg

        # Быстрая дорожка: если есть метод filter()
        try:
            filtered = self._rn.filter(seg.set_frame_rate(RN_SR).set_channels(1))
            # Вернём rate/каналы обратно pydub-методами при необходимости в вызывающем коде
            return filtered
        except Exception:
            # Ручная потоковая обработка
            mono = to_mono_float32(seg)
            sr = seg.frame_rate
            work = mono
            if sr != RN_SR:
                work = resample_float(mono, sr, RN_SR)
            frames = []
            try:
                for _, frm in self._rn.denoise_chunk(work):  # type: ignore
                    frames.append(frm.astype(np.float32, copy=False))
            except Exception:
                return seg
            if not frames:
                return seg
            den = np.concatenate(frames)
            if RN_SR != sr:
                den = resample_float(den, RN_SR, sr)
            return from_mono_float32(den, seg)
