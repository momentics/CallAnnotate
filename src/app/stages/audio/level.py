# -*- coding: utf-8 -*-
"""
level.py — измерение и нормализация RMS-уровня.
"""
from __future__ import annotations

import numpy as np


def rms_float(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float32))))


def apply_rms_ceiling(x: np.ndarray, target_rms_db: float) -> np.ndarray:
    """
    Понижаем RMS до целевого, не увеличивая слабые сигналы (во избежание поднятия шума).
    """
    if x.size == 0:
        return x
    cur = rms_float(x)
    if cur <= 0.0:
        return x
    target_lin = 10.0 ** (target_rms_db / 20.0)
    if cur > target_lin:
        gain = target_lin / cur
        return np.clip(x * gain, -1.0, 1.0).astype(np.float32, copy=False)
    return x
