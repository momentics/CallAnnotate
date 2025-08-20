# -*- coding: utf-8 -*-
"""
mix.py — склейка чанков с линейным или оконным смешиванием.
"""
from __future__ import annotations

from typing import List
import numpy as np


def merge_chunks_linear(chunks: List[np.ndarray], overlap_samples: int) -> np.ndarray:
    """
    Простая линейная склейка: от второго чанка отбрасывается перекрытие.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1 or overlap_samples <= 0:
        return chunks[0]
    out = chunks
    for c in chunks[1:]:
        if len(c) <= overlap_samples:
            out = np.concatenate([out, c])
        else:
            out = np.concatenate([out, c[overlap_samples:]])
    return out.astype(np.float32, copy=False) # type: ignore


def merge_chunks_windowed(chunks: List[np.ndarray], overlap_samples: int) -> np.ndarray:
    """
    Качественная склейка с Хэннинг-окном в зоне перекрытия.
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1 or overlap_samples <= 0:
        return chunks # type: ignore

    out = chunks
    hann = np.hanning(max(2, overlap_samples * 2)).astype(np.float32, copy=False)
    win_a = hann[:overlap_samples]
    win_b = hann[overlap_samples:]

    for c in chunks[1:]:
        if len(c) <= overlap_samples or len(out) <= overlap_samples:
            out = np.concatenate([out, c])
            continue
        end_a = out[-overlap_samples:].copy()
        beg_b = c[:overlap_samples].copy()
        out[-overlap_samples:] = end_a * win_a
        c[:overlap_samples] = beg_b * win_b
        out = np.concatenate([out, c[overlap_samples:]])
    return out.astype(np.float32, copy=False) # type: ignore
