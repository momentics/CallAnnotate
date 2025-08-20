# -*- coding: utf-8 -*-
"""
chunking.py — разбиение длинных сегментов AudioSegment на перекрывающиеся чанки.
"""
from __future__ import annotations

from typing import Iterator, Tuple
from pydub import AudioSegment


def iter_overlapping_chunks(seg: AudioSegment, chunk_ms: int, overlap_ms: int) -> Iterator[Tuple[int, int, AudioSegment]]:
    """
    Итератор чанков AudioSegment с перекрытием.
    Возвращает (start_ms, end_ms, chunk_seg).
    """
    total = len(seg)
    if chunk_ms <= 0:
        chunk_ms = total
    if overlap_ms < 0:
        overlap_ms = 0
    step = max(1, chunk_ms - overlap_ms)

    pos = 0
    while pos < total:
        end = min(pos + chunk_ms, total)
        yield pos, end, seg[pos:end] # type: ignore
        if end == total:
            break
        pos += step
