# -*- coding: utf-8 -*-
"""
Лёгкая обёртка над C-библиотекой RNNoise с возможностью
«мягкого» падения, если нативная SO-библиотека недоступна.

• RNNoise.denoise_chunk() ― генератор, возвращающий пары
  (speech_prob, denoised_frame) для последовательных 10-мс фреймов.
• RNNoise.filter() ― удобный метод, принимающий AudioSegment
  (pydub) и возвращающий денойз-копию.

При отсутствии `librnnoise.so` конструктор по умолчанию бросает
RuntimeError.  Для юнит-тестов можно передать
`allow_passthrough=True`, тогда будет использован «noop»-стаб.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from pathlib import Path
from typing import Generator, Tuple, Iterable

import numpy as np

try:
    from pydub import AudioSegment
except ModuleNotFoundError:  # pydub опционален
    AudioSegment = None  # type: ignore

FRAME_SIZE = 480           # 10 мс при 48 кГц
_SAMPLE_RATE = 48_000


class _PassthroughLib:
    """Минимальный стаб, имитирующий C-API RNNoise и делающий «ничего»."""
    def rnnoise_create(self) -> ctypes.c_void_p:                      # noqa
        return ctypes.c_void_p(1)

    def rnnoise_destroy(self, _state: ctypes.c_void_p) -> None:       # noqa
        return None

    def rnnoise_process_frame(self, _state: ctypes.c_void_p,          # noqa
                              out_frame: ctypes.POINTER(ctypes.c_float),
                              in_frame: ctypes.POINTER(ctypes.c_float)
                              ) -> ctypes.c_float:
        ctypes.memmove(out_frame, in_frame,
                       FRAME_SIZE * ctypes.sizeof(ctypes.c_float))
        return ctypes.c_float(0.0)


class RNNoise:  # pylint: disable=too-few-public-methods
    """
    Тонкая Python-обёртка над RNNoise.

    Parameters
    ----------
    sample_rate : int
        Допустимо лишь 48000 Гц.
    allow_passthrough : bool
        True ― не падать при отсутствии SO, использовать no-op реализацию
        (полезно для CI-тестов).
    """

    def __init__(self, sample_rate: int = _SAMPLE_RATE,
                 *, allow_passthrough: bool = False):
        if sample_rate != _SAMPLE_RATE:
            raise ValueError("RNNoise работает только с частотой 48 кГц")
        self.sample_rate = sample_rate

        lib_path = ctypes.util.find_library("rnnoise")
        if lib_path:
            self._lib = ctypes.CDLL(lib_path)
        else:
            for cand in ("/usr/local/lib/librnnoise.so",
                         "/usr/lib/librnnoise.so"):
                if Path(cand).exists():
                    self._lib = ctypes.CDLL(cand)
                    break
            else:
                if not allow_passthrough:
                    raise RuntimeError("Библиотека RNNoise не найдена")
                self._lib = _PassthroughLib()          # type: ignore

        # Настройка прототипов (для _PassthroughLib это no-op)
        for _ in (None,):
            try:
                self._lib.rnnoise_create.restype = ctypes.c_void_p
                self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
                self._lib.rnnoise_process_frame.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                ]
                self._lib.rnnoise_process_frame.restype = ctypes.c_float
            except AttributeError:
                break

        self._state = self._lib.rnnoise_create()
        if not self._state:
            raise RuntimeError("Не удалось создать состояние RNNoise")

    # --------------------------------------------------------------- #
    #  Special methods                                                #
    # --------------------------------------------------------------- #

    def __del__(self):
        if getattr(self, "_state", None):
            try:
                self._lib.rnnoise_destroy(self._state)
            finally:
                self._state = None            # type: ignore[attr-defined]

    # --------------------------------------------------------------- #
    #  Public API                                                     #
    # --------------------------------------------------------------- #

    def denoise_chunk(self, audio_data: np.ndarray
                      ) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Генератор денойз-фреймов.

        Parameters
        ----------
        audio_data : np.ndarray
            Массив формы (N,) или (1, N) dtype float32/64 в диапазоне −1…1.
        Yields
        ------
        (speech_prob, denoised_frame)
        """
        if audio_data.ndim == 1:
            samples = audio_data.astype(np.float32, copy=False)
        elif audio_data.ndim == 2 and audio_data.shape[0] == 1:
            samples = audio_data.flatten().astype(np.float32, copy=False)
        else:
            raise ValueError("audio_data должен быть (N,) или (1, N)")

        total = samples.shape[0]
        for pos in range(0, total, FRAME_SIZE):
            frame = samples[pos:pos + FRAME_SIZE]
            tail = len(frame) < FRAME_SIZE
            if tail:
                pad = np.zeros(FRAME_SIZE, dtype=np.float32)
                pad[:len(frame)] = frame
                frame = pad

            in_buf = (ctypes.c_float * FRAME_SIZE)(*frame)
            out_buf = (ctypes.c_float * FRAME_SIZE)()
            prob = self._lib.rnnoise_process_frame(self._state, out_buf, in_buf).value
            den = np.ctypeslib.as_array(out_buf).astype(np.float32)
            if tail:
                den = den[:total - pos]
            yield prob, den.copy()

    # ----------------------- Convenience --------------------------- #

    def filter(self, segment: "AudioSegment"):  # type: ignore[valid-type]
        """
        Деноис аудио-сегмента pydub.  При несоответствии частоты
        или отсутствии pydub возвращает исходный сегмент.
        """
        if AudioSegment is None or segment.frame_rate != self.sample_rate:
            return segment

        raw = np.array(segment.get_array_of_samples(), dtype=np.float32)
        max_int = float(2 ** (8 * segment.sample_width - 1))
        if segment.channels > 1:
            raw = raw.reshape(-1, segment.channels).mean(axis=1)
        raw /= max_int

        out_frames = (f for _, f in self.denoise_chunk(raw[None, :]))
        den = np.concatenate(list(out_frames))
        den_int = np.clip(den * max_int, -max_int, max_int-1).astype(
            {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}[segment.sample_width])
        return segment._spawn(den_int.tobytes())


__all__ = ["RNNoise", "FRAME_SIZE"]
