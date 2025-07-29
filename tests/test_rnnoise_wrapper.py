# src/app/rnnoise_wrapper.py

# -*- coding: utf-8 -*-
"""
Простая обёртка над нативной библиотекой RNNoise через ctypes

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import ctypes
import ctypes.util
import numpy as np

# Константы RNNoise
FRAME_SIZE = 480  # 10ms at 48kHz

class RNNoise:
    """Обёртка над нативной библиотекой RNNoise"""

    def __init__(self, sample_rate: int = 48000):
        if sample_rate != 48000:
            raise ValueError("RNNoise работает только с частотой 48 кГц")
        self.sample_rate = sample_rate

        # ищем библиотеку
        lib_path = ctypes.util.find_library('rnnoise')
        if not lib_path:
            for path in ['/usr/local/lib/librnnoise.so', '/usr/lib/librnnoise.so']:
                try:
                    self._lib = ctypes.CDLL(path)
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("Библиотека RNNoise не найдена")
        else:
            self._lib = ctypes.CDLL(lib_path)

        # пытаемся установить restype/argtypes, но пропускаем, если это простой метод
        try:
            self._lib.rnnoise_create.restype = ctypes.c_void_p
        except (AttributeError, TypeError):
            pass
        try:
            self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        except (AttributeError, TypeError):
            pass
        try:
            self._lib.rnnoise_process_frame.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float)
            ]
            self._lib.rnnoise_process_frame.restype = ctypes.c_float
        except (AttributeError, TypeError):
            pass

        # создаём состояние
        self._state = self._lib.rnnoise_create()
        if not self._state:
            raise RuntimeError("Не удалось создать состояние RNNoise")

    def __del__(self):
        if hasattr(self, '_state') and self._state:
            try:
                self._lib.rnnoise_destroy(self._state)
            except Exception:
                pass

    def denoise_chunk(self, audio_data: np.ndarray):
        """
        Подавление шума в аудио чанке
        Args:
            audio_data: numpy массив формы (1, N) с аудиоданными
        Yields:
            Кортежи (speech_prob, denoised_frame)
        """
        if audio_data.ndim != 2 or audio_data.shape[0] != 1:
            raise ValueError("Ожидается массив формы (1, N)")

        samples = audio_data.flatten().astype(np.float32)
        for i in range(0, len(samples), FRAME_SIZE):
            frame = samples[i:i + FRAME_SIZE]
            if len(frame) < FRAME_SIZE:
                padded = np.zeros(FRAME_SIZE, dtype=np.float32)
                padded[:len(frame)] = frame
                frame = padded

            # создаём C-массивы
            c_input = (ctypes.c_float * FRAME_SIZE)(*frame)
            c_output = (ctypes.c_float * FRAME_SIZE)()

            # порядок аргументов: state, input, output
            self._lib.rnnoise_process_frame(self._state, c_input, c_output)

            denoised = np.array(c_output, dtype=np.float32)
            # обрезаем паддинг
            if i + FRAME_SIZE > len(samples):
                denoised = denoised[:len(samples) - i]
            # возвращаем speech_prob не используется, ставим 0.0
            yield 0.0, denoised
