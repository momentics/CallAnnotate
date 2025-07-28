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
        
        # Поиск библиотеки
        lib_path = ctypes.util.find_library('rnnoise')
        if not lib_path:
            # Попробовать прямые пути
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
        
        # Определение функций
        self._lib.rnnoise_create.restype = ctypes.c_void_p
        self._lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        self._lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_float), 
            ctypes.POINTER(ctypes.c_float)
        ]
        self._lib.rnnoise_process_frame.restype = ctypes.c_float
        
        # Создание состояния
        self._state = self._lib.rnnoise_create()
        if not self._state:
            raise RuntimeError("Не удалось создать состояние RNNoise")
    
    def __del__(self):
        if hasattr(self, '_state') and self._state:
            self._lib.rnnoise_destroy(self._state)
    
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
        
        # Обработка по фреймам
        for i in range(0, len(samples), FRAME_SIZE):
            frame = samples[i:i + FRAME_SIZE]
            
            # Дополнение нулями если фрейм неполный
            if len(frame) < FRAME_SIZE:
                padded_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
                padded_frame[:len(frame)] = frame
                frame = padded_frame
            
            # Подготовка буферов
            input_frame = (ctypes.c_float * FRAME_SIZE)(*frame)
            output_frame = (ctypes.c_float * FRAME_SIZE)()
            
            # Обработка фрейма
            speech_prob = self._lib.rnnoise_process_frame(
                self._state, output_frame, input_frame
            )
            
            # Конвертация результата
            denoised = np.array(output_frame, dtype=np.float32)
            
            # Обрезка если исходный фрейм был короче
            if i + FRAME_SIZE > len(samples):
                original_len = len(samples) - i
                denoised = denoised[:original_len]
            
            yield speech_prob, denoised
