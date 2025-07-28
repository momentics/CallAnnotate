# src/app/stages/preprocessing.py

# -*- coding: utf-8 -*-
"""
Этап предобработки аудио: чанковая нормализация, подавление шума RNNoise_Wrapper и DeepFilterNet2
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import tempfile
from typing import Dict, Any, Optional, Callable

import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects

from rnnoise_wrapper import RNNoise  # обёртка RNNoise_Wrapper
from .base import BaseStage

try:
    from df.enhance import init_df, enhance
except ImportError:
    init_df = None
    enhance = None


class PreprocessingStage(BaseStage):
    def __init__(self, config: Dict[str, Any], models_registry=None):
        super().__init__(config, models_registry)
        self.model = None
        self.df_state = None

    @property
    def stage_name(self) -> str:
        return "preprocess"

    async def _initialize(self):
        # Проверяем DeepFilterNet2
        if init_df is None or enhance is None:
            raise RuntimeError("Не установлена зависимость deepfilternet")
        # Инициализация DeepFilterNet2
        self.model, self.df_state, _ = init_df(
            model=self.config.get("model", "DeepFilterNet2"),
            device=self.config.get("device", "cpu")
        )

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Разбивает WAV на чанки, нормализует громкость,
        применяет RNNoise_Wrapper (если доступно) и DeepFilterNet2,
        и объединяет результат в один WAV-файл.
        """
        dur_sec = float(self.config.get("chunk_duration", 2.0))
        overlap_sec = float(self.config.get("overlap", 0.5))
        target_rms = float(self.config.get("target_rms", -20.0))

        audio = AudioSegment.from_file(file_path)
        sr = audio.frame_rate
        chunk_ms = int(dur_sec * 1000)
        overlap_ms = int(overlap_sec * 1000)
        total_ms = len(audio)

        processed_chunks = []
        position = 0
        idx = 0

        while position < total_ms:
            end = min(position + chunk_ms, total_ms)
            chunk = audio[position:end]

            # Нормализация RMS
            normed = effects.normalize(chunk, headroom=abs(target_rms))

            # Попытка лёгкого подавления через RNNoise_Wrapper
            try:
                denoiser = RNNoise()
                denoised_seg = denoiser.filter(normed)
            except Exception:
                denoised_seg = normed  # если не удалось — используем исходный

            # Конвертация в numpy (моно)
            samples = np.array(denoised_seg.get_array_of_samples(), dtype=np.float32)
            samples = samples.reshape(-1, denoised_seg.channels).mean(axis=1)

            # Глубокое шумоподавление
            samples_df = enhance(self.model, self.df_state, samples)

            # Сохраняем чанки
            tmp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_chunk{idx}.wav")
            sf.write(tmp_path, samples_df, sr)
            processed_chunks.append(tmp_path)

            idx += 1
            position += chunk_ms - overlap_ms

            if progress_callback:
                percent = min(100, int(100 * position / total_ms))
                await progress_callback(percent, f"preprocess chunk {idx}")

        # Объединяем и экспортируем
        combined = AudioSegment.from_file(processed_chunks[0])
        for p in processed_chunks[1:]:
            combined += AudioSegment.from_file(p)

        out_path = file_path.replace(".wav", "_processed.wav")
        combined.export(out_path, format="wav")

        # Чистим временные файлы
        for p in processed_chunks:
            try:
                os.remove(p)
            except OSError:
                pass

        return {"processed_path": out_path}
