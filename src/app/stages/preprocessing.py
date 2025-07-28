# -*- coding: utf-8 -*-
"""
Этап предобработки аудио: чанковая нормализация и подавление шума
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import tempfile
import builtins
from typing import Dict, Any, Optional, Callable
from pydub import AudioSegment, effects
import numpy as np
import soundfile as sf

from .base import BaseStage

# ИМПОРТИРУЕМ на уровень модуля!
try:
    from df.enhance import init_df, enhance
except ImportError:
    init_df = None
    enhance = None


class PreprocessingStage(BaseStage):
    def __init__(self, config: Dict[str, Any], models_registry=None):
        super().__init__(config, models_registry)
        self.init_df = None
        self.enhance = None
        self.model = None
        self.df_state = None

    @property
    def stage_name(self) -> str:
        return "preprocess"

    async def _initialize(self):
        # Используем глобальные init_df/enhance
        if init_df is None or enhance is None:
            raise RuntimeError("Не установлена зависимость deepfilternet")
        self.init_df = init_df
        self.enhance = enhance
        self.config = self.config or {}
        self.model, self.df_state, _ = self.init_df(
            model=self.config.get("model"),
            device=self.config.get("device")
        )

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Разбивает файл на чанки, нормализует и очищает шум."""

        # параметры из конфига
        dur = self.config.get("chunk_duration", 2.0)
        ovl = self.config.get("overlap", 0.5)
        target_rms = self.config.get("target_rms", -20.0)

        # загрузка аудио
        audio = AudioSegment.from_file(file_path)
        sr = audio.frame_rate
        chunk_ms = int(dur * 1000)
        overlap_ms = int(ovl * 1000)

        processed_chunks = []
        position = 0
        total_ms = len(audio)
        idx = 0

        while position < total_ms:
            end = min(position + chunk_ms, total_ms)
            chunk = audio[position:end]

            # нормализация громкости
            normed = effects.normalize(chunk, headroom=abs(target_rms))

            # преобразование в numpy
            samples = np.array(normed.get_array_of_samples(), dtype=np.float32)
            samples = samples.reshape(-1, normed.channels).mean(axis=1)

            # подавление шума DeepFilterNet2
            enhanced = self.enhance(self.model, self.df_state, samples)
            tmp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_chunk{idx}.wav")
            sf.write(tmp_path, enhanced, sr)
            processed_chunks.append(tmp_path)

            idx += 1
            position += chunk_ms - overlap_ms
            if progress_callback:
                await progress_callback(int(100 * position / total_ms), f"Preprocess chunk {idx}")

        # записываем объединённый файл
        combined = sum(
            (AudioSegment.from_file(p) for p in processed_chunks[1:]),
            AudioSegment.from_file(processed_chunks[0])
        )
        out_path = file_path.replace(".wav", "_preprocessed.wav")
        combined.export(out_path, format="wav")

        # очистка temp
        for p in processed_chunks:
            try:
                os.remove(p)
            except:
                pass

        return {"processed_path": out_path}
