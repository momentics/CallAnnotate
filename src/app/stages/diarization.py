# src/app/stages/diarization.py
# -*- coding: utf-8 -*-
"""
Этап диаризации для CallAnnotate с поддержкой пакетной обработки очень длинных аудиофайлов
с использованием pyannote.audio ≥3.1 и SlidingWindow из pyannote.core.

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from ctypes import Union
import time
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Callable

import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Segment, SlidingWindow, Annotation

from ..config import AppSettings
from .base import BaseStage


class DiarizationStage(BaseStage):
    @property
    def stage_name(self) -> str:
        return "diarization"

    async def _initialize(self) -> None:
        """
        Инициализация pyannote.audio Pipeline без использования отсутствующих методов.
        """
        model_id = str(self.config.get("model"))
        use_auth = self.config.get("use_auth_token")
        device = self.config.get("device", "cpu")
        logger = self.logger

        def loader():
            logger.info(f"Загрузка пайплайна диаризации '{model_id}'")
            pipe = Pipeline.from_pretrained(model_id, use_auth_token=use_auth)
            try:
                pipe.to(device)
                logger.info(f"Пайплайн перенесён на устройство: {device}")
            except Exception:
                logger.warning("Не удалось перенести пайплайн на устройство, используется по умолчанию")
            return pipe

        if self.models_registry is not None:
            self.pipeline = self.models_registry.get_model(
                self.logger,
                f"diarization:{model_id}",
                loader,
                model_name=model_id,
                framework="pyannote.audio"
            )
        self._initialized = True
        logger.info("DiarizationStage инициализирован")

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Выполняет диаризацию аудио со скользящим окном,
        обрезая сегменты по границе длительности.
        Возвращает словарь с сегментами, списком спикеров и статистикой.
        """
        start_time = time.perf_counter()
        audio = Path(file_path)

        # узнаём длительность через soundfile
        with sf.SoundFile(str(audio)) as sf_desc:
            samplerate = sf_desc.samplerate
            frames = len(sf_desc)
            duration = frames / samplerate

        window_enabled = bool(self.config.get("window_enabled", False))
        window_size = float(self.config.get("window_size", duration))
        hop_size = float(self.config.get("hop_size", window_size))

        combined = Annotation(uri=audio.stem)

        if window_enabled and window_size < duration:
            sw = SlidingWindow(duration=window_size, step=hop_size)
            # align_last=False предотвращает расширение последнего окна
            windows = list(sw(Segment(0.0, duration), align_last=False))
            total = len(windows)
            for idx, window in enumerate(windows, start=1):
                ann = self.pipeline({
                    "uri": audio.stem,
                    "audio": str(audio),
                    "segment": window
                })
                for segment, _, label in ann.itertracks(yield_label=True):
                    abs_start = segment.start + window.start
                    abs_end = segment.end + window.start
                    # обрезаем до duration
                    abs_end = min(abs_end, duration)
                    combined[Segment(abs_start, abs_end)] = label
                if progress_callback is not None:
                    pct = int(90 * idx / total)
                    await progress_callback(pct, f"диаризация окно {idx}/{total}")
        else:
            ann = self.pipeline(str(audio))
            for segment, _, label in ann.itertracks(yield_label=True):
                # обрезаем до duration
                end = min(segment.end, duration)
                combined[Segment(segment.start, end)] = label
            if progress_callback:
                await progress_callback(90, "диаризация полного файла")

        support = combined.get_timeline().support_iter(collar=0.0)

        merged_segments: List[Dict[str, Any]] = []
        for seg in support:
            counts: Dict[str, float] = {}
                
            for item in combined.crop(seg).itertracks(yield_label=True):
                # Unpack safely
                if len(item) == 3:
                    subseg, track, label = item
                else:
                    subseg, track = item
                    label = track  # or assign a default label as needed

                dur = subseg.duration
                counts[str(label)] = counts.get(str(label), 0.0) + dur            
                
            speaker = max(counts, key=lambda k: counts[k], default="unknown") # if counts else "unknown"

            merged_segments.append({
                "start": round(seg.start, 3),
                "end": round(min(seg.end, duration), 3),
                "duration": round(min(seg.end, duration) - seg.start, 3),
                "speaker": speaker,
                "confidence": 0.0
            })
        if progress_callback:
            await progress_callback(100, "слияние сегментов завершено")

        speakers = sorted({s["speaker"] for s in merged_segments})
        processing_time = round(time.perf_counter() - start_time, 3)

        return {
            "segments": merged_segments,
            "speakers": speakers,
            "total_segments": len(merged_segments),
            "total_speakers": len(speakers),
            "processing_time": processing_time
        }


