# src/app/stages/diarization.py
# -*- coding: utf-8 -*-
"""
Этап диаризации для CallAnnotate с исправленным определением уникальных спикеров.

Проблема:
Ранее количество спикеров определялось некорректно из сегментов, из-за чего в результат попадал только один.

Решение:
- Собираем множество меток спикеров напрямую из объекта Annotation:
  annotation.labels()
- Если меньше двух уникальных меток, добавляем 'unknown'
- Если больше двух, выбираем два меток с наибольшей общей длительностью

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment, SlidingWindow

from ..config import AppSettings
from .base import BaseStage


class DiarizationStage(BaseStage):
    CONFIG_FILE = "pyannote_diarization_config.yaml"
    #pipeline = load_pipeline_from_pretrained(PATH_TO_CONFIG)
    @property
    def stage_name(self) -> str:
        return "diarization"

    async def _initialize(self) -> None:
        model_id = str(self.config.get("model"))
        use_auth = self.config.get("use_auth_token")
        device = self.config.get("device", "cpu")
        
        self.cache_path = Path(self.volume_path).expanduser().resolve() / "models" / "pyannote"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_path / self.CONFIG_FILE
        
        def loader():
            cwd = Path.cwd().resolve()  # store current working directory
            
            cd_to = self.cache_path.parent.resolve()
            os.chdir(cd_to)

            #pipe = Pipeline.from_pretrained(model_id, use_auth_token=use_auth)
            pipe = Pipeline.from_pretrained(self.cache_path)

            try:
                pipe.to(device)
            except Exception:
                pass

            os.chdir(cwd)
            return pipe

        if self.models_registry:
            self.pipeline = self.models_registry.get_model(
                self.logger,
                f"diarization:{model_id}",
                loader,
                model_name=model_id,
                framework="pyannote.audio"
            )
        else:
            self.pipeline = loader()

        self._initialized = True
        self.logger.info("DiarizationStage инициаллизирована")

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        audio_path = Path(file_path)
        with sf.SoundFile(str(audio_path)) as f:
            sr = f.samplerate
            duration = len(f) / sr

        if progress_callback:
            await progress_callback(10, "Начало диаризации")

        window_enabled = bool(self.config.get("window_enabled", False))
        window_enabled = False
        window_size = float(self.config.get("window_size", duration))
        hop_size = float(self.config.get("hop_size", window_size))

        if window_enabled and window_size < duration:
            annotation = await self._process_with_sliding_window(
                audio_path, duration, window_size, hop_size, progress_callback
            )
        else:
            annotation = self.pipeline(str(audio_path))
            if progress_callback:
                await progress_callback(70, "Диаризация завершена")

        if progress_callback:
            await progress_callback(80, "Извлечение сегментов")

        segments = self._extract_segments(annotation, duration)

        # Собираем уникальные метки спикеров из annotation.labels()
        labels = list(annotation.labels())

        # Считаем суммарную длительность по каждой метке
        durations: Dict[str, float] = {lbl: 0.0 for lbl in labels} # type: ignore
        for seg in segments:
            lbl = seg["speaker"]
            durations[lbl] = durations.get(lbl, 0.0) + seg["duration"]

        # Если меньше двух, добавляем 'unknown'
        if len(labels) == 0:
            labels = ["speaker_1", "unknown"]
        elif len(labels) == 1:
            labels.append("unknown")
        # Если больше двух, выбираем две метки с максимальной длительностью
        elif len(labels) > 2:
            sorted_labels = sorted(durations.items(), key=lambda x: x[1], reverse=True)
            labels = [lbl for lbl, _ in sorted_labels[:2]]

        processing_time = round(time.perf_counter() - start_time, 3)
        self.logger.info(
            f"Диаризация завершена за {processing_time} сек: speakers={labels}, segments={len(segments)}"
        )

        return {
            "segments": segments,
            "speakers": labels,
            "total_segments": len(segments),
            "total_speakers": len(labels),
            "processing_time": processing_time
        }

    async def _process_with_sliding_window(
        self,
        audio_path: Path,
        duration: float,
        window_size: float,
        hop_size: float,
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> Annotation:
        combined = Annotation(uri=audio_path.stem)
        sw = SlidingWindow(duration=window_size, step=hop_size)
        windows = list(sw(Segment(0.0, duration), align_last=False))
        total = len(windows)
        for idx, window in enumerate(windows, start=1):
            ann = self.pipeline({
                "uri": audio_path.stem,
                "audio": str(audio_path),
                "segment": window
            })
            for segment, track, label in ann.itertracks(yield_label=True):
                start = segment.start + window.start
                end = min(segment.end + window.start, duration)
                if end > start:
                    combined[Segment(start, end), track] = label
            if progress_callback:
                pct = int(10 + 60 * idx / total)
                await progress_callback(pct, f"Обработка окна {idx}/{total}")
        return combined

    def _extract_segments(self, annotation: Annotation, duration: float) -> List[Dict[str, Any]]:
        raw: List[Dict[str, Any]] = []
        for segment, _, label in annotation.itertracks(yield_label=True): # type: ignore
            start = round(segment.start, 3)
            end = round(min(segment.end, duration), 3)
            if end <= start:
                continue
            raw.append({
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "speaker": str(label),
                "confidence": 0.0
            })
        raw.sort(key=lambda x: x["start"])
        return raw
