# src/app/stages/transcription.py
# -*- coding: utf-8 -*-
"""
Этап транскрипции аудио для CallAnnotate.
Исправлена обработка confidence, прогресс-колбека и округление probability.
"""
from __future__ import annotations
import os
import time
import statistics
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import whisper
import numpy as np

from ..models_registry import models_registry

from ..config import AppSettings
from ..stages.base import BaseStage, StageResult



class TranscriptionStage(BaseStage):
    @property
    def stage_name(self) -> str:
        return "transcription"

    async def _initialize(self) -> None:
        cfg = self.config
        model_ref = cfg["model"]
        # определяем размер модели
        # Для любых вариантов ID вида “openai/whisper-small” или “whisper-small”
        model_size = model_ref.rsplit("-", 1)[-1]  # всегда “small”, “base” и т.д.

        device = cfg.get("device", "cpu")

        # параметры Whisper
        whisper_args = {
            "task": cfg.get("task", "transcribe"),
            "language": None if cfg.get("language") == "auto" else cfg.get("language"),
            "word_timestamps": True,
        }

        # убираем None
        self.whisper_kwargs = {k: v for k, v in whisper_args.items() if v is not None}

        # фильтры сегментов
        self.min_segment_duration = float(cfg.get("min_segment_duration", 0.2))
        self.max_silence_between = float(cfg.get("max_silence_between", 0.0))
        self.min_overlap_ratio = float(cfg.get("min_overlap", 0.3))

        # настройка кэша
        cache = Path(self.volume_path) / "models" / "whisper"
        os.environ["HF_HOME"] = str(cache)
        os.environ["TRANSFORMERS_CACHE"] = str(cache)
        os.environ["TORCH_HOME"] = str(cache)

        # загрузка модели
        self.model = models_registry.get_model(
            self.logger,
            f"whisper_{model_size}_{device}",
            lambda: whisper.load_model(model_size, device=device),
            stage="transcription",
            framework="OpenAI Whisper"
        )

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        diar_segments = previous_results.get("segments", [])

        # полный прогресс: 0→100
        def update(percent: int, msg: str):
            if progress_callback:
                progress_callback(percent, msg)

        update(0, "Начало транскрипции")

        # разбиваем на сегменты по диаризации
        result = self.model.transcribe(file_path, **self.whisper_kwargs)
        raw_segments = result.get("segments", [])
        raw_lang = result.get("language", self.config.get("language", "unknown"))

        segments_out = []
        words_out = []
        total_weight = 0.0
        weighted_conf = 0.0

        for idx, seg in enumerate(raw_segments, 1):
            start, end = seg["start"], seg["end"]
            duration = end - start
            if duration < self.min_segment_duration:
                continue

            # находим спикера
            speaker = None
            for d in diar_segments:
                if d["start"] <= start < d["end"]:
                    speaker = d["speaker"]
                    break

            # собираем слова
            word_probs = []
            for w in seg.get("words", []):
                prob = float(w.get("probability", 0.0))
                word_probs.append(prob)
                words_out.append({
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "word": w["word"],
                    "probability": round(prob, 3),
                    "speaker": speaker,
                })

            # confidence сегмента как среднее probability слов
            seg_conf = statistics.fmean(word_probs) if word_probs else 0.0
            weighted_conf += seg_conf * duration
            total_weight += duration

            segments_out.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": seg["text"].strip(),
                "speaker": speaker,
                "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 3),
                "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 3),
                "speaker_confidence": round(seg_conf, 3),
            })

            # обновляем прогресс
            update(10 + int(80 * idx / len(raw_segments)), f"Сегмент {idx}/{len(raw_segments)}")

        overall_conf = round((weighted_conf / total_weight) if total_weight > 0 else 0.0, 3)
        update(95, "Завершение транскрипции")

        processing_time = round(time.perf_counter() - start_time, 3)
        update(100, "Транскрипция завершена")

        return {
            "segments": segments_out,
            "words": words_out,
            "confidence": overall_conf,
            "language": raw_lang,
            "processing_time": processing_time,
        }

