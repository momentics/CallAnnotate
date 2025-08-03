# src/app/stages/transcription.py

# -*- coding: utf-8 -*-
"""
Этап транскрипции аудио для CallAnnotate.
Поправлены дефекты фильтрации по минимальной длительности
и учтён параметр max_silence_between для привязки сегментов к спикерам.
"""
from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional

import whisper
from app.models_registry import ModelRegistry  # подключаем для проверки

from ..stages.base import BaseStage


class TranscriptionStage(BaseStage):
    """Этап транскрипции на базе OpenAI Whisper."""

    # Допуск при сравнении длительностей (сек.).
    _EPS: float = 1e-6

    @property
    def stage_name(self) -> str:
        return "transcription"

    async def _initialize(self) -> None:
        # Основные параметры Whisper
        model_name: str = self.config["model"]
        if "whisper-" in model_name:
            model_size = model_name.split("whisper-")[-1]
        else:
            model_size = model_name.split("/")[-1]
        device: str = self.config["device"]

        # Параметры фильтрации сегментов
        self.min_segment_duration: float = float(
            self.config.get("min_segment_duration", 0.2)
        )
        self.max_silence_between: float = float(
            self.config.get("max_silence_between", 0.0)
        )
        self.min_overlap_ratio: float = float(
            self.config.get("min_overlap", 0.3)
        )

        self.logger.info("Loading Whisper model '%s' on %s", model_size, device)
        # Используем models_registry только если это наш ModelRegistry
        if isinstance(self.models_registry, ModelRegistry):
            cache_key = f"whisper_{model_size}_{device}"
            self.model = self.models_registry.get_model(
                cache_key,
                lambda: whisper.load_model(model_size, device=device),
                stage=self.stage_name,
                framework="OpenAI Whisper",
            )
        else:
            self.model = whisper.load_model(model_size, device=device)

        self.logger.info("TranscriptionStage initialised successfully")

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет транскрипцию аудио и собирает сегменты и слова с метриками.
        Если есть результаты диаризации, привязывает к ним спикеров;
        иначе возвращает все транскрибированные сегменты.
        """
        t0 = time.perf_counter()

        whisper_opts = {
            "language": None if self.config["language"] == "auto" else self.config["language"],
            "task": self.config["task"],
            "temperature": 0.0,
            "word_timestamps": True,
            "verbose": False,
            "fp16": False,
        }
        if progress_callback:
            await progress_callback(10, "Transcription started")

        # Транскрипция
        try:
            infer = self.model.transcribe(file_path, **whisper_opts)
        except RuntimeError:
            infer = {"segments": [], "confidence": 0.0}

        # Извлечение сырых сегментов и уверенности
        raw_segments: List[Dict[str, Any]] = infer.get("segments", []) or []
        confidence = float(infer.get("confidence", 0.0) or 0.0)

        # Формирование списка слов по всем сегментам
        words_all: List[Dict[str, Any]] = []
        for seg in raw_segments:
            for w in seg.get("words", []):
                words_all.append({
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                    "word": w.get("word", ""),
                    "probability": round(w.get("probability", 0.0), 3),
                    "speaker": None,
                })

        # Получаем результаты диаризации (если переданы)
        diar_segments: List[Dict[str, Any]] = []
        prev = previous_results.get("segments")
        if isinstance(prev, list):
            diar_segments = prev

        if progress_callback:
            await progress_callback(50, "Packaging segments")

        # Выравнивание сегментов: фильтрация и поиск спикера
        aligned_segments: List[Dict[str, Any]] = []
        for trans in raw_segments:
            t_start = float(trans.get("start", 0.0))
            t_end = float(trans.get("end", 0.0))
            seg_dur = t_end - t_start

            # Фильтр по минимальной длительности
            if seg_dur < self.min_segment_duration:
                continue

            speaker_id = None
            speaker_conf = 0.0

            # Если есть диаризация — ищем лучший спикер
            if diar_segments:
                best_overlap = 0.0
                best_sp = None
                for diar in diar_segments:
                    d_start = diar.get("start", 0.0)
                    d_end = diar.get("end", 0.0)
                    overlap = max(0.0, min(t_end, d_end) - max(t_start, d_start))
                    silence = max(0.0, max(d_start, t_start) - min(d_end, t_end))
                    if silence <= self.max_silence_between and seg_dur > 0:
                        ratio = overlap / seg_dur
                        if ratio > best_overlap:
                            best_overlap = ratio
                            best_sp = diar.get("speaker")
                if best_overlap + self._EPS >= self.min_overlap_ratio:
                    speaker_id = best_sp
                    speaker_conf = round(best_overlap, 3)

            aligned_segments.append({
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "text": trans.get("text", "").strip(),
                "speaker": speaker_id,
                "speaker_confidence": speaker_conf,
                "no_speech_prob": round(trans.get("no_speech_prob", 0.0), 3),
                "avg_logprob": round(trans.get("avg_logprob", 0.0), 3),
            })

        # Если нет сегментов — устанавливаем уверенность в 0
        if not aligned_segments:
            confidence = 0.0

        # Метрики avg_logprob и no_speech_prob на уровне всех сегментов
        try:
            avg_logprob = statistics.fmean(
                s.get("avg_logprob", 0.0) for s in raw_segments
            )
        except statistics.StatisticsError:
            avg_logprob = 0.0
        try:
            avg_no_speech = statistics.fmean(
                s.get("no_speech_prob", 0.0) for s in raw_segments
            )
        except statistics.StatisticsError:
            avg_no_speech = 0.0

        # Привязка слов к сегментам
        for w in words_all:
            for seg in aligned_segments:
                if seg["start"] <= w["start"] < seg["end"]:
                    w["speaker"] = seg["speaker"]
                    break

        if progress_callback:
            await progress_callback(95, "Finalizing transcription")

        processing_time = round(time.perf_counter() - t0, 3)
        if progress_callback:
            await progress_callback(100, "Transcription completed")

        self.logger.info(
            "segments=%d words=%d confidence=%.3f avg_logprob=%.3f no_speech_prob=%.3f processing_time=%.3f",
            len(aligned_segments),
            len(words_all),
            round(confidence, 3),
            round(avg_logprob, 3),
            round(avg_no_speech, 3),
            processing_time,
        )

        return {
            "segments": aligned_segments,
            "words": words_all,
            "confidence": round(confidence, 3),
            "avg_logprob": round(avg_logprob, 3),
            "no_speech_prob": round(avg_no_speech, 3),
            "processing_time": processing_time,
        }
