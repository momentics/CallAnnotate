# src/app/stages/transcription.py

# -*- coding: utf-8 -*-

"""
Этап транскрипции аудио для CallAnnotate с правильным выравниванием слов и спикеров.
Исправлена обработка confidence, прогресс-колбека и точное сопоставление временных меток.
"""

from __future__ import annotations
import os
import time
import statistics
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

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
        model_ref = cfg.get("model")
        model_size = model_ref.rsplit("-", 1)[-1] # type: ignore
        device = cfg.get("device", "cpu")

        whisper_args = {
            "task": cfg.get("task", "transcribe"),
            "language": None if cfg.get("language") == "auto" else cfg.get("language"),
            "word_timestamps": True,
        }
        self.whisper_kwargs = {k: v for k, v in whisper_args.items() if v is not None}

        self.min_segment_duration = float(cfg.get("min_segment_duration", 0.2))
        self.max_silence_between = float(cfg.get("max_silence_between", 0.5))
        self.min_overlap_ratio = float(cfg.get("min_overlap", 0.3))

        self.cache_path = Path(self.volume_path) / "models" / "whisper"
        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.model = models_registry.get_model(
            self.logger,
            f"whisper_{model_size}_{device}",
            lambda: whisper.load_model(model_size, device=device, download_root=str(self.cache_path)),
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

        async def update(pct: int, msg: str):
            if progress_callback:
                await progress_callback(pct, msg)

        await update(0, "Начало транскрипции")
        result = self.model.transcribe(file_path, **self.whisper_kwargs)
        raw_segments = result.get("segments", [])
        raw_lang = result.get("language", self.config.get("language", "unknown"))
        await update(50, "Выравнивание с диаризацией")

        aligned_segments, aligned_words = self._align_transcription_with_diarization(
            raw_segments, diar_segments
        )

        await update(90, "Вычисление метрик")
        overall_conf = self._calculate_overall_confidence(aligned_words)
        processing_time = round(time.perf_counter() - start_time, 3)
        await update(100, "Транскрипция завершена")

        return {
            "segments": aligned_segments,
            "words": aligned_words,
            "confidence": overall_conf,
            "language": raw_lang,
            "processing_time": processing_time
        }

    def _align_transcription_with_diarization(
        self,
        whisper_segments: List[Dict],
        diar_segments: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        aligned_segments = []
        aligned_words = []
        diar_index = self._create_diarization_index(diar_segments)

        for whisper_seg in whisper_segments:
            seg_start, seg_end = whisper_seg["start"], whisper_seg["end"]
            duration = seg_end - seg_start
            if duration < self.min_segment_duration:
                continue

            dominant_speaker, speaker_conf = self._find_dominant_speaker(
                seg_start, seg_end, diar_index
            )

            segment_words = []
            word_probs = []
            for word_data in whisper_seg.get("words", []):
                w_start, w_end = word_data["start"], word_data["end"]
                w_prob = float(word_data.get("probability", 0.0))
                w_speaker, w_sconf = self._find_speaker_for_word(
                    w_start, w_end, diar_index, dominant_speaker
                )
                word_entry = {
                    "start": round(w_start, 3),
                    "end": round(w_end, 3),
                    "word": word_data["word"],
                    "probability": round(w_prob, 3),
                    "speaker": w_speaker,
                    "speaker_confidence": round(w_sconf, 3)
                }
                segment_words.append(word_entry)
                aligned_words.append(word_entry)
                word_probs.append(w_prob)

            if segment_words:
                seg_conf = statistics.fmean(word_probs) if word_probs else 0.0
                aligned_segments.append({
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "text": whisper_seg["text"].strip(),
                    "speaker": dominant_speaker,
                    "speaker_confidence": round(speaker_conf, 3),
                    "no_speech_prob": round(float(whisper_seg.get("no_speech_prob", 0.0)), 3),
                    "avg_logprob": round(float(whisper_seg.get("avg_logprob", 0.0)), 3),
                    "confidence": round(seg_conf, 3),
                    "words": segment_words
                })

        return aligned_segments, aligned_words

    def _create_diarization_index(self, diar_segments: List[Dict]) -> List[Dict]:
        return sorted(
            [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": seg["speaker"],
                    "confidence": seg.get("confidence", 1.0)
                }
                for seg in diar_segments
            ],
            key=lambda x: x["start"]
        )

    def _find_dominant_speaker(
        self,
        start: float,
        end: float,
        diar_index: List[Dict],
        overlap_threshold: float = 0.1
    ) -> Tuple[str, float]:
        speaker_stats: Dict[str, Dict[str, float]] = {}
        for seg in diar_index:
            overlap_start = max(start, seg["start"])
            overlap_end = min(end, seg["end"])
            if overlap_start < overlap_end:
                d = overlap_end - overlap_start
                sp = seg["speaker"]
                stats = speaker_stats.setdefault(sp, {"duration": 0.0, "confidence": 0.0, "count": 0})
                stats["duration"] += d * seg["confidence"]
                stats["confidence"] += seg["confidence"]
                stats["count"] += 1

        if not speaker_stats:
            return "unknown", 0.0

        best, info = max(speaker_stats.items(), key=lambda x: x[1]["duration"])
        coverage = info["duration"] / (end - start) if (end - start) > 0 else 0.0
        avg_conf = info["confidence"] / info["count"] if info["count"] else 0.0
        return best, min(1.0, coverage * avg_conf)

    def _find_speaker_for_word(
        self,
        w_start: float,
        w_end: float,
        diar_index: List[Dict],
        fallback: str
    ) -> Tuple[str, float]:
        best_overlap = 0.0
        best_sp = fallback
        best_conf = 0.0
        w_dur = w_end - w_start

        for seg in diar_index:
            o_start = max(w_start, seg["start"])
            o_end = min(w_end, seg["end"])
            if o_start < o_end:
                o_d = o_end - o_start
                ratio = o_d / w_dur
                if ratio > best_overlap:
                    best_overlap = ratio
                    best_sp = seg["speaker"]
                    best_conf = seg["confidence"] * ratio

        if best_overlap < 0.3:
            return fallback, 0.5
        return best_sp, min(1.0, best_conf)

    def _calculate_overall_confidence(self, words: List[Dict]) -> float:
        if not words:
            return 0.0
        total_w = 0.0
        sum_conf = 0.0
        for w in words:
            dur = w["end"] - w["start"]
            combined = (w.get("probability", 0.0) + w.get("speaker_confidence", 0.0)) / 2.0
            sum_conf += combined * dur
            total_w += dur
        return round(sum_conf / total_w if total_w else 0.0, 3)
