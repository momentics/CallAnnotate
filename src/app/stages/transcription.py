# src/app/stages/transcription.py

# -*- coding: utf-8 -*-
"""
Этап транскрипции аудио для CallAnnotate.
Исправлена привязка транскрипции к диаризации:
- вместо транскрипции всего файла — транскрибируем каждый сегмент из диаризации
- слова и сегменты гарантированно попадают в границы сегмента
- собираем итоговый список сегментов и слов со спикерами

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""
from __future__ import annotations

import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

import whisper
import soundfile as sf

from ..stages.base import BaseStage
from ..models_registry import models_registry

class TranscriptionStage(BaseStage):
    """Этап транскрипции на базе OpenAI Whisper с по-сегментной обработкой"""

    # Допуск при сравнении длительностей (сек)
    _EPS: float = 1e-6

    @property
    def stage_name(self) -> str:
        return "transcription"

    async def _initialize(self) -> None:
        model_name: str = self.config["model"]
        if "whisper-" in model_name:
            model_size = model_name.split("whisper-")[-1]
        else:
            model_size = model_name.split("/")[-1]
        device: str = self.config["device"]

        self.min_segment_duration: float = float(self.config.get("min_segment_duration", 0.2))
        self.max_silence_between: float = float(self.config.get("max_silence_between", 0.0))
        self.min_overlap_ratio: float = float(self.config.get("min_overlap", 0.3))

        cache_dir = Path(self.volume_path) / "models"
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        os.environ["TORCH_HOME"] = str(cache_dir)

        self.logger.info(f"Loading Whisper model '{model_size}' on device {device}...")
        self.model = models_registry.get_model(
            self.logger,
            f"whisper_{model_size}_{device}",
            lambda: whisper.load_model(model_size, device=device),
            stage="transcription",
            framework="OpenAI Whisper"
        )
        self.logger.info("TranscriptionStage initialised successfully")

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Транскрибируем аудио по сегментам из диаризации:
        - для каждого сегмента диаризации вырезаем аудио и транскрибируем отдельно
        - собираем итоговые сегменты, слова и confidence
        """
        t0 = time.perf_counter()
        if progress_callback:
            await progress_callback(10, "Transcription started")

        # Получаем диаризованные сегменты из previous_results
        diar_segments = previous_results.get("segments", [])
        if not isinstance(diar_segments, list) or len(diar_segments) == 0:
            # Если нет диаризации, транскрибируем весь файл как один сегмент
            self.logger.warning("No diarization segments provided — transcribing whole file")
            return await self._transcribe_whole(file_path, progress_callback)

        segments: List[Dict[str, Any]] = []
        words_all: List[Dict[str, Any]] = []
        total_confidences: List[float] = []

        # Загружаем аудио целиком один раз для быстрой операции
        try:
            audio, sr = sf.read(file_path)
            audio_duration = len(audio) / sr
        except Exception as e:
            self.logger.warning(f"Failed to load audio with soundfile: {e}, using fallback...")
            audio_duration = None

        # Обработка каждого сегмента из диаризации
        for i, seg in enumerate(diar_segments):
            seg_start = max(float(seg.get("start", 0.0)), 0.0)
            seg_end = float(seg.get("end", audio_duration or 0.0))
            if seg_end <= seg_start:
                self.logger.debug(f"Skipping invalid diarization segment {i}: start >= end")
                continue
            seg_dur = seg_end - seg_start
            if seg_dur < self.min_segment_duration:
                self.logger.debug(f"Skipping too short diarization segment {i}, duration {seg_dur:.3f}s")
                continue

            if progress_callback:
                pct = 10 + int(70 * i / max(len(diar_segments), 1))
                await progress_callback(pct, f"Transcribing segment {i+1}/{len(diar_segments)}")

            # Вырезаем аудио с помощью soundfile или либо передаем сегмент для транскрипции
            # (whisper поддерживает передавать numpy массив с sample_rate)
            audio_chunk = None
            if audio_duration and audio is not None:
                # Отрезаем нужный фрагмент
                start_frame = int(seg_start * sr)
                end_frame = int(seg_end * sr)
                audio_chunk = audio[start_frame:end_frame]

            try:
                if audio_chunk is not None:
                    infer = self.model.transcribe(
                        audio_chunk,
                        language=None if self.config["language"] == "auto" else self.config["language"],
                        task=self.config["task"],
                        temperature=0.0,
                        word_timestamps=True,
                        verbose=False,
                        fp16=False,
                        initial_prompt=None,
                    )
                else:
                    # fallback: транскрибируем с указанием timestamps для файла, используя crop
                    infer = self.model.transcribe(
                        file_path,
                        language=None if self.config["language"] == "auto" else self.config["language"],
                        task=self.config["task"],
                        temperature=0.0,
                        word_timestamps=True,
                        verbose=False,
                        fp16=False,
                        initial_prompt=None,
                        # Для локального участка можно использовать параметр offset, duration, но Whisper не поддерживает явно
                        # Поэтому как fallback оставляем полный
                    )
            except Exception as e:
                self.logger.error(f"Error during transcription of segment {i}: {e}")
                continue

            seg_confidence = float(infer.get("confidence", 0.0) or 0.0)
            total_confidences.append(seg_confidence)

            raw_segments: List[Dict[str, Any]] = infer.get("segments", []) or []

            # Смещаем времена сегментов и слов на начало диаризации сегмента
            for rseg in raw_segments:
                r_start = rseg.get("start", 0.0)
                r_end = rseg.get("end", 0.0)
                r_text = rseg.get("text", "").strip()
                if r_end < r_start:
                    continue
                # Учитываем границы сегмента из диаризации, обрезаем при необходимости
                abs_start = seg_start + r_start
                abs_end = seg_start + r_end
                if abs_start < seg_start:
                    abs_start = seg_start
                if abs_end > seg_end:
                    abs_end = seg_end
                abs_dur = abs_end - abs_start
                if abs_dur < self.min_segment_duration:
                    continue

                # Привязываем к спикеру из диаризации
                speaker = seg.get("speaker")

                segments.append({
                    "start": round(abs_start, 3),
                    "end": round(abs_end, 3),
                    "text": r_text,
                    "speaker": speaker,
                    "speaker_confidence": seg_confidence,
                    "no_speech_prob": round(rseg.get("no_speech_prob", 0.0), 3),
                    "avg_logprob": round(rseg.get("avg_logprob", 0.0), 3),
                })

                # Добавляем слова с корректировкой времен
                for w in rseg.get("words", []):
                    w_start = w.get("start", 0.0)
                    w_end = w.get("end", 0.0)
                    abs_w_start = seg_start + w_start
                    abs_w_end = seg_start + w_end
                    # Корректируем, если слово выходит за границы сегмента
                    if abs_w_start < seg_start:
                        abs_w_start = seg_start
                    if abs_w_end > seg_end:
                        abs_w_end = seg_end
                    if abs_w_end <= abs_w_start:
                        continue
                    words_all.append({
                        "start": round(abs_w_start, 3),
                        "end": round(abs_w_end, 3),
                        "word": w.get("word", "").strip(),
                        "probability": round(w.get("probability", 0.0), 3),
                        "speaker": speaker,
                    })

        # Итоговый confidence — среднее по сегментам
        if total_confidences:
            confidence = float(statistics.fmean(total_confidences))
        else:
            confidence = 0.0

        # Метрики avg_logprob и no_speech_prob на уровне всех сегментов
        try:
            avg_logprob = statistics.fmean(
                s.get("avg_logprob", 0.0) for s in segments
            )
        except statistics.StatisticsError:
            avg_logprob = 0.0
        try:
            avg_no_speech = statistics.fmean(
                s.get("no_speech_prob", 0.0) for s in segments
            )
        except statistics.StatisticsError:
            avg_no_speech = 0.0

        if progress_callback:
            await progress_callback(95, "Finalizing transcription")

        processing_time = round(time.perf_counter() - t0, 3)
        if progress_callback:
            await progress_callback(100, "Transcription completed")

        self.logger.info(
            "segments=%d words=%d confidence=%.3f avg_logprob=%.3f no_speech_prob=%.3f processing_time=%.3f",
            len(segments), len(words_all), confidence, avg_logprob, avg_no_speech, processing_time,
        )

        return {
            "segments": segments,
            "words": words_all,
            "confidence": round(confidence, 3),
            "avg_logprob": round(avg_logprob, 3),
            "no_speech_prob": round(avg_no_speech, 3),
            "processing_time": round(processing_time, 3),
            "language": infer.get("language", "unknown") if infer else "unknown",
        }


    async def _transcribe_whole(
        self, file_path: str, progress_callback: Optional[Callable]
    ) -> Dict[str, Any]:
        """
        Вспомогательная транскрипция всего файла одним вызовом.
        Используется при отсутствии диаризации.
        """
        try:
            infer = self.model.transcribe(
                file_path,
                language=None if self.config["language"] == "auto" else self.config["language"],
                task=self.config["task"],
                temperature=0.0,
                word_timestamps=True,
                verbose=False,
                fp16=False,
            )
        except Exception as e:
            self.logger.error(f"Error during transcription of whole file: {e}")
            infer = {"segments": [], "confidence": 0.0}

        raw_segments = infer.get("segments", []) or []
        confidence = float(infer.get("confidence", 0.0) or 0.0)
        words_all = []
        for seg in raw_segments:
            for w in seg.get("words", []):
                words_all.append({
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                    "word": w.get("word", "").strip(),
                    "probability": round(w.get("probability", 0.0), 3),
                    "speaker": None,
                })

        avg_logprob = 0.0
        avg_no_speech = 0.0
        try:
            avg_logprob = statistics.fmean(s.get("avg_logprob", 0.0) for s in raw_segments)
        except statistics.StatisticsError:
            pass
        try:
            avg_no_speech = statistics.fmean(s.get("no_speech_prob", 0.0) for s in raw_segments)
        except statistics.StatisticsError:
            pass

        if progress_callback:
            await progress_callback(95, "Finalizing transcription")

        processing_time = 0.0

        return {
            "segments": raw_segments,
            "words": words_all,
            "confidence": round(confidence, 3),
            "avg_logprob": round(avg_logprob, 3),
            "no_speech_prob": round(avg_no_speech, 3),
            "processing_time": round(processing_time, 3),
            "language": infer.get("language", "unknown") if infer else "unknown",
        }
