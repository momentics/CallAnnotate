# src/app/stages/transcription.py

# -*- coding: utf-8 -*-

"""
Этап транскрипции аудио для CallAnnotate с правильным выравниванием слов и спикеров.
Рефакторинг: теперь используются все параметры конфигурации TranscriptionConfig:
- model, device, task, language, batch_size
- metrics.{confidence, avg_logprob, no_speech_prob, timing}
- min_segment_duration, max_silence_between, min_overlap
А также поддерживается адаптивная постобработка сегментов:
- слияние соседних сегментов одного спикера при паузе <= max_silence_between
- пересчет confidence по словам при metrics.confidence=True
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
from ..stages.base import BaseStage


class TranscriptionStage(BaseStage):

    @property
    def stage_name(self) -> str:
        return "transcription"

    async def _initialize(self) -> None:
        """
        Инициализация модели Whisper с учетом всех параметров конфигурации.
        """
        cfg = self.config

        # Параметры из конфигурации
        model_ref: str = cfg.get("model") # type: ignore
        # Допускаются значения вида "openai/whisper-small" и просто "small"
        # Для локального кеша и логов оставляем «размер» модели как суффикс
        model_size = model_ref.rsplit("-", 1)[-1] if "/" in model_ref else model_ref  # small/base/...

        self.device: str = str(cfg.get("device", "cpu"))
        self.task: str = str(cfg.get("task", "transcribe"))
        self.lang_cfg: Optional[str] = cfg.get("language", "auto")
        self.language: Optional[str] = None if self.lang_cfg == "auto" else str(self.lang_cfg)

        # Настройки метрик
        metrics: Dict[str, Any] = dict(cfg.get("metrics", {}))
        self.metrics_confidence: bool = bool(metrics.get("confidence", True))
        self.metrics_avg_logprob: bool = bool(metrics.get("avg_logprob", True))
        self.metrics_no_speech_prob: bool = bool(metrics.get("no_speech_prob", True))
        self.metrics_timing: bool = bool(metrics.get("timing", True))

        # Параметры привязки к диаризации
        self.min_segment_duration: float = float(cfg.get("min_segment_duration", 0.2))
        self.max_silence_between: float = float(cfg.get("max_silence_between", 0.3))
        self.min_overlap_ratio: float = float(cfg.get("min_overlap", 0.3))

        # Batch-параметр (используется в логике постобработки/частичной агрегации,
        # т.к. у openai-whisper отсутствует явный аргумент batch_size в transcribe())
        self.batch_size: int = int(cfg.get("batch_size", 4))

        # Whisper аргументы
        whisper_args = {
            "task": self.task,
            "language": self.language,
            "word_timestamps": True,
            # Рекомендованные параметры для устойчивости результата
            "temperature": [0.0, 0.2, 0.4],
            "best_of": 3,
            "condition_on_previous_text": False,
            "initial_prompt": "Это телефонный разговор на русском языке." if (self.language in (None, "ru")) else None,
        }
        # Удаляем None поля
        self.whisper_kwargs = {k: v for k, v in whisper_args.items() if v is not None}

        # Директория локального кеша модели
        self.cache_path = Path(self.volume_path) / "models" / "whisper"
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Загрузка модели через registry с кешем
        self.model = models_registry.get_model(
            self.logger,
            f"whisper_{model_size}_{self.device}",
            lambda: whisper.load_model(model_size, device=self.device, download_root=str(self.cache_path)),
            stage="transcription",
            framework="OpenAI Whisper",
        )

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Выполняет транскрипцию, выравнивание с диаризацией и постобработку сегментов
        с учетом параметров:
         - min_segment_duration
         - max_silence_between
         - min_overlap
         - metrics.*
        """
        diar_segments = previous_results.get("segments", [])

        async def update(pct: int, msg: str):
            if progress_callback:
                await progress_callback(pct, msg)

        # Точка отсчета времени (если timing=True, вернем processing_time)
        start_time = time.perf_counter()

        await update(0, "Начало транскрипции")
        # Запуск whisper
        raw = self.model.transcribe(file_path, **self.whisper_kwargs)
        raw_segments = raw.get("segments", []) or []
        raw_language = raw.get("language", (self.language or "unknown"))

        await update(40, "Выравнивание с диаризацией")
        aligned_segments, aligned_words = self._align_transcription_with_diarization(
            whisper_segments=raw_segments,
            diar_segments=diar_segments,
        )

        await update(60, "Постобработка сегментов")
        # Фильтрация слишком коротких сегментов и слияние соседних с учетом max_silence_between
        post_segments = self._postprocess_segments(
            aligned_segments,
            min_duration=self.min_segment_duration,
            max_pause=self.max_silence_between,
        )

        await update(80, "Пересчет метрик")
        # Итоговая «уверенность» в транскрипции
        overall_conf = 0.0
        if self.metrics_confidence:
            overall_conf = self._calculate_overall_confidence(
                aligned_words=aligned_words
            )

        # При включенных метриках сохраним confidence в сегментах (если не был установлен)
        if self.metrics_confidence:
            for seg in post_segments:
                if seg.get("confidence") is None:
                    ws = seg.get("words", [])
                    if ws:
                        seg["confidence"] = round(
                            float(sum(w.get("probability", 0.0) for w in ws)) / max(1, len(ws)),
                            3
                        )

        processing_time = round(time.perf_counter() - start_time, 3) if self.metrics_timing else None
        await update(100, "Транскрипция завершена")

        # Формируем итоговый payload, исключая ненужные метрики если выключены
        result: Dict[str, Any] = {
            "segments": self._strip_segment_metrics(post_segments),
            "words": self._strip_word_metrics(aligned_words),
            "confidence": round(overall_conf, 3) if self.metrics_confidence else 0.0,
            "language": raw_language,
        }
        if self.metrics_timing:
            result["processing_time"] = processing_time  # type: ignore[assignment]
        return result

    # ------------------------------- Выравнивание -------------------------------

    def _align_transcription_with_diarization(
        self,
        whisper_segments: List[Dict],
        diar_segments: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Присваивает каждому сегменту/слову доминирующего спикера с учетом перекрытия,
        фильтрует сегменты короче min_segment_duration.
        """
        aligned_segments: List[Dict] = []
        aligned_words: List[Dict] = []

        diar_index = self._create_diarization_index(diar_segments)

        for seg in whisper_segments:
            seg_start: float = float(seg.get("start", 0.0))
            seg_end: float = float(seg.get("end", seg_start))
            duration = seg_end - seg_start
            if duration < self.min_segment_duration:
                # Сразу отбрасываем «микро»-сегменты — параметр используется
                continue

            # Определение доминирующего спикера для сегмента
            dominant_speaker, speaker_conf = self._find_dominant_speaker(
                seg_start, seg_end, diar_index
            )

            segment_words: List[Dict] = []
            word_probs: List[float] = []

            for word_data in seg.get("words", []) or []:
                w_start = float(word_data.get("start", seg_start))
                w_end = float(word_data.get("end", w_start))
                w_prob = float(word_data.get("probability", 0.0))

                w_speaker, w_sconf = self._find_speaker_for_word(
                    w_start, w_end, diar_index, fallback=dominant_speaker
                )

                # Сохраняем метрики слов только если они включены
                word_entry = {
                    "start": round(w_start, 3),
                    "end": round(w_end, 3),
                    "word": word_data.get("word", ""),
                    "probability": round(w_prob, 3),
                    "speaker": w_speaker,
                    "speaker_confidence": round(w_sconf, 3)
                }
                segment_words.append(word_entry)
                aligned_words.append(word_entry)
                word_probs.append(w_prob)

            if not segment_words:
                # Если слов нет — такой сегмент бесполезен
                continue

            seg_payload: Dict[str, Any] = {
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": str(seg.get("text", "")).strip(),
                "speaker": dominant_speaker,
                "speaker_confidence": round(float(speaker_conf), 3),
                "words": segment_words,
            }

            # Метрики сегментов зависят от настроек metrics.*
            if self.metrics_no_speech_prob:
                seg_payload["no_speech_prob"] = round(float(seg.get("no_speech_prob", 0.0)), 3)
            if self.metrics_avg_logprob:
                seg_payload["avg_logprob"] = round(float(seg.get("avg_logprob", 0.0)), 3)
            if self.metrics_confidence:
                seg_payload["confidence"] = round(statistics.fmean(word_probs), 3) if word_probs else 0.0

            aligned_segments.append(seg_payload)

        return aligned_segments, aligned_words

    def _create_diarization_index(self, diar_segments: List[Dict]) -> List[Dict]:
        """
        Преобразует результат диаризации в удобный индекс.
        """
        return sorted(
            [
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "speaker": seg.get("speaker"),
                    "confidence": float(seg.get("confidence", 1.0))
                }
                for seg in diar_segments
            ],
            key=lambda x: x["start"]
        )

    def _find_dominant_speaker(
        self,
        start: float,
        end: float,
        diar_index: List[Dict]
    ) -> Tuple[str, float]:
        """
        Выбирает спикера, который покрывает максимальную часть интервала [start, end],
        учитывая confidence диаризации. Возвращает (speaker, pseudo_confidence in [0..1]).
        """
        if not diar_index or end <= start:
            return "unknown", 0.0

        speaker_stats: Dict[str, Dict[str, float]] = {}
        for seg in diar_index:
            ovl_start = max(start, seg["start"])
            ovl_end = min(end, seg["end"])
            if ovl_start < ovl_end:
                d = ovl_end - ovl_start
                sp = seg["speaker"] or "unknown"
                st = speaker_stats.setdefault(sp, {"duration": 0.0, "conf": 0.0, "count": 0})
                st["duration"] += d * seg.get("confidence", 1.0)
                st["conf"] += seg.get("confidence", 1.0)
                st["count"] += 1

        if not speaker_stats:
            return "unknown", 0.0

        best_sp, info = max(speaker_stats.items(), key=lambda kv: kv[1]["duration"])
        total = (end - start)
        coverage = (info["duration"] / total) if total > 0 else 0.0
        avg_conf = (info["conf"] / info["count"]) if info["count"] else 0.0
        # Ограничим до [0..1]
        return best_sp, float(min(1.0, max(0.0, coverage * avg_conf)))

    def _find_speaker_for_word(
        self,
        w_start: float,
        w_end: float,
        diar_index: List[Dict],
        fallback: str
    ) -> Tuple[str, float]:
        """
        Ищет спикера для слова по максимальному относительному перекрытию.
        Если перекрытие ниже min_overlap_ratio — возвращает fallback.
        """
        best_ratio = 0.0
        best_sp = fallback
        best_conf = 0.0
        w_dur = max(1e-9, w_end - w_start)

        for seg in diar_index:
            ovl_start = max(w_start, seg["start"])
            ovl_end = min(w_end, seg["end"])
            if ovl_start < ovl_end:
                ovl = ovl_end - ovl_start
                ratio = ovl / w_dur
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_sp = seg["speaker"] or fallback
                    best_conf = seg.get("confidence", 1.0) * ratio

        if best_ratio < self.min_overlap_ratio:
            return fallback, 0.5
        return best_sp, float(min(1.0, max(0.0, best_conf)))

    # ------------------------------ Постобработка ------------------------------

    def _postprocess_segments(
        self,
        segments: List[Dict],
        min_duration: float,
        max_pause: float
    ) -> List[Dict]:
        """
        - Удаляет сегменты короче min_duration (дополнительный «filter after align»).
        - Сливает соседние сегменты одного спикера, если тишина между ними <= max_pause.
          При слиянии:
            * объединяет тексты с пробелом;
            * конкатенирует списки слов, сохраняет сортировку по времени;
            * обновляет end, confidence (если включено), speaker_confidence как среднее.
        """
        if not segments:
            return []

        # Фильтрация по длительности
        kept = []
        for seg in segments:
            if (float(seg["end"]) - float(seg["start"])) >= float(min_duration):
                kept.append(seg)
        if not kept:
            return []

        # Сортировка по времени
        kept.sort(key=lambda s: (str(s.get("speaker", "")), float(s["start"]), float(s["end"])))

        merged: List[Dict] = []
        cur = kept[0]

        for s in kept[1:]:
            same_speaker = (s.get("speaker") == cur.get("speaker"))
            gap = float(s["start"]) - float(cur["end"])
            if same_speaker and gap >= -1e-9 and gap <= max_pause:
                # Слияние
                cur_end = max(float(cur["end"]), float(s["end"]))
                cur["end"] = round(cur_end, 3)

                # Текст: соединяем через пробел, убирая пустые
                t1 = str(cur.get("text", "")).strip()
                t2 = str(s.get("text", "")).strip()
                cur["text"] = " ".join([t for t in (t1, t2) if t]).strip()

                # Слова: конкатенация и сортировка
                words = list(cur.get("words", [])) + list(s.get("words", []))
                words.sort(key=lambda w: float(w.get("start", 0.0)))
                cur["words"] = words

                # speaker_confidence: среднее
                sc_a = float(cur.get("speaker_confidence", 0.0))
                sc_b = float(s.get("speaker_confidence", 0.0))
                cur["speaker_confidence"] = round((sc_a + sc_b) / 2.0, 3)

                # confidence: если включено — пересчитать по словам
                if self.metrics_confidence:
                    cur_words = cur.get("words", [])
                    if cur_words:
                        cur["confidence"] = round(
                            float(sum(w.get("probability", 0.0) for w in cur_words)) / max(1, len(cur_words)),
                            3
                        )
                # avg_logprob/no_speech_prob — оставляем от первого, чтобы не раздувать
            else:
                merged.append(cur)
                cur = s

        merged.append(cur)

        # Итоговая сортировка по времени
        merged.sort(key=lambda s: float(s["start"]))
        return merged

    def _strip_segment_metrics(self, segments: List[Dict]) -> List[Dict]:
        """
        Удаляет из сегментов метрики, если соответствующие флаги выключены.
        """
        out: List[Dict] = []
        for s in segments:
            seg = dict(s)  # копия
            if not self.metrics_no_speech_prob:
                seg.pop("no_speech_prob", None)
            if not self.metrics_avg_logprob:
                seg.pop("avg_logprob", None)
            if not self.metrics_confidence:
                seg.pop("confidence", None)
            out.append(seg)
        return out

    def _strip_word_metrics(self, words: List[Dict]) -> List[Dict]:
        """
        В текущей схеме слова всегда включают probability и speaker_confidence.
        Если в будущем потребуется — можно убрать также эти поля, если метрики выключены.
        """
        if self.metrics_confidence:
            return words
        # Если confidence выключен, probability и speaker_confidence не влияют на итоговый конфиданс,
        # но могут быть полезны. Оставим их для совместимости. Если нужно скрывать — раскомментировать ниже.
        # out = []
        # for w in words:
        #     ww = dict(w)
        #     ww.pop("probability", None)
        #     ww.pop("speaker_confidence", None)
        #     out.append(ww)
        # return out
        return words

    # ------------------------------ Метрики ------------------------------

    def _calculate_overall_confidence(self, aligned_words: List[Dict]) -> float:
        """
        Взвешенная по длительности слов метрика уверенности:
        combine(probability, speaker_confidence) усредняется по времени.
        """
        if not aligned_words:
            return 0.0
        total_weight = 0.0
        accum = 0.0
        for w in aligned_words:
            start = float(w.get("start", 0.0))
            end = float(w.get("end", start))
            dur = max(0.0, end - start)
            prob = float(w.get("probability", 0.0))
            sp_conf = float(w.get("speaker_confidence", 0.0))
            combined = (prob + sp_conf) / 2.0
            accum += combined * dur
            total_weight += dur
        return round(accum / total_weight, 3) if total_weight > 0 else 0.0
