# src/app/annotation.py
# -*- coding: utf-8 -*-
"""
Сервис аннотации аудио для CallAnnotate

Добавлено:
- Дополнительная финальная нормализация сегментов после сборки (после маппинга слов и подсчёта текста):
  удаление сверхкоротких артефактов и слияние соседних сегментов одного спикера
  (порог min_segment_duration берём из конфига, merge_gap=0.05s).
- Это гарантирует отсутствие микро-сегментов даже если они каким-то образом пройдут через предыдущие этапы.

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import logging
import inspect
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Awaitable, Dict, Any, Callable, List, Optional, Tuple

from .config import AppSettings
from .stages import PreprocessingStage, DiarizationStage, TranscriptionStage, RecognitionStage
from .stages.carddav_stage import CardDAVStage
from .schemas import (
    AnnotationResult, AudioMetadata, ProcessingInfo, FinalSpeaker,
    FinalSegment, FinalTranscription, Statistics, TranscriptionSegment, TranscriptionWord, ContactInfo
)
from .models_registry import models_registry
from .utils import extract_audio_metadata, ensure_directory


class AnnotationService:
    """Основной сервис аннотации аудиофайлов с использованием этапной архитектуры"""

    def __init__(self, config: AppSettings):
        self.config = config
        from .utils import setup_logging
        setup_logging(config)

        self.logger = logging.getLogger(__name__)
        self.stages = [
            # PreprocessingStage(self.config, self.config.preprocess.dict(), models_registry),
            DiarizationStage(self.config, self.config.diarization.dict(), models_registry),
            TranscriptionStage(self.config, self.config.transcription.dict(), models_registry),
            RecognitionStage(self.config, self.config.recognition.dict(), models_registry),
            CardDAVStage(self.config, self.config.carddav.dict(), models_registry)
        ]
        self.logger.info("AnnotationService инициализирован с архитектурой этапов")
        self.logger.info(f"AnnotationService инициализирован len(stages)={len(self.stages)} ")

    async def process_audio(
        self,
        file_path: str,
        task_id: str,
        progress_callback: Optional[Callable[[int, str], Awaitable[Any]]] = None
    ) -> Dict[str, Any]:
        try:
            src = Path(file_path)
            vol_path = Path(self.config.queue.volume_path).expanduser().resolve()
            incoming = vol_path / "incoming"
            ensure_directory(str(incoming))
            if src.resolve().parent != incoming:
                incoming_file = incoming / src.name
                shutil.copy(src, incoming_file)
                file_path = str(incoming_file)
            else:
                file_path = str(src)

            await self._update_progress(progress_callback, 0, "Начало обработки аудио")
            audio_metadata = extract_audio_metadata(file_path)
            await self._update_progress(progress_callback, 5, "Метаданные аудио извлечены")

            context: Dict[str, Any] = {}

            num_stages = len(self.stages)
            boundaries = [int(10 + i * (80 / num_stages)) for i in range(num_stages + 1)]

            proc = Path(self.config.queue.volume_path) / "processing"
            proc.mkdir(parents=True, exist_ok=True)
            processing_path = proc / Path(file_path).name
            shutil.move(file_path, processing_path)
            file_path = str(processing_path)

            for i, stage in enumerate(self.stages):
                start = boundaries[i]
                end = boundaries[i + 1]
                await self._update_progress(progress_callback, start, f"Начало этапа {stage.stage_name}")

                async def stage_cb(pct: int, msg: str):
                    overall = start + int((end - start) * pct / 100)
                    await self._update_progress(progress_callback, overall, msg)

                result = await stage.process(
                    file_path, task_id, context, stage_cb
                )

                if stage.stage_name == "preprocess":
                    processed = result.payload.get("processed_path")
                    if processed:
                        file_path = processed

                context[stage.stage_name] = result
                await self._update_progress(progress_callback, end, f"Этап {stage.stage_name} завершен")

            await self._update_progress(progress_callback, 95, "Сборка финального результата")
            final = self._build_final_annotation(task_id, audio_metadata, context)
            await self._update_progress(progress_callback, 100, "Аннотация завершена")
            return final.dict()

        except Exception as e:
            self.logger.error(f"Ошибка при обработке аудио {file_path}: {e}")
            raise

    async def _update_progress(
        self,
        callback: Optional[Callable[[int, str], Any]],
        progress: int,
        message: str
    ):
        if callback:
            result = callback(progress, message)
            if inspect.isawaitable(result):
                await result
        self.logger.info(f"Прогресс: {progress:3d}% – {message}")

    # ----------------------------- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ -----------------------------
    def _overlap_ratio(self, a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        """
        Возвращает долю перекрытия от длины меньшего интервала.
        Используется для устойчивой оценки соответствия.
        """
        inter_start = max(a_start, b_start)
        inter_end = min(a_end, b_end)
        inter = max(0.0, inter_end - inter_start)
        denom = max(1e-9, min(a_end - a_start, b_end - b_start))
        return inter / denom

    def _assign_speakers_to_transcription(
        self,
        diar_segments: List[Dict[str, Any]],
        diar_label_to_id: Dict[str, str],
        trans_words: List[Dict[str, Any]],
        trans_segments: List[Dict[str, Any]],
        min_overlap: float
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Присваивает спикеров словам и сегментам транскрипции на основании диаризации.
        """
        for w in trans_words:
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            best_ratio = 0.0
            best_label = None
            for d in diar_segments:
                ratio = self._overlap_ratio(ws, we, d["start"], d["end"])
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_label = d["speaker_label"] if "speaker_label" in d else d["speaker"]
            if best_label and best_ratio >= min_overlap:
                w["speaker"] = diar_label_to_id.get(best_label, diar_label_to_id.get(str(best_label), "unknown"))

        for s in trans_segments:
            ss = float(s.get("start", 0.0))
            se = float(s.get("end", ss))
            best_ratio = 0.0
            best_label = None
            for d in diar_segments:
                ratio = self._overlap_ratio(ss, se, d["start"], d["end"])
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_label = d["speaker_label"] if "speaker_label" in d else d["speaker"]
            if best_label and best_ratio >= min_overlap:
                s["speaker"] = diar_label_to_id.get(best_label, diar_label_to_id.get(str(best_label), "unknown"))
                s["speaker_confidence"] = round(best_ratio, 3)

        return trans_words, trans_segments

    def _merge_and_filter_final_segments(
        self,
        segments: List[FinalSegment],
        speakers_map: Dict[str, FinalSpeaker],
        min_duration: float,
        merge_gap: float = 0.05
    ) -> List[FinalSegment]:
        """
        Финальная нормализация итоговых сегментов:
        - Отбрасываем сверхкороткие (duration < min_duration).
        - Сливаем соседние сегменты одного и того же speaker, если они стыкуются внутри merge_gap.
        Обновляем текст и слова при слиянии.
        """
        # Фильтр по длительности
        seq = [s for s in segments if float(s.duration) >= float(min_duration)]
        if not seq:
            return []

        # Сортировка по времени
        seq.sort(key=lambda s: (s.speaker, s.start))
        merged: List[FinalSegment] = []
        cur = seq[0]

        for s in seq[1:]:
            same = (s.speaker == cur.speaker)
            gap = float(s.start) - float(cur.end)
            if same and gap >= -1e-6 and gap <= merge_gap:
                # Слияние: конкатенируем текст и слова, обновляем границы
                cur.end = max(cur.end, s.end)
                cur.duration = round(float(cur.end) - float(cur.start), 3)
                # Конкатенируем слова по времени
                words = list(cur.words) + list(s.words)
                words.sort(key=lambda w: w.start)
                cur.words = words
                # Текст пересобираем из слов (надёжнее)
                cur.text = " ".join((w.word or "").strip() for w in cur.words).strip()
                # Уверенность как среднее по словам (если нужны уточнения, можно изменить)
                if cur.words:
                    cur.confidence = round(sum(w.probability for w in cur.words) / max(1, len(cur.words)), 3)
            else:
                merged.append(cur)
                cur = s
        merged.append(cur)

        # Возвращаем к порядку по времени (независимо от спикера)
        merged.sort(key=lambda s: s.start)
        return merged

    def _build_final_annotation(
        self,
        task_id: str,
        audio_metadata: AudioMetadata,
        context: Dict[str, Any]
    ) -> AnnotationResult:
        diar_result = context.get("diarization")
        trans_result = context.get("transcription")
        recog_result = context.get("recognition")
        carddav_result = context.get("carddav")

        diar_cfg = self.config.diarization
        trans_cfg = self.config.transcription
        recog_cfg = self.config.recognition

        diar_model_name = None
        diar_framework = None
        if diar_result and hasattr(diar_result, "model_info"):
            diar_model_name = diar_result.model_info.get("model_name")
            diar_framework = diar_result.model_info.get("framework")
        if not diar_model_name:
            diar_model_name = diar_cfg.model
        if not diar_framework:
            diar_framework = "pyannote.audio"

        diar_model_info = {
            "stage": "diarization",
            "model_name": diar_model_name,
            "device": diar_cfg.device,
            "framework": diar_framework
        }

        trans_model_info = {
            "stage": "transcription",
            "model_size": trans_cfg.model.split("/")[-1],
            "device": trans_cfg.device,
            "framework": "OpenAI Whisper"
        }

        recog_model_info = {
            "stage": "recognition",
            "model_name": recog_cfg.model,
            "device": recog_cfg.device,
            "threshold": recog_cfg.threshold,
            "database_size": len(context.get("recognition").payload.get("speakers", {})) if recog_result else 0,  # type: ignore
            "framework": "SpeechBrain + FAISS"
        }

        processing_info = ProcessingInfo(
            diarization_model=diar_model_info,
            transcription_model=trans_model_info,
            recognition_model=recog_model_info,
            processing_time={
                "diarization": round(diar_result.processing_time, 3) if diar_result else 0.0,
                "transcription": round(trans_result.processing_time, 3) if trans_result else 0.0,
                "recognition": round(recog_result.processing_time, 3) if recog_result else 0.0,
                "carddav": round(carddav_result.processing_time, 3) if carddav_result else 0.0
            }
        )

        raw_diar = diar_result.payload.get("segments", []) if diar_result else []
        raw_trans = trans_result.payload if trans_result else {}
        raw_recog = recog_result.payload.get("speakers", {}) if recog_result else {}

        speakers_map: Dict[str, FinalSpeaker] = {}
        idx_counter = 1
        for seg in raw_diar:
            label = seg.get("speaker_label") or seg["speaker"]
            if label not in speakers_map:
                spk_id = f"speaker_{idx_counter:02d}"
                idx_counter += 1
                rec_info = raw_recog.get(label, {})
                init_conf = float(rec_info.get("confidence", 0.0))
                speakers_map[label] = FinalSpeaker(
                    id=spk_id,
                    label=str(label),
                    segments_count=0,
                    total_duration=0.0,
                    identified=bool(rec_info.get("identified", False)),
                    name=rec_info.get("name"),
                    contact_info=None,
                    voice_embedding=None,
                    confidence=round(init_conf, 3)
                )

        diar_label_to_id: Dict[str, str] = {lbl: fs.id for lbl, fs in speakers_map.items()}

        segments: List[FinalSegment] = []
        full_text_parts: List[str] = []
        total_words = 0
        total_speech = 0.0

        words_all: List[Dict[str, Any]] = list(raw_trans.get("words", []))
        trans_segments_raw: List[Dict[str, Any]] = list(raw_trans.get("segments", []))

        diar_for_map: List[Dict[str, Any]] = []
        for d in raw_diar:
            diar_for_map.append({
                "start": float(d["start"]),
                "end": float(d["end"]),
                "speaker_label": d.get("speaker_label") or d["speaker"]
            })

        min_overlap = max(0.0, min(1.0, float(self.config.transcription.min_overlap)))

        words_all, trans_segments_raw = self._assign_speakers_to_transcription(
            diar_segments=diar_for_map,
            diar_label_to_id=diar_label_to_id,
            trans_words=words_all,
            trans_segments=trans_segments_raw,
            min_overlap=min_overlap
        )

        # Формируем финальные сегменты по диаризации
        for idx, d in enumerate(raw_diar, start=1):
            start, end = float(d["start"]), float(d["end"])
            duration = round(end - start, 3)
            label = d.get("speaker_label") or d["speaker"]
            spk = speakers_map[label]
            spk.segments_count += 1
            spk.total_duration += duration
            total_speech += duration

            words_in = [
                w for w in words_all
                if float(w.get("start", 0.0)) < end and float(w.get("end", 0.0)) > start
            ]
            words_in.sort(key=lambda w: float(w.get("start", 0.0)))

            text = " ".join((w.get("word") or "").strip() for w in words_in).strip()

            seg_conf = float(d.get("confidence", 0.0))
            if seg_conf == 0.0 and words_in:
                seg_conf = sum(float(w.get("probability", 0.0)) for w in words_in) / max(1, len(words_in))

            segments.append(FinalSegment(
                id=idx,
                start=round(start, 3),
                end=round(end, 3),
                duration=duration,
                speaker=spk.id,
                speaker_label=str(label),
                text=text,
                words=[TranscriptionWord(**{
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "word": str(w["word"]),
                    "probability": float(w.get("probability", 0.0)),
                    "speaker": str(w.get("speaker", spk.id))
                }) for w in words_in],
                confidence=round(seg_conf, 3)
            ))
            if text:
                full_text_parts.append(f"[{spk.id}]: {text}")

            total_words += len(words_in)

        # Финальная нормализация итоговых сегментов: фильтр + слияние.
        min_seg = float(self.config.transcription.min_segment_duration)
        segments = self._merge_and_filter_final_segments(
            segments=segments,
            speakers_map=speakers_map,
            min_duration=min_seg,
            merge_gap=0.05
        )

        # Пересобираем full_text после нормализации
        full_text_parts = []
        for seg in segments:
            if seg.text:
                full_text_parts.append(f"[{seg.speaker}]: {seg.text}")

        overall_conf = float(raw_trans.get("confidence", 0.0))
        all_words_flat: List[TranscriptionWord] = []
        for seg in segments:
            all_words_flat.extend(seg.words)
        if overall_conf == 0.0 and all_words_flat:
            overall_conf = sum(float(w.probability) for w in all_words_flat) / max(1, len(all_words_flat))

        fixed_trans_segments: List[TranscriptionSegment] = []
        for s in trans_segments_raw:
            s_start = float(s.get("start", 0.0))
            s_end = float(s.get("end", s_start))
            s_text = s.get("text")
            if not s_text:
                ws = [w for w in all_words_flat if float(w.start) < s_end and float(w.end) > s_start]
                ws.sort(key=lambda w: float(w.start))
                s_text = " ".join((w.word or "").strip() for w in ws).strip()

            fixed_trans_segments.append(TranscriptionSegment(
                start=round(s_start, 3),
                end=round(s_end, 3),
                text=s_text or "",
                speaker=s.get("speaker"),
                speaker_confidence=s.get("speaker_confidence"),
                no_speech_prob=s.get("no_speech_prob"),
                avg_logprob=s.get("avg_logprob"),
                confidence=s.get("confidence")
            ))

        transcription = FinalTranscription(
            full_text="\n".join(full_text_parts),
            confidence=round(overall_conf, 3),
            language=raw_trans.get("language", "unknown"),
            segments=fixed_trans_segments,
            # Список слов возвращаем в составе fin-сегментов; при необходимости можно добавить общим списком
        )

        # CardDAV сопоставление, если доступно
        if carddav_result:
            cd_speakers = carddav_result.payload.get("speakers", {})
            for label, spk in speakers_map.items():
                cd = cd_speakers.get(label, {}).get("contact")
                if cd:
                    spk.contact_info = ContactInfo(**cd)

        final_speakers = list(speakers_map.values())
        stats = Statistics(
            total_speakers=len(final_speakers),
            identified_speakers=sum(1 for s in final_speakers if s.identified),
            unknown_speakers=sum(1 for s in final_speakers if not s.identified),
            total_segments=len(segments),
            total_words=sum(len(s.words) for s in segments),
            speech_duration=round(sum(s.duration for s in segments), 3),
            silence_duration=round(max(0.0, audio_metadata.duration - sum(s.duration for s in segments)), 3)
        )

        return AnnotationResult(
            task_id=task_id,
            version=str(self.config.server.version),
            created_at=datetime.now(timezone.utc),
            audio_metadata=audio_metadata,
            processing_info=processing_info,
            speakers=final_speakers,
            segments=segments,
            transcription=transcription,
            statistics=stats
        )
