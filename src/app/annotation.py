# src/app/annotation.py
# -*- coding: utf-8 -*-
"""
Сервис аннотации аудио для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import logging
import inspect
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Awaitable, Dict, Any, Callable, List, Optional

from .config import AppSettings
from .stages import PreprocessingStage, DiarizationStage, TranscriptionStage, RecognitionStage
from .stages.carddav_stage import CardDAVStage
from .schemas import (
    AnnotationResult, AudioMetadata, ProcessingInfo, FinalSpeaker,
    FinalSegment, FinalTranscription, Statistics, TranscriptionWord, ContactInfo
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
            PreprocessingStage(self.config, self.config.preprocess.dict(), models_registry),
            DiarizationStage(self.config, self.config.diarization.dict(), models_registry),
            TranscriptionStage(self.config, self.config.transcription.dict(), models_registry),
            RecognitionStage(self.config, self.config.recognition.dict(), models_registry),
            CardDAVStage(self.config, self.config.carddav.dict(), models_registry)
        ]
        self.logger.info("AnnotationService инициализирован с архитектурой этапов")

        # DEBUG: отключаем все этапы кроме диаризации
        # self.stages = [stage for stage in self.stages if isinstance(stage, DiarizationStage)
        #               #    or isinstance(stage, TranscriptionStage)
        #            ]

        self.logger.info(f"AnnotationService инициализирован len(stages)={len(self.stages)} ")

    async def process_audio(
        self,
        file_path: str,
        task_id: str,
        #progress_callback: Optional[Callable[[int, str], None]] = None
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

    def _build_final_annotation(
        self,
        task_id: str,
        audio_metadata: AudioMetadata,
        context: Dict[str, Any]
    ) -> AnnotationResult:
        """
        Собирает финальный результат аннотации, включая скорректированную confidence
        на уровне всей транскрипции и сегментов, а также confidence спикеров из RecognitionStage.
        """
        diar_result = context.get("diarization")
        trans_result = context.get("transcription")
        recog_result = context.get("recognition")
        carddav_result = context.get("carddav")

        # ProcessingInfo
        processing_info = ProcessingInfo(
            diarization_model=diar_result.model_info if diar_result else {},
            transcription_model=trans_result.model_info if trans_result else {},
            recognition_model=recog_result.model_info if recog_result else {},
            processing_time={
                "diarization": round(diar_result.processing_time, 3) if diar_result else 0.0,
                "transcription": round(trans_result.processing_time, 3) if trans_result else 0.0,
                "recognition": round(recog_result.processing_time, 3) if recog_result else 0.0,
                "carddav": round(carddav_result.processing_time, 3) if carddav_result else 0.0
            }
        )

        # Сырые данные
        raw_diar = diar_result.payload.get("segments", []) if diar_result else []
        raw_trans = trans_result.payload if trans_result else {}
        raw_recog = recog_result.payload.get("speakers", {}) if recog_result else {}

        # --- Построение карты спикеров ---
        speakers_map: Dict[str, FinalSpeaker] = {}
        idx_counter = 1
        # Инициализируем спикеров по меткам из диаризации
        for seg in raw_diar:
            label = seg["speaker"]
            if label not in speakers_map:
                spk_id = f"speaker_{idx_counter:02d}"
                idx_counter += 1
                # Начальное значение confidence: берём из recognition payload, если есть
                rec_info = raw_recog.get(label, {})
                init_conf = float(rec_info.get("confidence", 0.0))
                speakers_map[label] = FinalSpeaker(
                    id=spk_id,
                    label=label,
                    segments_count=0,
                    total_duration=0.0,
                    identified=bool(rec_info.get("identified", False)),
                    name=rec_info.get("name"),
                    contact_info=None,
                    voice_embedding=None,
                    confidence=round(init_conf, 3)
                )

        # --- Сборка сегментов и подсчёт статистики ---
        segments: List[FinalSegment] = []
        full_text_parts: List[str] = []
        total_words = 0
        total_speech = 0.0

        for i, d in enumerate(raw_diar, start=1):
            start, end = d["start"], d["end"]
            duration = round(end - start, 3)
            seg_speaker = speakers_map[d["speaker"]]
            seg_speaker.segments_count += 1
            seg_speaker.total_duration += duration
            total_speech += duration

            # Текст и слова из транскрипции
            words_in = [
                w for w in raw_trans.get("words", [])
                if w["start"] < end and w["end"] > start
            ]
            text = " ".join(w["word"] for w in words_in).strip()

            # Confidence сегмента
            seg_conf = 0.0
            for seg in raw_trans.get("segments", []):
                if abs(seg["start"] - start) < 1e-3:
                    seg_conf = seg.get("avg_logprob", 0.0)
                    break
            if seg_conf == 0.0 and words_in:
                seg_conf = sum(w["probability"] for w in words_in) / len(words_in)

            segments.append(FinalSegment(
                id=i,
                start=round(start, 3),
                end=round(end, 3),
                duration=duration,
                speaker=seg_speaker.id,
                speaker_label=d["speaker"],
                text=text,
                words=[TranscriptionWord(**w) for w in words_in],
                confidence=round(seg_conf, 3)
            ))
            full_text_parts.append(f"[{seg_speaker.id}]: {text}")
            total_words += len(words_in)

        # Итоговая транскрипция
        overall_conf = raw_trans.get("confidence", 0.0)
        if overall_conf == 0.0 and raw_trans.get("words"):
            overall_conf = sum(w["probability"] for w in raw_trans["words"]) / len(raw_trans["words"])

        transcription = FinalTranscription(
            full_text="\n".join(full_text_parts),
            confidence=round(overall_conf, 3),
            language=raw_trans.get("language", "unknown"),
            words=[TranscriptionWord(**w) for w in raw_trans.get("words", [])]
        )

        # Дополняем info по спикерам: имя и карточка из CardDAV
        for label, spk in speakers_map.items():
            # Recognition уже установил identified, name, confidence
            # CardDAV
            cd = carddav_result.payload.get("speakers", {}).get(label, {}) if carddav_result else {}
            contact = cd.get("contact")
            if contact:
                spk.contact_info = ContactInfo(**contact)

        final_speakers = list(speakers_map.values())

        stats = Statistics(
            total_speakers=len(final_speakers),
            identified_speakers=sum(1 for s in final_speakers if s.identified),
            unknown_speakers=sum(1 for s in final_speakers if not s.identified),
            total_segments=len(segments),
            total_words=total_words,
            speech_duration=round(total_speech, 3),
            silence_duration=round(max(0.0, audio_metadata.duration - total_speech), 3)
        )

        return AnnotationResult(
            task_id=task_id,
            version=str(self.config.server.version),
            created_at=datetime.now(),
            audio_metadata=audio_metadata,
            processing_info=processing_info,
            speakers=final_speakers,
            segments=segments,
            transcription=transcription,
            statistics=stats
        )
