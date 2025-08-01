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
from datetime import datetime
from typing import Dict, Any, Callable, Optional

from .stages import PreprocessingStage, DiarizationStage, TranscriptionStage, RecognitionStage
from .stages.carddav_stage import CardDAVStage

from .schemas import (
    AnnotationResult, AudioMetadata, ProcessingInfo, FinalSpeaker,
    FinalSegment, FinalTranscription, Statistics, TranscriptionWord, ContactInfo
)
from .models_registry import models_registry
from .utils import extract_audio_metadata


class AnnotationService:
    """Основной сервис аннотации аудиофайлов с использованием этапной архитектуры"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)

        if isinstance(config, dict):
            from .config import AppSettings
            self.config = AppSettings(**config)
        else:
            self.config = config

        self.stages = [
            PreprocessingStage(self.config.preprocess.dict(), models_registry),
            DiarizationStage(self.config.diarization.dict(), models_registry),
            TranscriptionStage(self.config.transcription.dict(), models_registry),
            RecognitionStage(self.config.recognition.dict(), models_registry),
            CardDAVStage(self.config.carddav.dict(), models_registry)
        ]
        self.logger.info("AnnotationService инициализирован с архитектурой этапов")

    async def process_audio(
        self,
        file_path: str,
        task_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        try:
            await self._update_progress(progress_callback, 0, "Начало обработки аудио")
            audio_metadata = extract_audio_metadata(file_path)
            await self._update_progress(progress_callback, 5, "Метаданные аудио извлечены")

            context: Dict[str, Any] = {}
            stage_progress = [10, 35, 65, 85]

            for i, stage in enumerate(self.stages):
                start = stage_progress[i]
                end = stage_progress[i+1] if i+1 < len(stage_progress) else 90

                await self._update_progress(progress_callback, start, f"Начало этапа {stage.stage_name}")

                async def stage_cb(pct: int, msg: str):
                    overall = start + int((end - start) * pct / 100)
                    await self._update_progress(progress_callback, overall, msg)

                result = await stage.process(
                    file_path, task_id, context, stage_cb
                )
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
            res = callback(progress, message)
            if inspect.isawaitable(res):
                await res
        self.logger.info(f"Прогресс {progress}%: {message}")

    def _build_final_annotation(
        self,
        task_id: str,
        audio_metadata: AudioMetadata,
        context: Dict[str, Any]
    ) -> AnnotationResult:
        diar = context.get("diarization")
        trans = context.get("transcription")
        recog = context.get("recognition")
        card = context.get("carddav")

        processing_info = ProcessingInfo(
            diarization_model=diar.model_info if diar else {},
            transcription_model=trans.model_info if trans else {},
            recognition_model=recog.model_info if recog else {},
            processing_time={
                "diarization": diar.processing_time if diar else 0.0,
                "transcription": trans.processing_time if trans else 0.0,
                "recognition": recog.processing_time if recog else 0.0,
                "carddav": card.processing_time if card else 0.0
            }
        )

        speakers_map: Dict[str, FinalSpeaker] = {}
        counter = 0

        diar_segments = diar.payload.get("segments", []) if diar else []
        recog_speakers = recog.payload.get("speakers", {}) if recog else {}
        card_speakers = card.payload.get("speakers", {}) if card else {}

        known = {v.name: v for v in getattr(self.config, "voices", [])}

        for seg in diar_segments:
            label = seg.get("speaker", "unknown")
            if label not in speakers_map:
                counter += 1
                spid = f"speaker_{counter:02d}"
                fs = FinalSpeaker(
                    id=spid,
                    label=label,
                    segments_count=0,
                    total_duration=0.0,
                    identified=False,
                    confidence=0.0,
                    name=None,
                    contact_info=None,
                    voice_embedding=None
                )
                rec = recog_speakers.get(label) or {}
                if rec.get("identified"):
                    fs.identified = True
                    fs.name = rec.get("name")
                    fs.confidence = rec.get("confidence", 0.0)
                    if fs.name in known:
                        fs.voice_embedding = known[fs.name].embedding

                cd = card_speakers.get(label) or {}
                contact = cd.get("contact")
                if contact:
                    contact.setdefault("uid", "") 
                    fs.contact_info = ContactInfo(**contact)

                speakers_map[label] = fs

        final_segments = []
        full_text_parts = []
        total_words = 0
        speech_dur = 0.0

        trans_words = trans.payload.get("words", []) if trans else []

        for seg in diar_segments:
            label = seg.get("speaker", "unknown")
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            duration = end - start

            sp = speakers_map[label]
            sp.segments_count += 1
            sp.total_duration += duration

            words = []
            text = ""

            for w in trans_words:
                ws, we = w.get("start", 0.0), w.get("end", 0.0)
                if max(start, ws) < min(end, we):
                    words.append(TranscriptionWord(**w))
                    text += w.get("word", "") + " "
                    total_words += 1

            text = text.strip()
            full_text_parts.append(f"[{sp.id}]: {text}")

            final_segments.append(FinalSegment(
                id=len(final_segments) + 1,
                start=start,
                end=end,
                duration=duration,
                speaker=sp.id,
                speaker_label=label,
                text=text,
                words=words,
                confidence=seg.get("confidence", 0.0)
            ))
            speech_dur += duration

        final_transcription = FinalTranscription(
            full_text="\n".join(full_text_parts),
            confidence=trans.payload.get("confidence", 0.0) if trans else 0.0,
            language=trans.payload.get("language", "unknown") if trans else "unknown",
            words=[TranscriptionWord(**w) for w in trans_words]
        )

        final_speakers = list(speakers_map.values())
        ident_cnt = sum(1 for s in final_speakers if s.identified)

        stats = Statistics(
            total_speakers=len(final_speakers),
            identified_speakers=ident_cnt,
            unknown_speakers=len(final_speakers) - ident_cnt,
            total_segments=len(final_segments),
            total_words=total_words,
            speech_duration=speech_dur,
            silence_duration=max(0, audio_metadata.duration - speech_dur)
        )

        version = getattr(self.config, "server", None)
        ver_str = version.version if version and hasattr(version, "version") else "1.0.0"

        return AnnotationResult(
            task_id=task_id,
            version=ver_str,
            created_at=datetime.now(),
            audio_metadata=audio_metadata.dict(),
            processing_info=processing_info.dict(),
            speakers=[fs.dict() for fs in final_speakers], 
            segments=[seg.dict() for seg in final_segments],
            transcription=final_transcription.dict(),
            statistics=stats.dict()
        )

