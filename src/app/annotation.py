# src/app/annotation.py
# -*- coding: utf-8 -*-
"""
Сервис аннотации аудио для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0

Исправлено: теперь сохраняется весь результат StageResult для корректного доступа к model_info.
"""
import logging
import inspect
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Optional

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

    async def process_audio(
        self,
        file_path: str,
        task_id: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
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
                # Сохраняем весь объект StageResult, а не только payload
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
        diar_result = context.get("diarization")
        trans_result = context.get("transcription")
        recog_result = context.get("recognition")
        carddav_result = context.get("carddav")

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

        speakers_map: Dict[str, FinalSpeaker] = {}
        counter = 0

        diar_segments = diar_result.payload.get("segments", []) if diar_result else []
        recog_speakers = recog_result.payload.get("speakers", {}) if recog_result else {}
        card_speakers = carddav_result.payload.get("speakers", {}) if carddav_result else {}

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

        trans_words = trans_result.payload.get("words", []) if trans_result else []

        for seg in diar_segments:
            label = seg.get("speaker", "unknown")
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            duration = round((end - start), 3)

            sp = speakers_map[label]
            sp.segments_count += 1
            sp.total_duration = round((sp.total_duration + duration), 3)

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
                duration=round(duration, 3),
                speaker=sp.id,
                speaker_label=label,
                text=text,
                words=words,
                confidence=seg.get("confidence", 0.0)
            ))
            speech_dur += duration

        final_transcription = FinalTranscription(
            full_text="\n".join(full_text_parts),
            confidence=trans_result.payload.get("confidence", 0.0) if trans_result else 0.0,
            language=trans_result.payload.get("language", "unknown") if trans_result else "unknown",
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
            speech_duration=round(speech_dur, 3),
            silence_duration=round(max(0, audio_metadata.duration - round(speech_dur, 3)), 3)
        )

        version = getattr(self.config, "server", None)
        ver_str = version.version if version and hasattr(version, "version") else "1.0.0"

        return AnnotationResult(
            task_id=task_id,
            version=ver_str,
            created_at=datetime.now(timezone.utc).isoformat(),
            audio_metadata=audio_metadata.dict(),
            processing_info=processing_info.dict(),
            speakers=[fs.dict() for fs in final_speakers],
            segments=[seg.dict() for seg in final_segments],
            transcription=final_transcription.dict(),
            statistics=stats.dict()
        )
