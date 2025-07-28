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

from .stages import PreprocessingStage, DiarizationStage, TranscriptionStage, RecognitionStage, CardDAVStage
from .schemas import (
    AnnotationResult, AudioMetadata, ProcessingInfo, FinalSpeaker, 
    FinalSegment, FinalTranscription, Statistics, TranscriptionWord
)
from .models_registry import models_registry
from .utils import extract_audio_metadata


class AnnotationService:
    """Основной сервис аннотации аудиофайлов с использованием этапной архитектуры"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        
        # Преобразование конфигурации в Pydantic модель
        if isinstance(config, dict):
            from .config import AppSettings
            self.config = AppSettings(**config)
        else:
            self.config = config
        
        # Инициализация этапов обработки
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
        """
        Полная обработка аудиофайла с аннотацией
        
        Args:
            file_path: Путь к аудиофайлу
            task_id: Идентификатор задачи
            progress_callback: Функция обратного вызова для обновления прогресса
        
        Returns:
            Словарь с полной аннотацией
        """
        
        try:
            await self._update_progress(progress_callback, 0, "Начало обработки аудио")
            
            # Извлечение метаданных аудио
            audio_metadata = extract_audio_metadata(file_path)
            await self._update_progress(progress_callback, 5, "Метаданные аудио извлечены")
            
            # Последовательное выполнение этапов
            context = {}
            stage_progress = [10, 35, 65, 85]  # Прогресс по этапам
            
            for i, stage in enumerate(self.stages):
                start_progress = stage_progress[i]
                end_progress = stage_progress[i+1] if i+1 < len(stage_progress) else 90
                
                await self._update_progress(
                    progress_callback, 
                    start_progress, 
                    f"Начало этапа {stage.stage_name}"
                )
                
                # Создание callback для прогресса этапа
                async def stage_progress_callback(stage_progress_pct: int, message: str):
                    overall_progress = start_progress + int(
                        (end_progress - start_progress) * stage_progress_pct / 100
                    )
                    await self._update_progress(progress_callback, overall_progress, message)
                
                # Выполнение этапа
                stage_result = await stage.process(
                    file_path, task_id, context, stage_progress_callback
                )
                
                # Сохранение результата в контексте
                context[stage.stage_name] = stage_result
                
                if not stage_result.success:
                    self.logger.error(f"Этап {stage.stage_name} завершился с ошибкой: {stage_result.error}")
                    # Продолжаем обработку даже при ошибке одного этапа
                
                await self._update_progress(
                    progress_callback, 
                    end_progress, 
                    f"Этап {stage.stage_name} завершен"
                )
            
            # Сборка финального результата
            await self._update_progress(progress_callback, 95, "Сборка финального результата")
            final_result = self._build_final_annotation(
                task_id, audio_metadata, context
            )
            
            await self._update_progress(progress_callback, 100, "Аннотация завершена")
            
            return final_result.dict()
            
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
        self.logger.info(f"Прогресс {progress}%: {message}")


    def _build_final_annotation(
        self,
        task_id: str,
        audio_metadata: AudioMetadata,
        context: Dict[str, Any]
    ) -> AnnotationResult:
        """Сборка финального JSON с аннотацией"""
        
        # Извлечение результатов этапов
        diarization_result = context.get("diarization")
        transcription_result = context.get("transcription")
        recognition_result = context.get("recognition")
        carddav_result = context.get("carddav")
        
        # Создание информации о процессе
        processing_info = ProcessingInfo(
            diarization_model=diarization_result.model_info if diarization_result else {},
            transcription_model=transcription_result.model_info if transcription_result else {},
            recognition_model=recognition_result.model_info if recognition_result else {},
            processing_time={
                "diarization": diarization_result.processing_time if diarization_result else 0.0,
                "transcription": transcription_result.processing_time if transcription_result else 0.0,
                "recognition": recognition_result.processing_time if recognition_result else 0.0,
                "carddav": carddav_result.processing_time if carddav_result else 0.0
            }
        )
        
        # Обработка спикеров
        speakers_map = {}
        speaker_id = 0
        
        diarization_segments = diarization_result.payload.get("segments", []) if diarization_result else []
        recognition_speakers = recognition_result.payload.get("speakers", {}) if recognition_result else {}
        carddav_speakers = carddav_result.payload.get("speakers", {}) if carddav_result else {}
        
        for segment in diarization_segments:
            speaker_label = segment.get("speaker", "unknown")
            
            if speaker_label not in speakers_map:
                speaker_id += 1
                speaker_info = FinalSpeaker(
                    id=f"speaker_{speaker_id:02d}",
                    label=speaker_label,
                    segments_count=0,
                    total_duration=0.0
                )
                
                # Добавление информации о распознавании
                recognition_info = recognition_speakers.get(speaker_label, {})
                if recognition_info:
                    speaker_info.identified = recognition_info.get("identified", False)
                    speaker_info.name = recognition_info.get("name")
                    speaker_info.confidence = recognition_info.get("confidence", 0.0)
                
                # Добавление информации из CardDAV
                carddav_info = carddav_speakers.get(speaker_label, {})
                if carddav_info and carddav_info.get("contact"):
                    speaker_info.contact_info = carddav_info["contact"]
                
                speakers_map[speaker_label] = speaker_info
        
        # Обработка сегментов и транскрипции
        final_segments = []
        full_text_parts = []
        total_words = 0
        speech_duration = 0.0
        
        transcription_segments = transcription_result.payload.get("segments", []) if transcription_result else []
        transcription_words = transcription_result.payload.get("words", []) if transcription_result else []
        
        for segment in diarization_segments:
            speaker_label = segment.get("speaker", "unknown")
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            duration = end_time - start_time
            
            # Обновление статистики спикера
            if speaker_label in speakers_map:
                speaker_info = speakers_map[speaker_label]
                speaker_info.segments_count += 1
                speaker_info.total_duration += duration
            
            # Поиск соответствующих слов из транскрипции
            segment_words = []
            segment_text = ""
            
            for word_info in transcription_words:
                word_start = word_info.get("start", 0.0)
                word_end = word_info.get("end", 0.0)
                
                # Проверка пересечения временных интервалов
                if self._intervals_overlap(start_time, end_time, word_start, word_end):
                    segment_words.append(TranscriptionWord(**word_info))
                    segment_text += word_info.get("word", "") + " "
                    total_words += 1
            
            segment_text = segment_text.strip()
            
            if speaker_label in speakers_map:
                speaker_id_str = speakers_map[speaker_label].id
                full_text_parts.append(f"[{speaker_id_str}]: {segment_text}")
            
            # Создание сегмента
            final_segment = FinalSegment(
                id=len(final_segments) + 1,
                start=start_time,
                end=end_time,
                duration=duration,
                speaker=speakers_map[speaker_label].id if speaker_label in speakers_map else "unknown",
                speaker_label=speaker_label,
                text=segment_text,
                words=segment_words,
                confidence=segment.get("confidence", 0.0)
            )
            
            final_segments.append(final_segment)
            speech_duration += duration
        
        # Финализация транскрипции
        final_transcription = FinalTranscription(
            full_text="\n".join(full_text_parts),
            words=[TranscriptionWord(**word) for word in transcription_words],
            confidence=transcription_result.payload.get("confidence", 0.0) if transcription_result else 0.0,
            language=transcription_result.payload.get("language", "unknown") if transcription_result else "unknown"
        )
        
        # Вычисление статистики
        speakers_list = list(speakers_map.values())
        identified_speakers = sum(1 for s in speakers_list if s.identified)
        
        statistics = Statistics(
            total_speakers=len(speakers_list),
            identified_speakers=identified_speakers,
            unknown_speakers=len(speakers_list) - identified_speakers,
            total_segments=len(final_segments),
            total_words=total_words,
            speech_duration=speech_duration,
            silence_duration=max(0, audio_metadata.duration - speech_duration)
        )
        
        return AnnotationResult(
            task_id=task_id,
            created_at=datetime.now(),
            audio_metadata=audio_metadata,
            processing_info=processing_info,
            speakers=speakers_list,
            segments=final_segments,
            transcription=final_transcription,
            statistics=statistics
        )
    
    def _intervals_overlap(self, start1: float, end1: float, start2: float, end2: float) -> bool:
        """Проверка пересечения временных интервалов"""
        return max(start1, start2) < min(end1, end2)
    
    async def cleanup(self):
        """Очистка ресурсов всех этапов"""
        for stage in self.stages:
            try:
                await stage.cleanup()
            except Exception as e:
                self.logger.error(f"Ошибка очистки этапа {stage.stage_name}: {e}")
