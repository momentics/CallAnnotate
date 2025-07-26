# -*- coding: utf-8 -*-
"""
Сервис аннотации аудио для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Callable, Optional

from .diarization import DiarizationService
from .transcription import TranscriptionService
from .recognition import RecognitionService
from .carddav_client import CardDAVClient
from .utils import AudioMetadata, extract_audio_metadata


class AnnotationService:
    """Основной сервис аннотации аудиофайлов"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Инициализация сервисов
        self.diarization_service = DiarizationService(config.get('diarization', {}))
        self.transcription_service = TranscriptionService(config.get('transcription', {}))
        self.recognition_service = RecognitionService(config.get('recognition', {}))
        self.carddav_client = CardDAVClient(config.get('carddav', {}))
        
        # Пути для промежуточных результатов
        self.volume_path = Path(config.get('volume_path', '/volume'))
        self.intermediate_path = self.volume_path / 'intermediate'
    
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
            
            # Этап 1: Диаризация говорящих
            await self._update_progress(progress_callback, 10, "Начало диаризации")
            diarization_result = await self.diarization_service.process(file_path, task_id)
            await self._save_intermediate_result('diarization', task_id, diarization_result)
            await self._update_progress(progress_callback, 30, "Диаризация завершена")
            
            # Этап 2: Транскрипция
            await self._update_progress(progress_callback, 35, "Начало транскрипции")
            transcription_result = await self.transcription_service.process(
                file_path, task_id, diarization_result
            )
            await self._save_intermediate_result('transcription', task_id, transcription_result)
            await self._update_progress(progress_callback, 60, "Транскрипция завершена")
            
            # Этап 3: Распознавание голосов
            await self._update_progress(progress_callback, 65, "Начало распознавания голосов")
            recognition_result = await self.recognition_service.process(
                file_path, task_id, diarization_result
            )
            await self._save_intermediate_result('recognition', task_id, recognition_result)
            await self._update_progress(progress_callback, 80, "Распознавание голосов завершено")
            
            # Этап 4: Связывание с CardDAV
            await self._update_progress(progress_callback, 85, "Поиск контактов в CardDAV")
            carddav_result = await self.carddav_client.link_contacts(recognition_result)
            await self._save_intermediate_result('carddav', task_id, carddav_result)
            await self._update_progress(progress_callback, 90, "Связывание контактов завершено")
            
            # Этап 5: Сборка финального результата
            await self._update_progress(progress_callback, 95, "Сборка финального результата")
            final_result = await self._build_final_annotation(
                task_id, audio_metadata, diarization_result,
                transcription_result, recognition_result, carddav_result
            )
            
            await self._update_progress(progress_callback, 100, "Аннотация завершена")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке аудио {file_path}: {e}")
            raise
    
    async def _update_progress(
        self,
        callback: Optional[Callable[[int, str], None]],
        progress: int,
        message: str
    ):
        """Обновление прогресса выполнения"""
        if callback:
            await callback(progress, message)
        self.logger.info(f"Прогресс {progress}%: {message}")
    
    async def _save_intermediate_result(
        self,
        stage: str,
        task_id: str,
        result: Dict[str, Any]
    ):
        """Сохранение промежуточного результата"""
        try:
            stage_path = self.intermediate_path / stage
            stage_path.mkdir(parents=True, exist_ok=True)
            
            result_file = stage_path / f"{task_id}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Промежуточный результат {stage} сохранен для задачи {task_id}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении промежуточного результата {stage}: {e}")
    
    async def _build_final_annotation(
        self,
        task_id: str,
        audio_metadata: AudioMetadata,
        diarization_result: Dict[str, Any],
        transcription_result: Dict[str, Any],
        recognition_result: Dict[str, Any],
        carddav_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Сборка финального JSON с аннотацией"""
        
        # Основная структура результата
        final_annotation = {
            "task_id": task_id,
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "audio_metadata": {
                "filename": audio_metadata.filename,
                "duration": audio_metadata.duration,
                "sample_rate": audio_metadata.sample_rate,
                "channels": audio_metadata.channels,
                "format": audio_metadata.format,
                "bitrate": audio_metadata.bitrate,
                "size_bytes": audio_metadata.size_bytes
            },
            "processing_info": {
                "diarization_model": diarization_result.get("model_info", {}),
                "transcription_model": transcription_result.get("model_info", {}),
                "recognition_model": recognition_result.get("model_info", {}),
                "processing_time": {
                    "diarization": diarization_result.get("processing_time", 0),
                    "transcription": transcription_result.get("processing_time", 0),
                    "recognition": recognition_result.get("processing_time", 0),
                    "carddav": carddav_result.get("processing_time", 0)
                }
            },
            "speakers": [],
            "segments": [],
            "transcription": {
                "full_text": "",
                "confidence": 0.0,
                "language": transcription_result.get("language", "unknown"),
                "words": []
            },
            "statistics": {
                "total_speakers": 0,
                "identified_speakers": 0,
                "unknown_speakers": 0,
                "total_segments": 0,
                "total_words": 0,
                "speech_duration": 0.0,
                "silence_duration": 0.0
            }
        }
        
        # Обработка спикеров
        speakers_map = {}
        speaker_id = 0
        
        for segment in diarization_result.get("segments", []):
            speaker_label = segment.get("speaker", "unknown")
            
            if speaker_label not in speakers_map:
                speaker_id += 1
                speaker_info = {
                    "id": f"speaker_{speaker_id:02d}",
                    "label": speaker_label,
                    "segments_count": 0,
                    "total_duration": 0.0,
                    "identified": False,
                    "name": None,
                    "contact_info": None,
                    "voice_embedding": None,
                    "confidence": 0.0
                }
                
                # Добавление информации о распознавании
                recognition_info = recognition_result.get("speakers", {}).get(speaker_label, {})
                if recognition_info:
                    speaker_info.update({
                        "identified": recognition_info.get("identified", False),
                        "name": recognition_info.get("name"),
                        "confidence": recognition_info.get("confidence", 0.0),
                        "voice_embedding": recognition_info.get("embedding_path")
                    })
                
                # Добавление информации из CardDAV
                carddav_info = carddav_result.get("speakers", {}).get(speaker_label, {})
                if carddav_info:
                    speaker_info["contact_info"] = carddav_info
                
                speakers_map[speaker_label] = speaker_info
                final_annotation["speakers"].append(speaker_info)
        
        # Обработка сегментов и транскрипции
        full_text_parts = []
        total_words = 0
        speech_duration = 0.0
        
        for segment in diarization_result.get("segments", []):
            speaker_label = segment.get("speaker", "unknown")
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            duration = end_time - start_time
            
            # Обновление статистики спикера
            speaker_info = speakers_map[speaker_label]
            speaker_info["segments_count"] += 1
            speaker_info["total_duration"] += duration
            
            # Поиск соответствующих слов из транскрипции
            segment_words = []
            segment_text = ""
            
            for word_info in transcription_result.get("words", []):
                word_start = word_info.get("start", 0.0)
                word_end = word_info.get("end", 0.0)
                
                # Проверка пересечения временных интервалов
                if self._intervals_overlap(start_time, end_time, word_start, word_end):
                    segment_words.append(word_info)
                    segment_text += word_info.get("word", "") + " "
                    total_words += 1
            
            segment_text = segment_text.strip()
            full_text_parts.append(f"[{speaker_info['id']}]: {segment_text}")
            
            # Создание сегмента
            annotation_segment = {
                "id": len(final_annotation["segments"]) + 1,
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "speaker": speaker_info["id"],
                "speaker_label": speaker_label,
                "text": segment_text,
                "words": segment_words,
                "confidence": segment.get("confidence", 0.0)
            }
            
            final_annotation["segments"].append(annotation_segment)
            speech_duration += duration
        
        # Финализация транскрипции
        final_annotation["transcription"]["full_text"] = "\n".join(full_text_parts)
        final_annotation["transcription"]["words"] = transcription_result.get("words", [])
        final_annotation["transcription"]["confidence"] = transcription_result.get("confidence", 0.0)
        
        # Вычисление статистики
        total_speakers = len(final_annotation["speakers"])
        identified_speakers = sum(1 for s in final_annotation["speakers"] if s["identified"])
        unknown_speakers = total_speakers - identified_speakers
        
        final_annotation["statistics"] = {
            "total_speakers": total_speakers,
            "identified_speakers": identified_speakers,
            "unknown_speakers": unknown_speakers,
            "total_segments": len(final_annotation["segments"]),
            "total_words": total_words,
            "speech_duration": speech_duration,
            "silence_duration": max(0, audio_metadata.duration - speech_duration)
        }
        
        return final_annotation
    
    def _intervals_overlap(self, start1: float, end1: float, start2: float, end2: float) -> bool:
        """Проверка пересечения временных интервалов"""
        return max(start1, start2) < min(end1, end2)
