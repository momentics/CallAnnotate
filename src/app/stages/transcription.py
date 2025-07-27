# -*- coding: utf-8 -*-
"""
Этап транскрипции аудио для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import whisper
from typing import Dict, Any, Optional, Callable, List

from .base import BaseStage


class TranscriptionStage(BaseStage):
    """Этап транскрипции на основе OpenAI Whisper"""
    
    @property
    def stage_name(self) -> str:
        return "transcription"
    
    async def _initialize(self):
        """Инициализация модели транскрипции"""
        model_size = self.config.get("model_size", "base")
        device = self.config.get("device", "cpu")
        
        self.logger.info(f"Загрузка модели Whisper: {model_size}")
        
        if self.models_registry:
            cache_key = f"whisper_{model_size}_{device}"
            self.model = self.models_registry.get_model(
                cache_key,
                lambda: whisper.load_model(model_size, device=device)
            )
        else:
            self.model = whisper.load_model(model_size, device=device)
        
        self.model_size = model_size
        self.device = device
        
        self.logger.info("Модель транскрипции загружена успешно")
    
    async def _process_impl(
        self, 
        file_path: str, 
        task_id: str, 
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Выполнение транскрипции"""
        
        if progress_callback:
            await progress_callback(10, "Начало транскрипции")
        
        # Настройки транскрипции
        options_config = self.config.get("options", {})
        transcribe_options = {
            "language": options_config.get("language"),  # None для автоопределения
            "task": options_config.get("task", "transcribe"),
            "temperature": options_config.get("temperature", 0.0),
            "best_of": options_config.get("best_of", 5),
            "beam_size": options_config.get("beam_size", 5),
            "word_timestamps": True,  # Всегда включаем временные метки слов
            "verbose": False
        }
        
        # Удаляем None значения
        transcribe_options = {k: v for k, v in transcribe_options.items() if v is not None}
        
        if progress_callback:
            await progress_callback(30, "Выполнение транскрипции")
        
        # Выполнение транскрипции
        result = self.model.transcribe(file_path, **transcribe_options)
        
        if progress_callback:
            await progress_callback(80, "Обработка результатов транскрипции")
        
        # Извлечение слов с временными метками
        words = []
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        words.append({
                            "start": round(word_info.get("start", 0.0), 3),
                            "end": round(word_info.get("end", 0.0), 3),
                            "word": word_info.get("word", "").strip(),
                            "probability": round(word_info.get("probability", 0.0), 3)
                        })
        
        # Сегменты с текстом
        segments = []
        if "segments" in result:
            for segment in result["segments"]:
                segments.append({
                    "start": round(segment.get("start", 0.0), 3),
                    "end": round(segment.get("end", 0.0), 3),
                    "text": segment.get("text", "").strip(),
                    "no_speech_prob": round(segment.get("no_speech_prob", 0.0), 3),
                    "avg_logprob": round(segment.get("avg_logprob", 0.0), 3)
                })
        
        # Сопоставление с диаризацией если доступна
        enhanced_segments = self._align_with_diarization(
            segments, previous_results.get("segments", [])
        )
        
        if progress_callback:
            await progress_callback(100, "Транскрипция завершена")
        
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", "unknown"),
            "segments": enhanced_segments,
            "words": words,
            "total_words": len(words),
            "confidence": self._calculate_average_confidence(words)
        }
    
    def _align_with_diarization(
        self, 
        transcription_segments: List[Dict], 
        diarization_segments: List[Dict]
    ) -> List[Dict]:
        """Сопоставление транскрипции с диаризацией"""
        if not diarization_segments:
            return transcription_segments
        
        enhanced_segments = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            
            # Поиск наиболее подходящего сегмента диаризации
            best_speaker = None
            best_overlap = 0.0
            
            for diar_seg in diarization_segments:
                diar_start = diar_seg["start"]
                diar_end = diar_seg["end"]
                
                # Вычисление пересечения
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    trans_duration = trans_end - trans_start
                    
                    if trans_duration > 0:
                        overlap_ratio = overlap_duration / trans_duration
                        
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_speaker = diar_seg["speaker"]
            
            # Добавление информации о спикере
            enhanced_seg = trans_seg.copy()
            if best_speaker and best_overlap > 0.3:  # Порог пересечения 30%
                enhanced_seg["speaker"] = best_speaker
                enhanced_seg["speaker_confidence"] = round(best_overlap, 3)
            
            enhanced_segments.append(enhanced_seg)
        
        return enhanced_segments
    
    def _calculate_average_confidence(self, words: List[Dict]) -> float:
        """Вычисление средней уверенности распознавания"""
        if not words:
            return 0.0
        
        confidences = [word.get("probability", 0.0) for word in words]
        return round(sum(confidences) / len(confidences), 3)
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Информация о модели транскрипции"""
        return {
            "stage": self.stage_name,
            "model_size": getattr(self, 'model_size', 'unknown'),
            "device": getattr(self, 'device', 'unknown'),
            "framework": "OpenAI Whisper"
        }
