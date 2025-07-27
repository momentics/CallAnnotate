# -*- coding: utf-8 -*-
"""
Pydantic схемы для CallAnnotate API и внутренних структур данных

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class TaskStatus(str, Enum):
    """Статус задачи"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioMetadata(BaseModel):
    """Метаданные аудиофайла"""
    filename: str
    duration: float = Field(..., ge=0, description="Длительность в секундах")
    sample_rate: int = Field(..., gt=0, description="Частота дискретизации")
    channels: int = Field(..., gt=0, description="Количество каналов")
    format: str = Field(..., description="Формат файла")
    bitrate: Optional[int] = Field(None, gt=0, description="Битрейт")
    size_bytes: int = Field(..., ge=0, description="Размер файла в байтах")


class BaseStageResult(BaseModel):
    """Базовый результат этапа обработки"""
    stage_name: str
    processing_time: float = Field(..., ge=0, description="Время обработки в секундах")
    model_info: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class DiarizationSegment(BaseModel):
    """Сегмент диаризации"""
    start: float = Field(..., ge=0, description="Начало сегмента в секундах")
    end: float = Field(..., gt=0, description="Конец сегмента в секундах")
    duration: float = Field(..., gt=0, description="Длительность сегмента")
    speaker: str = Field(..., description="Идентификатор спикера")
    confidence: float = Field(0.9, ge=0.0, le=1.0, description="Уверенность диаризации")
    
    @validator('end')
    def end_after_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError('end должен быть больше start')
        return v
    
    @validator('duration')
    def duration_matches(cls, v, values):
        if 'start' in values and 'end' in values:
            expected_duration = values['end'] - values['start']
            if abs(v - expected_duration) > 0.001:  # Допуск на округление
                raise ValueError('duration должен соответствовать end - start')
        return v


class DiarizationResult(BaseStageResult):
    """Результат диаризации"""
    stage_name: str = "diarization"
    segments: List[DiarizationSegment] = Field(default_factory=list)
    speakers: List[str] = Field(default_factory=list)
    speaker_stats: Dict[str, Dict[str, Union[float, int]]] = Field(default_factory=dict)
    total_segments: int = Field(0, ge=0)
    total_speakers: int = Field(0, ge=0)


class TranscriptionWord(BaseModel):
    """Слово с временными метками"""
    start: float = Field(..., ge=0, description="Начало слова в секундах")
    end: float = Field(..., gt=0, description="Конец слова в секундах")
    word: str = Field(..., description="Текст слова")
    probability: float = Field(..., ge=0.0, le=1.0, description="Вероятность распознавания")


class TranscriptionSegment(BaseModel):
    """Сегмент транскрипции"""
    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    text: str = Field(..., description="Текст сегмента")
    no_speech_prob: float = Field(0.0, ge=0.0, le=1.0)
    avg_logprob: float = Field(..., description="Средняя логарифмическая вероятность")
    speaker: Optional[str] = Field(None, description="Спикер (если привязан к диаризации)")
    speaker_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class TranscriptionResult(BaseStageResult):
    """Результат транскрипции"""
    stage_name: str = "transcription"
    text: str = Field("", description="Полный текст")
    language: str = Field("unknown", description="Определенный язык")
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    words: List[TranscriptionWord] = Field(default_factory=list)
    total_words: int = Field(0, ge=0)
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Средняя уверенность")


class SpeakerRecognition(BaseModel):
    """Результат распознавания спикера"""
    identified: bool = Field(False, description="Спикер идентифицирован")
    name: Optional[str] = Field(None, description="Имя спикера")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Уверенность распознавания")
    reason: Optional[str] = Field(None, description="Причина не распознавания")


class RecognitionResult(BaseStageResult):
    """Результат распознавания голосов"""
    stage_name: str = "recognition"
    speakers: Dict[str, SpeakerRecognition] = Field(default_factory=dict)
    total_processed: int = Field(0, ge=0)
    identified_count: int = Field(0, ge=0)


class ContactInfo(BaseModel):
    """Информация о контакте из CardDAV"""
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phones: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    organization: Optional[str] = None


class EnhancedSpeakerInfo(SpeakerRecognition):
    """Расширенная информация о спикере с контактными данными"""
    contact: Optional[ContactInfo] = None


class CardDAVResult(BaseStageResult):
    """Результат связывания с CardDAV"""
    stage_name: str = "carddav"
    speakers: Dict[str, EnhancedSpeakerInfo] = Field(default_factory=dict)
    contacts_found: int = Field(0, ge=0)
    total_searched: int = Field(0, ge=0)


class ProcessingInfo(BaseModel):
    """Информация о процессе обработки"""
    diarization_model: Dict[str, Any] = Field(default_factory=dict)
    transcription_model: Dict[str, Any] = Field(default_factory=dict)
    recognition_model: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Dict[str, float] = Field(default_factory=dict)


class FinalSpeaker(BaseModel):
    """Финальная информация о спикере"""
    id: str = Field(..., description="Внутренний ID спикера")
    label: str = Field(..., description="Метка из диаризации")
    segments_count: int = Field(0, ge=0)
    total_duration: float = Field(0.0, ge=0.0)
    identified: bool = False
    name: Optional[str] = None
    contact_info: Optional[ContactInfo] = None
    voice_embedding: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class FinalSegment(BaseModel):
    """Финальный сегмент с полной информацией"""
    id: int = Field(..., ge=1)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    duration: float = Field(..., gt=0.0)
    speaker: str = Field(..., description="ID спикера")
    speaker_label: str = Field(..., description="Оригинальная метка спикера")
    text: str = Field("", description="Текст сегмента")
    words: List[TranscriptionWord] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class FinalTranscription(BaseModel):
    """Финальная транскрипция"""
    full_text: str = Field("", description="Полный текст с разметкой спикеров")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    language: str = Field("unknown")
    words: List[TranscriptionWord] = Field(default_factory=list)


class Statistics(BaseModel):
    """Статистика обработки"""
    total_speakers: int = Field(0, ge=0)
    identified_speakers: int = Field(0, ge=0)
    unknown_speakers: int = Field(0, ge=0)
    total_segments: int = Field(0, ge=0)
    total_words: int = Field(0, ge=0)
    speech_duration: float = Field(0.0, ge=0.0)
    silence_duration: float = Field(0.0, ge=0.0)


class AnnotationResult(BaseModel):
    """Финальный результат аннотации"""
    task_id: str
    version: str = "1.0.0"
    created_at: datetime
    audio_metadata: AudioMetadata
    processing_info: ProcessingInfo
    speakers: List[FinalSpeaker] = Field(default_factory=list)
    segments: List[FinalSegment] = Field(default_factory=list)
    transcription: FinalTranscription
    statistics: Statistics
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# API схемы
class TaskCreateResponse(BaseModel):
    """Ответ на создание задачи"""
    job_id: str
    status: TaskStatus
    message: str
    progress_url: str
    result_url: str


class TaskStatusResponse(BaseModel):
    """Ответ статуса задачи"""
    job_id: str
    status: TaskStatus
    message: str
    result: Optional[AnnotationResult] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str
    version: str
    queue_length: int = Field(0, ge=0)
    active_tasks: int = Field(0, ge=0)


class InfoResponse(BaseModel):
    """Информация о сервисе"""
    service: str
    version: str
    description: str
    max_file_size: int = Field(..., gt=0)


# WebSocket схемы
class WebSocketMessage(BaseModel):
    """Базовое WebSocket сообщение"""
    type: str
    timestamp: Optional[datetime] = None


class TaskProgressMessage(WebSocketMessage):
    """Сообщение о прогрессе задачи"""
    type: str = "task_progress"
    task_id: str
    progress: int = Field(..., ge=0, le=100)
    message: str


class TaskCompletedMessage(WebSocketMessage):
    """Сообщение о завершении задачи"""
    type: str = "task_completed"
    task_id: str
    status: TaskStatus
    message: str
    result: Optional[AnnotationResult] = None


class TaskFailedMessage(WebSocketMessage):
    """Сообщение об ошибке задачи"""
    type: str = "task_failed"
    task_id: str
    status: TaskStatus
    message: str
    error: str
