# src/app/schemas.py
# -*- coding: utf-8 -*-
"""
Pydantic схемы для CallAnnotate API и внутренних структур данных

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CreateJobRequest(BaseModel):
    filename: str = Field(..., description="Имя файла в /volume/incoming/")
    priority: int = Field(5, ge=1, le=10, description="Приоритет задачи (1-10)")


class FileInfo(BaseModel):
    """Информация о файле"""
    filename: str
    path: str
    size_bytes: Optional[int] = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: TaskStatus
    message: str
    created_at: datetime
    file_info: FileInfo
    progress_url: str
    result_url: str

class ProgressInfo(BaseModel):
    """Информация о прогрессе"""
    percentage: int = Field(..., ge=0, le=100)
    current_stage: str
    stage_progress: Optional[int] = Field(None, ge=0, le=100)
    stages_completed: List[str] = Field(default_factory=list)
    stages_remaining: List[str] = Field(default_factory=list)
    estimated_completion: Optional[datetime] = None


class JobTimestamps(BaseModel):
    """Временные метки задачи"""
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: TaskStatus
    message: str
    progress: Optional[ProgressInfo] = None
    timestamps: JobTimestamps
    file_info: Optional[FileInfo] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None




class BaseStageResult(BaseModel):
    stage_name: str
    processing_time: float = Field(..., ge=0)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class DiarizationSegment(BaseModel):
    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    duration: float = Field(..., gt=0)
    speaker: str
    confidence: float = Field(0.9, ge=0.0, le=1.0)

    @validator('end')
    def end_after_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError('end должен быть больше start')
        return v

    @validator('duration')
    def duration_matches(cls, v, values):
        if 'start' in values and 'end' in values:
            expected = values['end'] - values['start']
            if abs(v - expected) > 0.001:
                raise ValueError('duration должен соответствовать end - start')
        return v




class SpeakerRecognition(BaseModel):
    identified: bool = Field(False)
    name: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: Optional[str] = None

class ContactInfo(BaseModel):
    """Информация о контакте из CardDAV"""
    uid: str = Field("", description="UID контакта в CardDAV")
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phones: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    organization: Optional[str] = None





    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# WebSocket схемы
class WebSocketMessage(BaseModel):
    """Базовое WebSocket сообщение"""
    type: str
    timestamp: Optional[datetime] = None


class JobProgressMessage(WebSocketMessage):
    """Сообщение о прогрессе задачи"""
    type: str = "job_progress"
    job_id: str
    progress: ProgressInfo
    status: TaskStatus


class JobCompletedMessage(WebSocketMessage):
    """Сообщение о завершении задачи"""
    type: str = "job_completed"
    job_id: str
    status: TaskStatus
    result_url: str


class JobFailedMessage(WebSocketMessage):
    """Сообщение об ошибке задачи"""
    type: str = "job_failed"
    job_id: str
    status: TaskStatus
    error: Dict[str, str]


# Health Check схемы
class HealthResponse(BaseModel):
    """Ответ health check"""
    status: str
    version: str
    uptime: int = Field(0, ge=0)
    queue_length: int = Field(0, ge=0)
    active_tasks: int = Field(0, ge=0)
    components: Dict[str, str] = Field(default_factory=dict)


class InfoResponse(BaseModel):
    """Информация о сервисе"""
    service: str
    version: str
    description: str
    max_file_size: int = Field(..., gt=0)
    supported_formats: List[str] = Field(default_factory=list)
    processing_mode: str
    volume_paths: Dict[str, str] = Field(default_factory=dict)
    api_endpoints: Dict[str, str] = Field(default_factory=dict)


class QueueStatusResponse(BaseModel):
    """Статус очереди"""
    queue_length: int = Field(0, ge=0)
    processing_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    queued_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    average_processing_time: float = Field(0.0, ge=0.0)
    system_load: Dict[str, float] = Field(default_factory=dict)


# Схемы ошибок
class ErrorDetail(BaseModel):
    """Детали ошибки"""
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Ответ с ошибкой"""
    error: ErrorDetail



class VoiceInfoBase(BaseModel):
    name: str = Field(..., description="Уникальное имя голоса")
    embedding: str = Field(..., description="Путь к файлу эмбеддинга")

class VoiceInfoCreate(VoiceInfoBase):
    pass

class VoiceInfo(VoiceInfoBase):
    class Config:
        from_attributes = True


class ContactCreate(BaseModel):
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phones: List[str] = []
    emails: List[str] = []
    organization: Optional[str] = None

class ContactUpdate(BaseModel):
    full_name: Optional[str] = None
    phones: List[str] = []
    emails: List[str] = []
    organization: Optional[str] = None

#------------------
class TranscriptionWord(BaseModel):
    """Слово с временными метками"""
    start: float = Field(..., ge=0, description="Начало слова в секундах")
    end: float = Field(..., gt=0, description="Конец слова в секундах")
    word: str = Field(..., description="Текст слова")
    probability: float = Field(..., ge=0.0, le=1.0, description="Вероятность распознавания")
    speaker: Optional[str] = Field(None, description="ID спикера, говорившего слово")
    speaker_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Уверенность привязки слова к спикеру")


class TranscriptionSegment(BaseModel):
    """Сегмент транскрипции с метаданными"""
    start: float = Field(..., ge=0, description="Начало сегмента в секундах")
    end: float = Field(..., gt=0, description="Конец сегмента в секундах")
    text: str = Field(..., description="Текст сегмента")
    speaker: Optional[str] = Field(None, description="ID спикера из этапа диаризации")
    speaker_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Уверенность спикера")
    no_speech_prob: Optional[float] = Field(None, ge=0.0, le=1.0, description="Вероятность отсутствия речи")
    avg_logprob: Optional[float] = Field(None, description="Средняя логарифмическая вероятность")
    confidence: Optional[float] = Field(None, description="Средняя вероятность слов сегмента")


class FinalTranscription(BaseModel):
    """Финальная транскрипция"""
    full_text: str = Field("", description="Полный текст с разметкой спикеров")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    language: str = Field("unknown")
    segments: List[TranscriptionSegment] = Field(default_factory=list, description="Список сегментов транскрипции")
    #words: List[TranscriptionWord] = Field(default_factory=list, description="Детализированные слова")


class FinalSpeaker(BaseModel):
    """Финальная информация о спикере"""
    id: str
    label: str
    segments_count: int
    total_duration: float
    identified: bool
    name: Optional[str] = None
    contact_info: Optional[Any] = None
    voice_embedding: Optional[str] = None
    confidence: float


class FinalSegment(BaseModel):
    """Финальный сегмент с полной информацией"""
    id: int
    start: float
    end: float
    duration: float
    speaker: str
    speaker_label: str
    text: str
    words: List[TranscriptionWord]
    confidence: float


class AudioMetadata(BaseModel):
    filename: str
    duration: float
    sample_rate: int
    channels: int
    format: str
    bitrate: Optional[int]
    size_bytes: int


class ProcessingInfo(BaseModel):
    diarization_model: Dict[str, Any]
    transcription_model: Dict[str, Any]
    recognition_model: Dict[str, Any]
    processing_time: Dict[str, float]


class Statistics(BaseModel):
    total_speakers: int
    identified_speakers: int
    unknown_speakers: int
    total_segments: int
    total_words: int
    speech_duration: float
    silence_duration: float


class AnnotationResult(BaseModel):
    task_id: str
    version: str
    created_at: datetime
    audio_metadata: AudioMetadata
    processing_info: ProcessingInfo
    speakers: List[FinalSpeaker]
    segments: List[FinalSegment]
    transcription: FinalTranscription
    statistics: Statistics

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
