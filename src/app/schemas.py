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
    percentage: int = Field(..., ge=0, le=100)
    current_stage: str
    stage_progress: Optional[int] = Field(None, ge=0, le=100)
    stages_completed: List[str] = Field(default_factory=list)
    stages_remaining: List[str] = Field(default_factory=list)
    estimated_completion: Optional[datetime] = None


class JobTimestamps(BaseModel):
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


class AudioMetadata(BaseModel):
    filename: str
    duration: float = Field(..., ge=0)
    sample_rate: int = Field(..., gt=0)
    channels: int = Field(..., gt=0)
    format: str
    bitrate: Optional[int] = None
    size_bytes: int = Field(..., ge=0)


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


class TranscriptionWord(BaseModel):
    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    word: str
    probability: float = Field(..., ge=0.0, le=1.0)


class TranscriptionSegment(BaseModel):
    start: float = Field(..., ge=0)
    end: float = Field(..., gt=0)
    text: str
    no_speech_prob: float = Field(0.0, ge=0.0, le=1.0)
    avg_logprob: float
    speaker: Optional[str] = None
    speaker_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class SpeakerRecognition(BaseModel):
    identified: bool = Field(False)
    name: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: Optional[str] = None


class ContactInfo(BaseModel):
    uid: str = Field("", description="UID контакта в CardDAV")
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phones: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    organization: Optional[str] = None


class ProcessingInfo(BaseModel):
    diarization_model: Dict[str, Any] = Field(default_factory=dict)
    transcription_model: Dict[str, Any] = Field(default_factory=dict)
    recognition_model: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Dict[str, float] = Field(default_factory=dict)


class FinalSpeaker(BaseModel):
    id: str
    label: str
    segments_count: int = Field(0, ge=0)
    total_duration: float = Field(0.0, ge=0.0)
    identified: bool = False
    name: Optional[str] = None
    contact_info: Optional[ContactInfo] = None
    voice_embedding: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class FinalSegment(BaseModel):
    id: int = Field(..., ge=1)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)
    duration: float = Field(..., gt=0.0)
    speaker: str
    speaker_label: str
    text: str = ""
    words: List[TranscriptionWord] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class FinalTranscription(BaseModel):
    full_text: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    language: str
    words: List[TranscriptionWord] = Field(default_factory=list)


class Statistics(BaseModel):
    total_speakers: int = Field(0, ge=0)
    identified_speakers: int = Field(0, ge=0)
    unknown_speakers: int = Field(0, ge=0)
    total_segments: int = Field(0, ge=0)
    total_words: int = Field(0, ge=0)
    speech_duration: float = Field(0.0, ge=0.0)
    silence_duration: float = Field(0.0, ge=0.0)


class AnnotationResult(BaseModel):
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
        json_encoders = {datetime: lambda v: v.isoformat()}



class FileInfo(BaseModel):
    """Информация о файле"""
    filename: str
    path: str
    size_bytes: Optional[int] = None


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


# Аудио метаданные и обработка
class AudioMetadata(BaseModel):
    """Метаданные аудиофайла"""
    filename: str
    duration: float = Field(..., ge=0, description="Длительность в секундах")
    sample_rate: int = Field(..., gt=0, description="Частота дискретизации")
    channels: int = Field(..., gt=0, description="Количество каналов")
    format: str = Field(..., description="Формат файла")
    bitrate: Optional[int] = Field(None, gt=0, description="Битрейт")
    size_bytes: int = Field(..., ge=0, description="Размер файла в байтах")



class TranscriptionWord(BaseModel):
    """Слово с временными метками"""
    start: float = Field(..., ge=0, description="Начало слова в секундах")
    end: float = Field(..., gt=0, description="Конец слова в секундах")
    word: str = Field(..., description="Текст слова")
    probability: float = Field(..., ge=0.0, le=1.0, description="Вероятность распознавания")


class ContactInfo(BaseModel):
    """Информация о контакте из CardDAV"""
    uid: str = Field("", description="UID контакта в CardDAV")
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phones: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    organization: Optional[str] = None


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

class VoiceInfoUpdate(BaseModel):
    embedding: str = Field(..., description="Путь к новому файлу эмбеддинга")

class VoiceInfo(VoiceInfoBase):
    class Config:
        from_attributes = True


class ContactInfo(BaseModel):
    uid: str = Field(..., description="UID контакта в CardDAV")
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phones: List[str] = []
    emails: List[str] = []
    organization: Optional[str] = None

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

