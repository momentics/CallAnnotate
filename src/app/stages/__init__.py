# -*- coding: utf-8 -*-
"""
Этапы обработки аудио для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from .base import BaseStage, StageResult
from .preprocessing import PreprocessingStage          
from .diarization import DiarizationStage
from .transcription import TranscriptionStage
from .recognition import RecognitionStage
from .carddav import CardDAVStage

__all__ = [
    "BaseStage",
    "StageResult", 
    "PreprocessingStage",                               
    "DiarizationStage",
    "TranscriptionStage",
    "RecognitionStage",
    "CardDAVStage"
]
