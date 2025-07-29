# -*- coding: utf-8 -*-
"""
Этап диаризации говорящих для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import torch
from typing import Dict, Any, Optional, Callable

from pyannote.audio import Pipeline
from pyannote.core import Annotation

from .base import BaseStage


class DiarizationStage(BaseStage):
    """Этап диаризации говорящих на основе pyannote.audio"""

    @property
    def stage_name(self) -> str:
        return "diarization"

    async def _initialize(self):
        """Инициализация модели диаризации"""
        model_name = self.config.get("model", "pyannote/speaker-diarization-3.1")
        auth_token = self.config.get("use_auth_token")
        device = self.config.get("device", "cpu")

        self.logger.info(f"Загрузка модели диаризации: {model_name}")

        if self.models_registry:
            cache_key = f"diarization_{model_name}_{device}"
            self.pipeline = self.models_registry.get_model(
                cache_key,
                lambda: self._load_pipeline(model_name, auth_token, device)
            )
        else:
            self.pipeline = self._load_pipeline(model_name, auth_token, device)

        self.model_name = model_name
        self.device = device

        self.logger.info("Модель диаризации загружена успешно")

    def _load_pipeline(
        self, model_name: str, auth_token: Optional[str], device: str
    ) -> Pipeline:
        """Загрузка pipeline диаризации"""
        pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=auth_token
        )
        pipeline.to(torch.device(device))
        return pipeline

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Выполнение диаризации"""

        if progress_callback:
            await progress_callback(10, "Начало диаризации")

        diarization: Annotation = self.pipeline(file_path)

        if progress_callback:
            await progress_callback(80, "Диаризация завершена, формирование сегментов")

        segments = []
        speaker_stats = {}

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            seg_data = {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "duration": round(segment.end - segment.start, 3),
                "speaker": speaker,
                "confidence": getattr(segment, "confidence", 0.0)
            }
            segments.append(seg_data)

            stats = speaker_stats.setdefault(speaker, {"total_duration": 0.0, "segments_count": 0})
            stats["total_duration"] += seg_data["duration"]
            stats["segments_count"] += 1

        segments.sort(key=lambda x: x["start"])

        if progress_callback:
            await progress_callback(100, "Диаризация завершена")

        return {
            "segments": segments,
            "speakers": list(speaker_stats.keys()),
            "speaker_stats": speaker_stats,
            "total_segments": len(segments),
            "total_speakers": len(speaker_stats)
        }

    def _get_model_info(self) -> Dict[str, Any]:
        """Информация о модели диаризации"""
        return {
            "stage": self.stage_name,
            "model_name": getattr(self, "model_name", None),
            "device": getattr(self, "device", None),
            "framework": "pyannote.audio"
        }
