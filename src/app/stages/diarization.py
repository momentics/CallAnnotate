# src/app/stages/diarization.py

# -*- coding: utf-8 -*-
"""
Этап диаризации говорящих для CallAnnotate
(поддерживает два режима: монолитный и скользящее окно)

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import torch
from typing import Dict, Any, Optional, Callable

from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from .base import BaseStage


class DiarizationStage(BaseStage):
    @property
    def stage_name(self) -> str:
        return "diarization"

    async def _initialize(self):
        cfg = self.config
        model_name = cfg.get("model", "pyannote/speaker-diarization-3.1")
        auth_token = cfg.get("use_auth_token")
        device = cfg.get("device", "cpu")

        # загрузка пайплайна
        if self.models_registry:
            cache_key = f"diarization_{model_name}_{device}"
            self.pipeline = self.models_registry.get_model(
                cache_key,
                lambda: self._load_pipeline(model_name, auth_token, device)
            )
        else:
            self.pipeline = self._load_pipeline(model_name, auth_token, device)

        # режим скользящего окна
        explicit = bool(cfg.get("window_enabled", False))
        auto = "window_size" in cfg and "hop_size" in cfg
        self.window_enabled = explicit or auto
        self.window_size = float(cfg.get("window_size", 0.0))
        self.hop_size = float(cfg.get("hop_size", 0.0))

        self.model_name = model_name
        self.device = device

    def _load_pipeline(self, model_name: str, auth_token: Optional[str], device: str) -> Pipeline:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=auth_token)
        pipeline.to(torch.device(device))
        return pipeline

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        # Монолитный режим
        if not self.window_enabled:
            if progress_callback:
                await progress_callback(10, "Начало диаризации")
            diarization: Annotation = self.pipeline(file_path)
            if progress_callback:
                await progress_callback(80, "Формирование сегментов")
            segments = []
            speaker_stats: Dict[str, Any] = {}
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

        # Оконный режим
        full = Annotation()
        try:
            duration = float(self.pipeline.get_audio_duration(file_path))
        except Exception:
            duration = 0.0

        start = 0.0
        while start < duration:
            end = min(start + self.window_size, duration)
            seg_range = Segment(start, end)
            sub_ann = self.pipeline.crop(
                {"uri": file_path, "audio": file_path},
                seg_range
            )
            for turn, _, speaker in sub_ann.itertracks(yield_label=True):
                # если returned turn уже в абсолютных координатах, не сдвигаем
                if turn.start >= start and turn.end <= end:
                    shifted = turn  # абсолютный
                else:
                    shifted = Segment(turn.start + start, turn.end + start)
                full[shifted, speaker] = True
            start += self.hop_size

        # Объединение по спикерам в единые интервалы
        intervals: Dict[str, list] = {}
        for segment, _, speaker in full.itertracks(yield_label=True):
            intervals.setdefault(speaker, []).append(segment)

        segments = []
        speaker_stats: Dict[str, Any] = {}
        for speaker, segs in intervals.items():
            min_start = min(s.start for s in segs)
            max_end = max(s.end for s in segs)
            dur = max_end - min_start
            seg = {
                "start": round(min_start, 3),
                "end": round(max_end, 3),
                "duration": round(dur, 3),
                "speaker": speaker,
                "confidence": 0.0
            }
            segments.append(seg)
            speaker_stats[speaker] = {"total_duration": dur, "segments_count": 1}

        segments.sort(key=lambda x: x["start"])
        return {
            "segments": segments,
            "speakers": list(speaker_stats.keys()),
            "speaker_stats": speaker_stats,
            "total_segments": len(segments),
            "total_speakers": len(speaker_stats)
        }

    def _get_model_info(self) -> Dict[str, Any]:
        info = {
            "stage": self.stage_name,
            "model_name": getattr(self, "model_name", None),
            "device": getattr(self, "device", None),
            "framework": "pyannote.audio"
        }
        if self.window_enabled:
            info["window_size"] = self.window_size
            info["hop_size"] = self.hop_size
        return info
