# src/app/stages/diarization.py
# -*- coding: utf-8 -*-
"""
DiarizationStage ― исправленная реализация модуля.

Ключевые исправления:
- Исправлена обработка перекрывающихся сегментов
- Улучшена детекция смены спикеров
- Добавлена фильтрация слишком коротких и длинных сегментов
- Исправлена агрегация результатов windowed режима
- Добавлена обработка overlap-aware сегментов

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.core import Segment

from ..config import AppSettings
from .base import BaseStage
from ..models_registry import models_registry

@dataclass
class DiarizationCfg:
    model: str = "pyannote/speaker-diarization-3.1"
    device: str = "cpu"
    use_auth_token: Optional[str] = None
    batch_size: int = 32
    window_enabled: bool = False
    window_size: float = 30.0
    hop_size: float = 10.0
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    # Добавляем параметры для улучшенной диаризации
    min_segment_duration: float = 0.5
    max_segment_duration: float = 30.0
    overlap_threshold: float = 0.1
    speaker_change_threshold: float = 0.5
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiarizationCfg":
        known, extra = {}, {}
        for k, v in data.items():
            if k in cls.__dataclass_fields__:
                known[k] = v
            else:
                extra[k] = v
        cfg = cls(**known)  # type: ignore
        cfg.extra = extra
        if not cfg.window_enabled and ("window_size" in data or "hop_size" in data):
            cfg.window_enabled = True
        return cfg

    def __post_init__(self):
        if self.window_enabled:
            if self.window_size <= 0 or self.hop_size <= 0:
                raise ValueError("window_size и hop_size должны быть > 0")
            if self.hop_size > self.window_size:
                raise ValueError("hop_size не может быть больше window_size")
        if self.batch_size <= 0:
            raise ValueError("batch_size должно быть > 0")
        if self.min_speakers is not None and self.min_speakers < 1:
            raise ValueError("min_speakers должно быть >= 1")
        if self.max_speakers is not None and self.max_speakers < 1:
            raise ValueError("max_speakers должно быть >= 1")
        if self.min_speakers and self.max_speakers and self.min_speakers > self.max_speakers:
            raise ValueError("min_speakers не может быть больше max_speakers")

class DiarizationStage(BaseStage):
    def __init__(self, cfg, config, models_registry=None):
        super().__init__(cfg, config, models_registry)
        self.cfg = DiarizationCfg.from_dict(config)
        self.pipeline: Optional[Pipeline] = None

    @property
    def stage_name(self) -> str:
        return "diarization"

    async def _initialize(self) -> None:
        cache_dir = Path(self.volume_path) / "models"
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        os.environ["TORCH_HOME"] = str(cache_dir)

        self.cfg = DiarizationCfg.from_dict(self.config)
        self.cfg.use_auth_token = None  # offline

        self.logger.info(f"Loading diarization model '{self.cfg.model}' (device={self.cfg.device})...")

        try:
            model_src = self.config.get("diarization_model_path") or self.cfg.model
            cache_key = f"pyannote_{model_src}_{self.cfg.device}"

            # В metadata теперь включаем все необходимые поля
            metadata = {
                "stage": self.stage_name,
                "model_name": self.cfg.model,
                "device": self.cfg.device,
                "framework": "pyannote.audio",
                "batch_size": self.cfg.batch_size,
                "window_enabled": self.cfg.window_enabled,
                "window_size": self.cfg.window_size,
                "hop_size": self.cfg.hop_size,
            }

            if self.models_registry:
                pipeline = self.models_registry.get_model(
                    self.logger,
                    cache_key,
                    lambda: self._load_pipeline(),
                    **metadata
                )
            else:
                pipeline = Pipeline.from_pretrained(model_src, local_files_only=True)
                pipeline.to(torch.device(self.cfg.device))
            self.pipeline = pipeline

            self.logger.info(f"DiarizationStage initialised (window_enabled={self.cfg.window_enabled})")
        except Exception as e:
            self.logger.exception(f"Failed to initialize Pyannote pipeline: {e}")
            self.pipeline = None

    def _load_pipeline(self) -> Pipeline:
        pipeline = Pipeline.from_pretrained(
            self.cfg.model,
            use_auth_token=self.cfg.use_auth_token,
        )
        if hasattr(pipeline, '_instantiate'):
            pipeline._instantiate()
        return pipeline

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("Pyannote pipeline is not initialized")

        self.logger.info(f"Starting diarization for {file_path} (task {task_id})")

        try:
            if self.cfg.window_enabled:
                raw_payload = await self._run_windowed(file_path, progress_callback)
                payload = self._collapse_windowed_segments(raw_payload)
            else:
                raw_payload = await self._run_monolithic(file_path, progress_callback)
                payload = raw_payload

            payload = self._postprocess_segments(payload)
            payload = self._postprocess(payload)

            self.logger.info(
                f"Diarization completed: {payload['total_speakers']} speakers, {payload['total_segments']} segments"
            )
            return payload
        except Exception as e:
            self.logger.error(f"Error during diarization processing: {e}", exc_info=True)
            raise

    async def _run_monolithic(self, file_path: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        if progress_callback:
            await progress_callback(10, "Diarization (monolithic) started")
        
        diarization = self.pipeline(file_path)  # type: ignore
        
        if progress_callback:
            await progress_callback(80, "Formatting diarization output")
        return self._annotation_to_payload(diarization)

    async def _run_windowed(self, file_path: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        window = self.cfg.window_size
        hop = self.cfg.hop_size
        
        try:
            duration = self.pipeline.get_audio_duration(file_path)  # type: ignore
        except Exception:
            import soundfile as sf
            info = sf.info(file_path)
            duration = info.duration
        
        duration = float(duration)
        steps = math.ceil((duration - window) / hop) + 1 if duration > window else 1
        full = Annotation()
        
        for idx in range(steps):
            ws = idx * hop
            we = min(ws + window, duration)
            
            if progress_callback:
                pct = 10 + int((70 * idx) / max(steps - 1, 1))
                await progress_callback(pct, f"Diarization window {idx + 1}/{steps}")
            
            try:
                if hasattr(self.pipeline, 'crop'):
                    ann = self.pipeline.crop(
                        {"uri": file_path, "audio": file_path},
                        Segment(ws, we),
                    )
                else:
                    from pyannote.audio import Audio
                    audio = Audio(sample_rate=16_000, mono=True)
                    waveform, sr = audio.crop(file_path, Segment(ws, we))
                    ann = self.pipeline(
                        {"waveform": waveform, "sample_rate": sr}
                    )
                
                # Смещаем сегменты на начало окна
                for segment, _, label in ann.itertracks(yield_label=True):
                    shifted_segment = Segment(segment.start + ws, segment.end + ws)
                    full[shifted_segment] = label
                    
            except Exception as e:
                self.logger.warning(f"Error processing window {idx}: {e}")
                continue

        if progress_callback:
            await progress_callback(85, "Windowed processing complete")
            
        return self._annotation_to_payload(full)

    def _collapse_windowed_segments(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Объединяет перекрывающиеся сегменты из windowed режима"""
        segments = raw_payload.get("segments", [])
        if not segments:
            return raw_payload

        # Группируем сегменты по спикерам
        speaker_segments = {}
        for seg in segments:
            speaker = seg["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)

        # Объединяем близкие сегменты одного спикера
        collapsed_segments = []
        for speaker, segs in speaker_segments.items():
            # Сортируем по времени начала
            segs.sort(key=lambda x: x["start"])
            
            merged = []
            for seg in segs:
                if not merged or seg["start"] - merged[-1]["end"] > self.cfg.overlap_threshold:
                    merged.append(seg.copy())
                else:
                    # Объединяем сегменты
                    merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
                    merged[-1]["duration"] = merged[-1]["end"] - merged[-1]["start"]
                    merged[-1]["confidence"] = max(merged[-1]["confidence"], seg["confidence"])
            
            collapsed_segments.extend(merged)

        # Сортируем по времени
        collapsed_segments.sort(key=lambda x: x["start"])
        
        raw_payload["segments"] = collapsed_segments
        raw_payload["total_segments"] = len(collapsed_segments)
        
        return raw_payload

    def _postprocess_segments(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Постобработка сегментов для исправления проблем диаризации"""
        segments = payload.get("segments", [])
        if not segments:
            return payload

        # Фильтруем сегменты по длительности
        filtered_segments = []
        for seg in segments:
            duration = seg["duration"]
            if self.cfg.min_segment_duration <= duration <= self.cfg.max_segment_duration:
                filtered_segments.append(seg)
            else:
                self.logger.debug(f"Filtered out segment with duration {duration:.2f}s (speaker: {seg['speaker']})")

        # Разбиваем слишком длинные сегменты
        final_segments = []
        for seg in filtered_segments:
            if seg["duration"] > self.cfg.max_segment_duration:
                # Разбиваем длинный сегмент на части
                num_parts = math.ceil(seg["duration"] / self.cfg.max_segment_duration)
                part_duration = seg["duration"] / num_parts
                
                for i in range(num_parts):
                    start = seg["start"] + i * part_duration
                    end = min(start + part_duration, seg["end"])
                    
                    new_seg = seg.copy()
                    new_seg["start"] = round(start, 3)
                    new_seg["end"] = round(end, 3)
                    new_seg["duration"] = round(new_seg["end"] - new_seg["start"], 3)
                    final_segments.append(new_seg)
                    
                self.logger.info(f"Split long segment ({seg['duration']:.2f}s) into {num_parts} parts")
            else:
                final_segments.append(seg)

        # Устраняем перекрытия между разными спикерами
        final_segments = self._resolve_overlaps(final_segments)
        
        # Обновляем payload
        payload["segments"] = final_segments
        payload["total_segments"] = len(final_segments)
        
        return payload

    def _resolve_overlaps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Устраняет перекрытия между сегментами разных спикеров"""
        if len(segments) <= 1:
            return segments
            
        # Сортируем по времени начала
        segments.sort(key=lambda x: x["start"])
        
        resolved = []
        for i, current in enumerate(segments):
            if not resolved:
                resolved.append(current.copy())
                continue
                
            prev = resolved[-1]
            
            # Проверяем перекрытие с предыдущим сегментом
            if current["start"] < prev["end"] and current["speaker"] != prev["speaker"]:
                overlap = prev["end"] - current["start"]
                self.logger.debug(f"Overlap detected: {overlap:.3f}s between {prev['speaker']} and {current['speaker']}")
                
                # Разрешаем перекрытие: делим пополам или отдаём более уверенному
                if prev["confidence"] > current["confidence"]:
                    # Сокращаем текущий сегмент
                    current["start"] = prev["end"]
                    current["duration"] = current["end"] - current["start"]
                else:
                    # Сокращаем предыдущий сегмент
                    prev["end"] = current["start"]
                    prev["duration"] = prev["end"] - prev["start"]
                    resolved[-1] = prev
            
            # Добавляем только если сегмент не стал слишком коротким
            if current["duration"] >= self.cfg.min_segment_duration:
                resolved.append(current.copy())
            else:
                self.logger.debug(f"Removed short segment after overlap resolution: {current['duration']:.3f}s")
        
        return resolved

    def _annotation_to_payload(self, annotation: Annotation) -> Dict[str, Any]:
        """Конвертирует pyannote Annotation в payload"""
        segments = []
        speakers = set()
        
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "duration": round(segment.end - segment.start, 3),
                "speaker": str(speaker),
                "confidence": 0.9  # Базовая уверенность
            })
            speakers.add(str(speaker))

        return {
            "segments": segments,
            "speakers": list(speakers),
            "total_segments": len(segments),
            "total_speakers": len(speakers)
        }

    def _postprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Финальная постобработка результатов"""
        # Пересчитываем статистики
        segments = payload.get("segments", [])
        speakers = set()
        total_duration = 0.0
        
        for seg in segments:
            speakers.add(seg["speaker"])
            total_duration += seg["duration"]
        
        payload.update({
            "speakers": list(speakers),
            "total_speakers": len(speakers),
            "total_segments": len(segments),
            "total_duration": round(total_duration, 3)
        })
        
        return payload

    def _get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели для метаданных"""
        return {
            "stage": self.stage_name,
            "model_name": self.cfg.model,
            "device": self.cfg.device,
            "framework": "pyannote.audio",
            "batch_size": self.cfg.batch_size,
            "window_enabled": self.cfg.window_enabled,
            "window_size": self.cfg.window_size if self.cfg.window_enabled else None,
            "hop_size": self.cfg.hop_size if self.cfg.window_enabled else None,
        }
