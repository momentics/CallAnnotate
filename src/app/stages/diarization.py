# src/app/stages/diarization.py
# -*- coding: utf-8 -*-
"""
DiarizationStage ― переосмысленная реализация модуля.

Ключевые изменения:
- Управление config через dataclass DiarizationCfg
- Корректная активация оконного режима
- Правильный подсчёт окон в windowed-режиме с учётом длительности
- Агрегация сегментов оконного режима путем объединения интервалов
- Корректное заполнение метаданных (в том числе оконных параметров)
- Улучшенная обработка ошибок и логирование

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from __future__ import annotations

import math
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from pyannote.core import Segment

from .base import BaseStage


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
    def __init__(self, config: Dict[str, Any], models_registry=None):
        super().__init__(config, models_registry)
        self.cfg = DiarizationCfg.from_dict(config)
        self.pipeline: Optional[Pipeline] = None

    @property
    def stage_name(self) -> str:
        return "diarization"

    async def _initialize(self) -> None:
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.cfg = DiarizationCfg.from_dict(self.config)

        if not self.cfg.use_auth_token:
            env_token = os.getenv("HF_TOKEN")
            if env_token:
                self.cfg.use_auth_token = env_token
                self.logger.info(
                    "HF_TOKEN найден в переменных окружения и будет использован для загрузки модели диаризации."
                )

        self.logger.info(f"Loading diarization model '{self.cfg.model}' (device={self.cfg.device})...")
        try:
            if self.models_registry:
                cache_key = f"pyannote_{self.cfg.model}_{self.cfg.device}"
                pipeline = self.models_registry.get_model(
                    cache_key,
                    lambda: self._load_pipeline(),
                    stage="diarization",
                    framework="pyannote.audio",
                )
            else:
                pipeline = self._load_pipeline()
            # Move pipeline to device in-place, preserving methods
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
        # Ensure any internal instantiation hooks are applied
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

            payload = self._postprocess(payload)
            self.logger.info(f"Diarization completed: {payload['total_speakers']} speakers, {payload['total_segments']} segments")
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
        steps = math.ceil(duration / hop)
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
                    #ann = self.pipeline(file_path, start=ws, end=we)  # type: ignore

                    from pyannote.audio import Audio      # локальный импорт

                    # 1. вырезаем нужный кусок в память
                    audio = Audio(sample_rate=16_000, mono=True)
                    waveform, sr = audio.crop(file_path, Segment(ws, we))

                    # 2. подаём его в pipeline как дикт с waveform
                    ann = self.pipeline(
                        {"waveform": waveform, "sample_rate": sr}
                    )


            except Exception as ex:
                self.logger.error(f"Error in diarization crop step: {ex}")
                continue
            for turn, _, speaker in ann.itertracks(yield_label=True):
                full[Segment(turn.start + ws, turn.end + ws)] = speaker
        if progress_callback:
            await progress_callback(85, "Aggregating diarization results")
        return self._annotation_to_payload(full)

    def _collapse_windowed_segments(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        collapsed = {
            "speakers": [],
            "speaker_stats": {},
            "segments": [],
            "total_speakers": 0,
            "total_segments": 0
        }
        for speaker in payload.get("speakers", []):
            segments = [seg for seg in payload["segments"] if seg["speaker"] == speaker]
            if not segments:
                continue
            start = min(seg["start"] for seg in segments)
            end = max(seg["end"] for seg in segments)
            duration = round(end - start, 3)
            confidence = round(sum(seg.get("confidence", 0.0) for seg in segments) / len(segments), 3)
            collapsed["segments"].append({
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": duration,
                "speaker": speaker,
                "confidence": confidence,
            })
            collapsed["speaker_stats"][speaker] = {
                "total_duration": duration,
                "segments_count": 1,
            }
        collapsed["speakers"] = list(collapsed["speaker_stats"].keys())
        collapsed["total_speakers"] = len(collapsed["speakers"])
        collapsed["total_segments"] = len(collapsed["segments"])
        return collapsed

    @staticmethod
    def _annotation_to_payload(annotation: Annotation) -> Dict[str, Any]:
        segments = []
        speaker_stats: Dict[str, Dict[str, float]] = {}
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            start = round(segment.start, 3)
            end = round(segment.end, 3)
            duration = round(end - start, 3)
            if duration < 0.001:
                continue
            confidence = getattr(segment, "confidence", 0.9)
            segments.append({
                "start": start,
                "end": end,
                "duration": duration,
                "speaker": speaker,
                "confidence": confidence,
            })
            stats = speaker_stats.setdefault(speaker, {"total_duration": 0.0, "segments_count": 0})
            stats["total_duration"] = round(stats["total_duration"] + duration, 3)
            stats["segments_count"] += 1
        segments.sort(key=lambda seg: (seg["start"], seg["end"]))
        return {
            "segments": segments,
            "speakers": list(speaker_stats.keys()),
            "speaker_stats": speaker_stats,
            "total_segments": len(segments),
            "total_speakers": len(speaker_stats),
        }

    def _postprocess(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not payload.get("segments"):
            self.logger.warning("No segments found")
            return payload
        if self.cfg.min_speakers or self.cfg.max_speakers:
            sorted_speakers = sorted(
                payload["speaker_stats"].items(),
                key=lambda kv: kv[1]["total_duration"],
                reverse=True,
            )
            max_allowed = self.cfg.max_speakers or len(sorted_speakers)
            keep = {spk for spk, _ in sorted_speakers[:max_allowed]}
            if self.cfg.min_speakers and len(keep) < self.cfg.min_speakers:
                extra = [spk for spk, _ in sorted_speakers[max_allowed:] if spk not in keep]
                keep.update(extra[:self.cfg.min_speakers - len(keep)])
            filtered_segments = [s for s in payload["segments"] if s["speaker"] in keep]
            filtered_stats = {spk: st for spk, st in payload["speaker_stats"].items() if spk in keep}
            payload["segments"] = filtered_segments
            payload["speaker_stats"] = filtered_stats
            payload["speakers"] = list(filtered_stats.keys())
            payload["total_speakers"] = len(filtered_stats)
            payload["total_segments"] = len(filtered_segments)
        return payload

    def _get_model_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "stage": self.stage_name,
            "model_name": self.cfg.model,
            "device": self.cfg.device,
            "framework": "pyannote.audio",
            "batch_size": self.cfg.batch_size,
            "window_enabled": self.cfg.window_enabled,
        }
        if "window_size" in self.config or self.cfg.window_enabled:
            info["window_size"] = self.cfg.window_size
        if "hop_size" in self.config or self.cfg.window_enabled:
            info["hop_size"] = self.cfg.hop_size
        if self.cfg.min_speakers is not None:
            info["min_speakers"] = self.cfg.min_speakers
        if self.cfg.max_speakers is not None:
            info["max_speakers"] = self.cfg.max_speakers
        if self.cfg.extra:
            info["extra_params"] = self.cfg.extra
        return info

    async def cleanup(self) -> None:
        if self.pipeline and hasattr(self.pipeline, 'to') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.pipeline = None
        self.logger.info("DiarizationStage cleanup completed")
