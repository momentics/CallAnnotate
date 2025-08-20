# src/app/stages/preprocessing.py
# -*- coding: utf-8 -*-
"""
Рефакторинг этапа предобработки аудио для CallAnnotate.

Ключевые улучшения:
- Изоляция сложных зависимостей (SoX, RNNoise, DFN) во вспомогательных модулях.
- Потоковая обработка длинных файлов: чтение -> чанки -> фильтры -> склейка.
- Детерминированная нормализация RMS: только «вниз», без усиления шума.
- Устойчивое поведение: при любых ошибках возвращаем входной результат.
- Подробные комментарии и понятные лог-сообщения.

Совместимость:
- Подписывается под BaseStage и возвращает {"processed_path", "original_path", "processing_time"}.
- Учитывает конфигурацию из AppSettings.PreprocessingConfig.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional
from pathlib import Path
import time
import logging

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from .base import BaseStage
from ..config import AppSettings
from ..rnnoise_wrapper import FRAME_SIZE as RN_FRAME_SIZE  # используем для тестов/валидаций

# Локальные утилиты аудио-подсистемы
from .audio.io import (
    load_audio_segment,
    ensure_channels_and_rate,
    to_mono_float32,
    from_mono_float32,
)
from .audio.sox_tools import apply_sox_noise_reduce, is_sox_available
from .audio.denoise_rnnoise import RNNoiseWrapper
from .audio.denoise_dfn import DeepFilter
from .audio.chunking import iter_overlapping_chunks
from .audio.mix import merge_chunks_linear, merge_chunks_windowed
from .audio.level import apply_rms_ceiling


class PreprocessingStage(BaseStage):
    """
    Пайплайн: (опционально) SoX -> RNNoise -> DeepFilterNet -> Merge -> RMS ceiling -> Save.
    Каждый шаг устойчив к ошибкам и может быть отключён конфигурацией.
    """

    @property
    def stage_name(self) -> str:
        return "preprocess"

    async def _initialize(self) -> None:
        """
        Инициализация подэтапов и состояний. Делается один раз на весь жизненный цикл экземпляра.
        """
        cfg = self.config
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        # Флаги и параметры
        self.debug_mode: bool = bool(cfg.get("debug_mode", False))

        # SoX
        self.sox_enabled: bool = bool(cfg.get("sox_enabled", False))
        self.sox_noise_profile_duration: float = float(cfg.get("sox_noise_profile_duration", 2.0))
        self.sox_noise_reduction: float = float(cfg.get("sox_noise_reduction", 0.3))
        self.sox_gain_normalization: bool = bool(cfg.get("sox_gain_normalization", True))
        self.sox_available: bool = is_sox_available() if self.sox_enabled else False

        # RNNoise
        self.rnnoise_enabled: bool = bool(cfg.get("rnnoise_enabled", True))
        self._rn = RNNoiseWrapper(enabled=self.rnnoise_enabled, allow_passthrough=True)

        # DeepFilterNet
        self.deepfilter_enabled: bool = bool(cfg.get("deepfilter_enabled", False))
        df_target_sr: int = int(cfg.get("deepfilter_sample_rate", 48_000))
        model_name: str = str(cfg.get("model", "DeepFilterNet2"))
        device: str = str(cfg.get("device", "cpu"))
        self._df = DeepFilter(enable=self.deepfilter_enabled, model_name=model_name, device=device, target_sr=df_target_sr)

        # IO/выход
        self.output_suffix: str = str(cfg.get("output_suffix", "_processed"))
        self.audio_format: str = str(cfg.get("audio_format", "wav")).lower()
        self.channels_mode: str = str(cfg.get("channels", "mono")).lower()
        self.sample_rate_target: Optional[int] = cfg.get("sample_rate_target", 16000)
        self.sample_rate_target = int(self.sample_rate_target) if self.sample_rate_target not in (None, "null") else None

        # Чанкование
        self.chunk_duration_sec: float = float(cfg.get("chunk_duration", 10.0))
        self.overlap_sec: float = float(cfg.get("overlap", 0.5))
        self.chunk_overlap_method: str = str(cfg.get("chunk_overlap_method", "linear")).lower()
        self.progress_step: int = int(cfg.get("progress_interval", 10))

        # Уровень
        self.target_rms_db: float = float(cfg.get("target_rms", -20.0))

        # Лимиты/служебное (не жёстко применяем в коде, но используем в дальнейших улучшениях)
        self.processing_threads: int = int(cfg.get("processing_threads", 1))
        self.memory_limit_mb: int = int(cfg.get("memory_limit_mb", 1024))
        self.temp_cleanup: bool = bool(cfg.get("temp_cleanup", True))
        self.preserve_original: bool = bool(cfg.get("preserve_original", True))
        self.save_intermediate: bool = bool(cfg.get("save_intermediate", False))

        self.logger.info(
            "Preprocessing initialized: sox=%s rnnoise=%s dfn=%s out=%s/%s, chunk=%.1fs overlap=%.3fs",
            self.sox_available, self._rn.enabled, self._df.is_ready(), self.channels_mode, self.audio_format,
            self.chunk_duration_sec, self.overlap_sec
        )

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        """
        Основная обработка:
        1) (опционально) SoX noisered на исходном файле -> временный WAV (если возможно).
        2) Чтение через pydub -> нормализация каналов/частоты.
        3) Чанки: RNNoise -> DeepFilterNet.
        4) Склейка chunk-результатов.
        5) Финальный RMS ceiling.
        6) Сохранение в /volume/processing с суффиксом.

        Возврат: processed_path/original_path/processing_time
        """
        started = time.perf_counter()
        async def progress(p: int, msg: str):
            if progress_callback:
                await progress_callback(p, msg)
            self.logger.debug("preprocess: %3d%% %s", p, msg)

        await progress(3, "preprocess: старт")

        # 1) SoX (опционально, безопасный возврат исходника при ошибке)
        src_for_pipeline = file_path
        temp_dir = str(Path(self.volume_path) / "temp")

        if self.sox_available:
            await progress(6, "preprocess: sox профилирование")
            src_for_pipeline = apply_sox_noise_reduce(
                src_path=file_path,
                dst_dir=temp_dir,
                task_id=task_id,
                profile_duration=self.sox_noise_profile_duration,
                reduction=self.sox_noise_reduction,
                rms_gain_normalization=self.sox_gain_normalization,
                target_rms_db=self.target_rms_db
            )

        # 2) Чтение, выравнивание каналов/частоты
        await progress(10, "preprocess: загрузка аудио")
        try:
            seg = load_audio_segment(src_for_pipeline)
        except Exception as e:
            # В крайнем случае отдаём исходник без изменения
            self.logger.warning("preprocess: ошибка чтения (%s), пропуск этапа", e)
            return {
                "processed_path": file_path,
                "original_path": file_path,
                "processing_time": round(time.perf_counter() - started, 3)
            }

        seg = ensure_channels_and_rate(seg, self.channels_mode, self.sample_rate_target)
        sr = seg.frame_rate
        total_ms = len(seg)

        # 3) Чанкование и обработка
        chunk_ms = int(max(1.0, self.chunk_duration_sec * 1000.0))
        overlap_ms = int(max(0.0, self.overlap_sec * 1000.0))
        overlap_samples = int(round(overlap_ms * sr / 1000.0))
        chunks_f32: List[np.ndarray] = []
        processed_count = 0

        await progress(15, "preprocess: обработка чанков")
        for i, (s_ms, e_ms, sub) in enumerate(iter_overlapping_chunks(seg, chunk_ms, overlap_ms), start=1):
            # RNNoise — возвращает сегмент (возможно, с rate=48k); доведём потом
            try:
                rn = self._rn.filter_segment(sub) if self._rn.enabled else sub
                # Нормализуем rate/каналы к текущим (pydub копирует метаданные)
                if rn.frame_rate != sr:
                    rn = rn.set_frame_rate(sr)
                if rn.channels != seg.channels:
                    rn = rn.set_channels(seg.channels)
            except Exception:
                rn = sub  # fallback

            # DFN — работает в float32 моно
            f32 = to_mono_float32(rn)
            try:
                if self._df.is_ready():
                    f32 = self._df.process(f32, sr)
            except Exception:
                # fallback на raw после RNNoise
                pass
            chunks_f32.append(f32.astype(np.float32, copy=False))

            processed_count += 1
            if self.progress_step and (processed_count % max(1, self.progress_step) == 0):
                pct = 15 + int(60.0 * (e_ms / max(1, total_ms)))
                await progress(min(75, pct), f"preprocess: чанки {processed_count}")

        if not chunks_f32:
            # Нечего обрабатывать — отдаём оригинал
            self.logger.warning("preprocess: нет чанков, пропуск")
            return {
                "processed_path": file_path,
                "original_path": file_path,
                "processing_time": round(time.perf_counter() - started, 3)
            }

        # 4) Склейка
        await progress(80, "preprocess: склейка")
        if self.chunk_overlap_method == "windowed":
            merged = merge_chunks_windowed(chunks_f32, overlap_samples=overlap_samples)
        else:
            merged = merge_chunks_linear(chunks_f32, overlap_samples=overlap_samples)

        # 5) Финальная RMS нормализация вниз
        await progress(88, "preprocess: RMS ceiling")
        merged = apply_rms_ceiling(merged, self.target_rms_db)

        # 6) Сохранение PCM_16 в processing/
        await progress(95, "preprocess: сохранение")
        processing_dir = Path(self.volume_path) / "processing"
        processing_dir.mkdir(parents=True, exist_ok=True)

        in_path = Path(file_path)
        out_name = f"{in_path.stem}{self.output_suffix}.{self.audio_format}"
        out_path = processing_dir / out_name

        try:
            sf.write(str(out_path), np.clip(merged, -1.0, 1.0), sr, subtype="PCM_16")
        except Exception as e:
            # На случай проблем записи — откат к исходнику
            self.logger.error("preprocess: ошибка записи файла (%s), отдаём оригинал", e)
            return {
                "processed_path": file_path,
                "original_path": file_path,
                "processing_time": round(time.perf_counter() - started, 3)
            }

        await progress(100, "preprocess: завершено")
        return {
            "processed_path": str(out_path),
            "original_path": file_path,
            "processing_time": round(time.perf_counter() - started, 3)
        }
