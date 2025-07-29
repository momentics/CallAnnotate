# -*- coding: utf-8 -*-
"""
Этап предобработки аудио: SoX-поддержка, чанковая нормализация,
подавление шума RNNoise и DeepFilterNet 2.

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from ..rnnoise_wrapper import RNNoise
from .base import BaseStage

try:
    from df.enhance import enhance, init_df
except ImportError:  # DeepFilterNet2 не установлен
    enhance = None
    init_df = None


class PreprocessingStage(BaseStage):
    """Предобработка входного аудио (sox → RNNoise → DeepFilterNet 2)."""

    # --------------------------------------------------------------------- #
    #  PUBLIC API                                                           #
    # --------------------------------------------------------------------- #
    @property
    def stage_name(self) -> str:
        return "preprocess"

    # --------------------------- INITIALISATION -------------------------- #
    async def _initialize(self) -> None:
        """Ленивая инициализация всех внешних компонентов."""
        self.debug_mode: bool = bool(self.config.get("debug_mode", False))

        # DeepFilterNet2 --------------------------------------------------- #
        if init_df is None or enhance is None:
            raise RuntimeError(
                "Не установлена зависимость deepfilternet. "
                "Установите пакет `deepfilternet` для работы этапа preprocess."
            )
        model_name = self.config.get("model", "DeepFilterNet2")
        device = self.config.get("device", "cpu")

        if self.debug_mode:
            self.logger.info(
                "Инициализация DeepFilterNet2: model=%s, device=%s",
                model_name,
                device,
            )

        self.model, self.df_state, _ = init_df(model=model_name, device=device)

        # RNNoise ---------------------------------------------------------- #
        self.rnnoise: Optional[RNNoise] = None
        if self.config.get("rnnoise_enabled", True):
            try:
                sample_rate = int(self.config.get("rnnoise_sample_rate", 48_000))
                # сначала пробуем привычный вариант с аргументом sample_rate
                self.rnnoise = RNNoise(sample_rate=sample_rate)
            except TypeError:
                # поддержка тестовых заглушек без параметров
                try:
                    self.rnnoise = RNNoise()  # type: ignore[arg-type]
                except Exception as e:  # pragma: no cover
                    self._log_rnnoise_error(e)
                    self.rnnoise = None
            except (RuntimeError, OSError) as e:
                self._log_rnnoise_error(e)
                self.rnnoise = None

        # SoX -------------------------------------------------------------- #
        self.sox_available: bool = self._detect_sox()

        if self.debug_mode:
            self.logger.info(
                "Инициализация завершена (DeepFilterNet=%s | RNNoise=%s | SoX=%s)",
                "ON" if self.model is not None else "OFF",
                "ON" if self.rnnoise is not None else "OFF",
                "ON" if self.sox_available else "OFF",
            )

    # ----------------------------- PROCESS ------------------------------- #
    async def _process_impl(  # noqa: C901 — сложность оправдана множеством веток
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Полный цикл предобработки одного аудиофайла."""
        t0 = time.perf_counter()

        # --- основные параметры ------------------------------------------ #
        chunk_ms: int = int(float(self.config.get("chunk_duration", 2.0)) * 1_000)
        overlap_ms: int = int(float(self.config.get("overlap", 0.5)) * 1_000)
        target_rms: float = float(self.config.get("target_rms", -20.0))
        output_suffix: str = str(self.config.get("output_suffix", "_processed"))
        save_intermediate: bool = bool(self.config.get("save_intermediate", False))
        progress_step: int = int(self.config.get("progress_interval", 10))

        # --- SoX предварительная обработка ------------------------------- #
        src_file: str = (
            await self._apply_sox(file_path, task_id, target_rms)
            if self.sox_available
            else file_path
        )

        if progress_callback:
            await progress_callback(10, "Загрузка аудио")

        audio = AudioSegment.from_file(src_file)
        sr: int = audio.frame_rate
        total_ms: int = len(audio)

        # --- чанковая обработка ------------------------------------------ #
        processed_segments: List[np.ndarray] = []
        position_ms: int = 0
        idx: int = 0
        step_ms: int = chunk_ms - overlap_ms
        while position_ms < total_ms:
            end_ms = min(position_ms + chunk_ms, total_ms)
            seg = audio[position_ms:end_ms]

            # RNNoise
            if self.rnnoise is not None:
                seg = await self._apply_rnnoise(seg, idx)

            # DeepFilterNet 2
            seg_np = await self._apply_deepfilter(seg, sr, idx)
            processed_segments.append(seg_np)

            idx += 1
            position_ms += step_ms
            if (
                progress_callback
                and idx % max(progress_step, 1) == 0
                and total_ms
            ):
                pct = 10 + int(60 * position_ms / total_ms)
                await progress_callback(pct, f"preprocess chunk {idx}")

        if not processed_segments:
            raise RuntimeError("Не удалось сгенерировать ни одного чанка.")

        # --- склеивание чанков ------------------------------------------ #
        if progress_callback:
            await progress_callback(75, "Склеивание чанков")

        overlap_method: str = str(
            self.config.get("chunk_overlap_method", "linear")
        ).lower()
        merged = await self._merge_chunks(
            processed_segments, overlap_ms, sr, overlap_method
        )

        # --- финальная нормализация -------------------------------------- #
        if progress_callback:
            await progress_callback(90, "Финальная нормализация")

        merged = await self._apply_final_normalization(merged, target_rms)

        # --- сохранение --------------------------------------------------- #
        if progress_callback:
            await progress_callback(95, "Сохранение результата")

        out_path: str = await self._save(
            merged,
            original=file_path,
            suffix=output_suffix,
            sample_rate=sr,
        )

        if progress_callback:
            await progress_callback(100, "Preprocess завершён")

        return {
            "processed_path": out_path,
            "processing_time": round(time.perf_counter() - t0, 3),
        }

    # --------------------------------------------------------------------- #
    #  PRIVATE HELPERS                                                      #
    # --------------------------------------------------------------------- #
    # ---------------------------- SOX WRAPPER ---------------------------- #
    async def _apply_sox(
        self, file_path: str, task_id: str, target_rms_db: float
    ) -> str:
        """Шумоподавление + нормализация через SoX."""
        prof = Path(tempfile.gettempdir()) / f"{task_id}.prof"
        dst = Path(tempfile.gettempdir()) / f"{task_id}_sox.wav"

        try:
            subprocess.run(
                ["sox", file_path, "-n", "trim", "0", "2", "noiseprof", str(prof)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "sox",
                    file_path,
                    str(dst),
                    "noisered",
                    str(prof),
                    str(self.config.get("sox_noise_reduction", 0.3)),
                    "gain",
                    "-n",
                    str(target_rms_db),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return str(dst)
        except subprocess.CalledProcessError as e:
            self.logger.warning(
                "SoX processing failed (%s), используем оригинальный файл.", e
            )
            return file_path
        finally:
            if prof.exists():
                prof.unlink(missing_ok=True)

    # -------------------------- RNNOISE FILTER --------------------------- #
    async def _apply_rnnoise(self, seg: AudioSegment, idx: int) -> AudioSegment:
        """Оборачиваем использование RNNoise, чтобы облегчить тестовые заглушки."""
        if not self.rnnoise:
            return seg

        try:
            if hasattr(self.rnnoise, "filter"):
                return self.rnnoise.filter(seg)  # type: ignore[return-value]
            # Fallback — менее эффективный путь
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= np.iinfo(seg.array_type).max
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels).mean(axis=1)

            frames = []
            for _, frame in self.rnnoise.denoise_chunk(samples[np.newaxis, :]):  # type: ignore[attr-defined]
                frames.append(frame)
            if not frames:
                return seg

            denoised = np.concatenate(frames).astype(np.float32)
            denoised_int = (denoised * np.iinfo(seg.array_type).max).astype(
                seg.array_type
            )
            return seg._spawn(denoised_int.tobytes())
        except Exception as e:  # pragma: no cover
            self.logger.warning("RNNoise error on chunk %s: %s", idx, e)
            return seg

    # ---------------------- DEEPFILTERNET 2 ENHANCE ----------------------- #
    async def _apply_deepfilter(
        self, seg: AudioSegment, sample_rate: int, idx: int
    ) -> np.ndarray:
        """Преобразуем AudioSegment → numpy → DeepFilterNet 2 → numpy."""
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(seg.array_type).max
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)

        if self.model is None:
            return samples  # deepfilternet выключен (например, в unit-тестах)

        try:
            return enhance(self.model, self.df_state, samples)
        except Exception as e:  # pragma: no cover
            self.logger.error("DeepFilterNet error on chunk %s: %s", idx, e)
            return samples

    # --------------------------- MERGE CHUNKS ----------------------------- #
    async def _merge_chunks(
        self,
        segments: List[np.ndarray],
        overlap_ms: int,
        sample_rate: int,
        method: str,
    ) -> np.ndarray:
        """Склеиваем обработанные чанки в единую дорожку."""
        if len(segments) == 1:
            return segments[0]

        overlap_samples = int(overlap_ms * sample_rate / 1_000)

        out = segments[0].copy()
        for seg in segments[1:]:
            # пропускаем короткие сегменты, которые полностью лежат в зоне перекрытия
            if len(seg) <= overlap_samples:
                continue

            if method == "windowed" and overlap_samples:
                window = np.hanning(overlap_samples * 2)
                out[-overlap_samples:] *= window[:overlap_samples]
                seg[:overlap_samples] *= window[overlap_samples:]

            out = np.concatenate([out, seg[overlap_samples:]])

        return out

    # ------------------------- FINAL NORMALIZATION ----------------------- #
    async def _apply_final_normalization(
        self, audio: np.ndarray, target_rms_db: float
    ) -> np.ndarray:
        """Нормализация уровня громкости.

        Усиление применяется ТОЛЬКО когда текущий RMS превышает целевой,
        чтобы избежать нежелательного поднятия уровня при очевидном снижении
        сигнала (например, после RNNoise).
        """
        current_rms = float(np.sqrt(np.mean(audio**2)))
        if current_rms == 0.0:
            return audio

        target_linear = 10 ** (target_rms_db / 20)
        if current_rms > target_linear:  # только уменьшаем громкость
            gain = target_linear / current_rms
            audio = np.clip(audio * gain, -1.0, 1.0)
        return audio

    # --------------------------- SAVE RESULT ----------------------------- #
    async def _save(
        self,
        audio: np.ndarray,
        original: str,
        suffix: str,
        sample_rate: int,
    ) -> str:
        out_path = str(Path(original).with_stem(Path(original).stem + suffix))
        sf.write(out_path, audio, sample_rate, subtype="PCM_16")
        return out_path

    # --------------------------- UTILITIES ------------------------------- #
    def _detect_sox(self) -> bool:
        return bool(
            self.config.get("sox_enabled", True)
            and os.getenv("SOX_SKIP", "0") != "1"
            and shutil.which("sox")
        )

    def _log_rnnoise_error(self, exc: Exception) -> None:
        self.logger.warning(
            "RNNoise инициализация не удалась: %s. Шумоподавление будет пропущено", exc
        )

    # ------------------------- MODEL METADATA --------------------------- #
    def _get_model_info(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_name,
            "sox": self.sox_available,
            "rnnoise": self.rnnoise is not None,
            "deepfilternet": self.model is not None,
            "debug": self.debug_mode,
        }
