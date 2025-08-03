# src/app/stages/preprocessing.py

# -*- coding: utf-8 -*-
"""
Этап предобработки аудио: SoX-поддержка, чанковая нормализация,
RNNoise и DeepFilterNet.

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from scipy.signal import resample_poly

from ..rnnoise_wrapper import RNNoise
from .base import BaseStage

# Поддержка разных расположений функций init_df / enhance в DeepFilterNet
init_df: Optional[Callable] = None
enhance: Optional[Callable] = None
for _path in (
    "df.deepfilternet2",
    "df.deepfilternet3",
    "df.deepfilternet",
    "df.enhance",
):
    try:
        mod = __import__(_path, fromlist=["init_df", "enhance"])
        init_df = getattr(mod, "init_df", None)
        enhance = getattr(mod, "enhance", None)
        if callable(init_df) and callable(enhance):
            break
    except ModuleNotFoundError:
        continue


class PreprocessingStage(BaseStage):
    """SoX → RNNoise → DeepFilterNet → нормализация"""

    @property
    def stage_name(self) -> str:
        return "preprocess"

    async def _initialize(self) -> None:
        self.logger.info("=== PreprocessingStage: старт инициализации ===")
        self.debug_mode: bool = bool(self.config.get("debug_mode", False))
        self.deepfilter_enabled = bool(self.config.get("deepfilter_enabled", False))
        self.deepfilter_sample_rate = int(self.config.get("deepfilter_sample_rate", 48000))

        self.rnnoise_sample_rate = int(self.config.get("rnnoise_sample_rate", 48000))

        if init_df is None or enhance is None:
            raise RuntimeError(
                "Не установлена зависимость deepfilternet. "
                "Установите пакет `deepfilternet` для работы этапа preprocess."
            )
        
        if self.deepfilter_enabled:

            model_name = self.config.get("model", "DeepFilterNet2")
            device = self.config.get("device", "cpu")
            

            # Универсальный вызов init_df: учитываем mocks с любым signature
            try:
                sig = inspect.signature(init_df)
                # если функция не принимает позиционные обязательные args (len==0)
                # или поддерживает **kwargs, вызываем без аргументов
                if len(sig.parameters) == 0 or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    self.model, self.df_state, _ = init_df()  # type: ignore[arg-type]
                elif "model" in sig.parameters and "device" in sig.parameters:
                    self.model, self.df_state, _ = init_df(model=model_name, device=device)  # type: ignore[arg-type]
                else:
                    self.model, self.df_state, _ = init_df(model_name, device)  # type: ignore[arg-type]
                
                self.logger.info(f"DeepFilterNet инициализировано с sample_rate={self.deepfilter_sample_rate}, model={model_name}, device={device}")
            except Exception as e: 
                self._log_deepfilternet_error(e)
                self.deepfilter_enabled = False
                raise
        else:
            self.model = None

        # RNNoise
        self.rnnoise: Optional[RNNoise] = None
        if self.config.get("rnnoise_enabled", True):
            try:
                sig = inspect.signature(RNNoise.__init__)
                if "sample_rate" in sig.parameters:
                    self.rnnoise = RNNoise(sample_rate=self.rnnoise_sample_rate)
                else:
                    self.rnnoise = RNNoise()  # type: ignore[arg-type]
                    if hasattr(self.rnnoise, "sample_rate"):
                        self.rnnoise.sample_rate = self.rnnoise_sample_rate
                if self.debug_mode:
                    self.logger.info(f"RNNoise инициализировано с sample_rate={self.rnnoise.sample_rate}")
            except Exception as e:
                self._log_rnnoise_error(e)

        # SoX
        self.sox_available = self._detect_sox()

        self.logger.info(self._get_model_info())

    # берет любой файл из поддерживаемых, производит шумодав
    # и выдает файл в формате PCM_16, добавляя к имени файла постфикс
    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        
        self.logger.info(
            "=== PreprocessingStage: начало обработки файла %s (task_id=%s) ===",
            file_path,
            task_id,
        )
        t0 = time.perf_counter()

        chunk_ms = int(self.config.get("chunk_duration", 2.0) * 1000)
        overlap_ms = int(self.config.get("overlap", 0.5) * 1000)
        target_rms = float(self.config.get("target_rms", -20.0))
        output_suffix = str(self.config.get("output_suffix", "_processed"))
        progress_step = int(self.config.get("progress_interval", 10))

        # SoX обработанный файл или оригинальный файл
        src = (
            await self._apply_sox(file_path, task_id, target_rms)
            if self.sox_available
            else file_path
        )

        if progress_callback:
            await progress_callback(10, "Загрузка аудио")

        audio = AudioSegment.from_file(src)
        sr = audio.frame_rate
        total_ms = len(audio)

        segments: List[np.ndarray] = []
        pos = 0
        idx = 0
        step = chunk_ms - overlap_ms
        while pos < total_ms:
            end = min(pos + chunk_ms, total_ms)
            seg = audio[pos:end]

            if self.rnnoise:
                seg = await self._apply_rnnoise(seg, idx)


            if self.deepfilter_enabled:
                # apply deepfilter, then convert to numpy array
                filtered = await self._apply_deepfilter(seg, sr, idx)
                # `filtered` may be AudioSegment or numpy array
                if hasattr(filtered, "get_array_of_samples"):
                    # convert AudioSegment to float32 numpy array in [-1,1]
                    arr = np.array(filtered.get_array_of_samples(), dtype=np.float32)
                    arr /= np.iinfo(filtered.array_type).max
                    segments.append(arr)
                else:
                    segments.append(filtered)
            else:
                # Это является бест практикой, преобразовывать в [-1:1] float32
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                samples /= np.iinfo(seg.array_type).max
                segments.append(samples)

            idx += 1
            pos += step

            if progress_callback and idx % max(progress_step, 1) == 0:
                pct = 10 + int(60 * pos / total_ms)
                await progress_callback(pct, f"preprocess chunk {idx}")

        if not segments:
            raise RuntimeError("Не удалось сгенерировать ни одного чанка.")

        if progress_callback:
            await progress_callback(75, "Склеивание чанков")

        merged = await self._merge_chunks(
            segments,
            overlap_ms,
            sr,
            str(self.config.get("chunk_overlap_method", "linear")).lower(),
        )

        if progress_callback:
            await progress_callback(90, "Финальная нормализация")

        merged = await self._apply_final_normalization(merged, target_rms)

        if progress_callback:
            await progress_callback(95, "Сохранение результата")

        self.logger.info("=== PreprocessingStage: сохранение результата ===")

        out_path = await self._save(
            merged,
            original=file_path,
            suffix=output_suffix,
            sample_rate=sr,
        )

        dt = time.perf_counter() - t0
        
        self.logger.info(
            "=== PreprocessingStage: обработка завершена за %.3f с, результат: %s ===",
            dt,
            out_path,
        )

        if progress_callback:
            await progress_callback(100, "Preprocess завершён")

        return {"processed_path": out_path, "processing_time": round(dt, 3)}

    # --------------------------------------------------------------------- #
    # DEEPFILTERNET                                                         #
    # --------------------------------------------------------------------- #
    async def _apply_deepfilter(self, seg: AudioSegment, sr: int, idx: int) -> np.ndarray:

        # это является лучшей практикой, преобразовывать в [-1:1] float32
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= np.iinfo(seg.array_type).max

        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)

        if self.model is None:
            return samples

        sr = seg.frame_rate

        try:
            if sr <= self.deepfilter_sample_rate:
                samples = self.upsample_audio(samples, sr, self.deepfilter_sample_rate)
            else:
                samples = self.downsample_audio(samples, sr, self.deepfilter_sample_rate)

            tensor = torch.from_numpy(samples).unsqueeze(0).float()
            # shape becomes (1, N) – (каналы, сэмплы)
            out = enhance(self.model, self.df_state, tensor)

            if isinstance(out, torch.Tensor):
                # out now has shape (1, N); squeeze back
                denoised = out.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                denoised = np.asarray(out, dtype=np.float32)

            if sr <= self.deepfilter_sample_rate:
                denoised = self.downsample_audio(denoised, self.deepfilter_sample_rate, sr)
            else:
                denoised = self.upsample_audio(denoised, self.deepfilter_sample_rate, sr)

            # вернем деноизированный в оригинальном формате, с нормализацией
            denoised_int = (denoised * np.iinfo(seg.array_type).max).astype(seg.array_type)
            return seg._spawn(denoised_int.tobytes())
        
        except Exception as e:
            self.logger.error("DeepFilterNet error on chunk %s: %s", idx, e, exc_info=True)
            return samples

    async def _apply_sox(self, file_path: str, task_id: str, target_rms_db: float) -> str:
        """Шумоподавление + нормализация через SoX."""

        # профайл шума
        prof = Path(tempfile.gettempdir()) / f"{task_id}.prof"
        # результирующий файл после обработки
        dst = Path(tempfile.gettempdir()) / f"{task_id}_sox.wav"

        try:
            # получаем профиль шума
            subprocess.run(
                ["sox", file_path, "-n", "trim", "0", "2", "noiseprof", str(prof)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # обрабаываем файл профилем и нормализуем RMS
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
            # вернем путь к файлу после шумодава
            return str(dst)
        
        except subprocess.CalledProcessError as e:
            self.logger.warning("SoX processing failed (%s), используем оригинальный файл.", e)
            # в случае ошибки возвращаем оригинальный путь
            return file_path
        finally:
            # закрываем файл профиля
            prof.unlink(missing_ok=True)

    async def _apply_rnnoise(self, seg: AudioSegment, idx: int) -> AudioSegment:
        """Применение RNNoise к чанку."""
        if not self.rnnoise:
            return seg

        try:
            # Это является бест практикой, преобразовывать в [-1:1] float32
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= np.iinfo(seg.array_type).max

            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels).mean(axis=1)

            sr = seg.frame_rate

            self.logger.info(f"RNNoise: orig_sr={sr}, samples_shape={samples.shape}")

            if sr <= self.rnnoise_sample_rate:
                samples = self.upsample_audio(samples, sr, self.rnnoise_sample_rate)
                self.logger.info(f"RNNoise: upsampled to 48kHz, new_length={len(samples)}")
            else:
                samples = self.downsample_audio(samples, sr, self.rnnoise_sample_rate)
                self.logger.info(f"RNNoise: downsampled to 48kHz, new_length={len(samples)}")



            if hasattr(self.rnnoise, "filter"):
                # --- NEW CODE ---------------------------------------------------------
                filtered = self.rnnoise.filter(seg)

                # 1.  AudioSegment → float32 NumPy array in range [-1, 1]
                if isinstance(filtered, AudioSegment):
                    filt_samples = np.array(filtered.get_array_of_samples(),
                                            dtype=np.float32)
                    filt_samples /= np.iinfo(filtered.array_type).max
                    if filtered.channels > 1:
                        filt_samples = (
                            filt_samples.reshape(-1, filtered.channels).mean(axis=1)
                        )
                    filt_sr = filtered.frame_rate
                else:                           # already ndarray
                    filt_samples = filtered
                    filt_sr = self.rnnoise_sample_rate      # rnnoise works at 48 kHz
                # ---------------------------------------------------------------------

                # 2.  Resample to original sample-rate
                if filt_sr > sr:
                    resampled = self.downsample_audio(filt_samples, filt_sr, sr)
                elif filt_sr < sr:
                    resampled = self.upsample_audio(filt_samples, filt_sr, sr)
                else:
                    resampled = filt_samples

                # 3.  NumPy array → AudioSegment that matches the input format
                resampled_int = (
                    resampled * np.iinfo(seg.array_type).max
                ).astype(seg.array_type)
                return seg._spawn(resampled_int.tobytes())

            frames = []
            # пропускаем speech probability и возвращаем только фреймы
            for _, frame in self.rnnoise.denoise_chunk(samples[np.newaxis, :]):  # type: ignore[attr-defined]
                if sr <= self.rnnoise_sample_rate:
                    frames.append(self.downsample_audio(frame, self.rnnoise_sample_rate, sr))
                else:
                    frames.append(self.upsample_audio(frame, self.rnnoise_sample_rate, sr))

            if not frames:
                return seg

            # уже ресемплированы в sr
            denoised = np.concatenate(frames).astype(np.float32)
            denoised_int = (denoised * np.iinfo(seg.array_type).max).astype(seg.array_type)

            return seg._spawn(denoised_int.tobytes())
        
        except Exception as e:
            self.logger.warning("RNNoise error on chunk %s: %s", idx, e)
            return seg

    async def _merge_chunks(
        self,
        segments: List[np.ndarray],
        overlap_ms: int,
        sample_rate: int,
        method: str,
    ) -> np.ndarray:
        """Merge processed audio chunks.

        Args:
            segments: List of 1-D numpy arrays representing audio chunks.
            overlap_ms: Overlap duration between chunks in milliseconds.
            sample_rate: Sample rate of the audio in Hz.
            method: "linear" for simple overlap-and-concatenate,
                    "windowed" to apply a Hanning window in the overlap region.

        Returns:
            A single 1-D numpy array representing the merged audio.
        """
        # If only one segment, return it unchanged
        if len(segments) == 1:
            return segments[0]

        # Calculate number of samples corresponding to overlap_ms
        overlap_samples = int(overlap_ms * sample_rate / 1000)

        # Start with the first chunk unchanged
        output = segments[0].copy()

        # Precompute Hanning window if needed
        if method == "windowed" and overlap_samples > 0:
            # Create a Hanning window twice as long, split into two halves
            hann = np.hanning(overlap_samples * 2)
            win_a = hann[:overlap_samples]
            win_b = hann[overlap_samples:]

        # Iterate through subsequent chunks
        for chunk in segments[1:]:
            # If chunk shorter than overlap, skip blending
            if len(chunk) <= overlap_samples:
                continue

            if method == "windowed" and overlap_samples > 0:
                # Apply windowed overlap-add:
                # taper end of output and start of chunk
                output_end = output[-overlap_samples:].copy()
                chunk_start = chunk[:overlap_samples].copy()
                output[-overlap_samples:] = output_end * win_a
                chunk[:overlap_samples] = chunk_start * win_b
            # Concatenate, skipping overlap_samples from the start of chunk
            output = np.concatenate([output, chunk[overlap_samples:]])

        return output


    async def _merge_chunks1(
        self,
        segments: List[np.ndarray],
        overlap_ms: int,
        sample_rate: int,
        method: str,
    ) -> np.ndarray:
        """Склеивание обработанных чанков."""
        # если один сегмент, то нечего склеивать
        if len(segments) == 1:
            return segments[0]

        # считаем сколько семплов оверлапится
        overlap_samples = int(overlap_ms * sample_rate / 1000)
        # первый сегмент не оверлапится
        out = segments[0].copy()

        # далее - склеивание. Либо линейное, либо оконное
        for seg in segments[1:]:
            # длина сегмента меньше длины оверлапа
            if len(seg) <= overlap_samples:
                continue
            # накладывание с наложением по ханингу, иначе пропускаем ханинг
            if method == "windowed" and overlap_samples:
                window = np.hanning(overlap_samples * 2)
                out[-overlap_samples:] *= window[:overlap_samples]
                seg[:overlap_samples] *= window[overlap_samples:]
            # склеиваем подготовленные куски
            out = np.concatenate([out, seg[overlap_samples:]])
        # по завершению возвращаем
        return out

    async def _apply_final_normalization(self, audio: np.ndarray, target_rms_db: float) -> np.ndarray:
        """Финальная нормализация уровня громкости."""
        current_rms = float(np.sqrt(np.mean(audio**2)))

        if current_rms == 0.0:
            # тишина
            return audio

        # иначе выравниваем RMS
        target_linear = 10 ** (target_rms_db / 20)
        if current_rms > target_linear:
            gain = target_linear / current_rms
            audio = np.clip(audio * gain, -1.0, 1.0)

        # возвращаем выровненный результат
        return audio

    async def _save(self, audio: np.ndarray, 
                    original: str, 
                    suffix: str, 
                    sample_rate: int) -> str:
        """Сохранение результата на диск."""
        out_path = str(Path(original).with_stem(Path(original).stem + suffix))
        # преобразуем итоговый формат файла после шумоподавления
        sf.write(out_path, audio, sample_rate, subtype="PCM_16")
        # возвращаем путь до него
        return out_path

    def _detect_sox(self) -> bool:
        """Проверка доступности SoX."""
        return bool(
            self.config.get("sox_enabled", False)
            and shutil.which("sox")
        )

    def _log_rnnoise_error(self, exc: Exception) -> None:
        """Логирование ошибок при инициализации RNNoise."""
        self.logger.warning(
            "RNNoise инициализация не удалась: %s. Шумоподавление RNNoise будет пропущено", exc
        )

    def _log_deepfilternet_error(self, exc: Exception) -> None:
        """Логирование ошибок при инициализации deepfilternet."""
        self.logger.warning(
            "DeepFilterNet инициализация не удалась: %s. Шумоподавление DeepFilterNet будет пропущено", exc
        )

    def _get_model_info(self) -> Dict[str, Any]:
        """Метаданные конфигурации PreprocessingStage."""
        return {
            "stage": self.stage_name,
            "sox": self.sox_available,
            "rnnoise": self.rnnoise is not None,
            "deepfilternet": self.model is not None,
            "debug": self.debug_mode,
        }


    def upsample_audio(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Апсемплирует сигнал из orig_sr до target_sr (target_sr > orig_sr) с фильтрацией"""
        if target_sr <= orig_sr:
            raise ValueError("target_sr должен быть больше orig_sr для апсемплинга")
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        # полифазный ресемплер
        upsampled = resample_poly(samples, up, down)
        return upsampled.astype(np.float32)

    def downsample_audio(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Даунсемплирует сигнал из orig_sr до target_sr (target_sr < orig_sr) с фильтрацией"""
        if target_sr >= orig_sr:
            raise ValueError("target_sr должен быть меньше orig_sr для даунсемплинга")
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        # полифазный ресемплер
        downsampled = resample_poly(samples, up, down)
        return downsampled.astype(np.float32)

