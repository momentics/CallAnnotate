# src/app/stages/preprocessing.py

# -*- coding: utf-8 -*-
"""
Этап предобработки аудио: SoX-поддержка, чанковая нормализация,
подавление шума RNNoise_Wrapper и DeepFilterNet2
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional, Callable

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from ..rnnoise_wrapper import RNNoise
from .base import BaseStage

try:
    from df.enhance import init_df, enhance
except ImportError:
    init_df = None
    enhance = None


class PreprocessingStage(BaseStage):
    @property
    def stage_name(self) -> str:
        return "preprocess"

    async def _initialize(self):
        if init_df is None or enhance is None:
            raise RuntimeError("Не установлена зависимость deepfilternet")
        self.model, self.df_state, _ = init_df(
            model=self.config.get("model", "DeepFilterNet2"),
            device=self.config.get("device", "cpu")
        )
        # Prepare RNNoise once; allow RNNoise() without args for mocks
        try:
            # DummyRNNoise mock may not accept sample_rate arg
            self.rnnoise = RNNoise()
        except (TypeError, RuntimeError, OSError) as e:
            self.logger.warning(f"RNNoise инициализация не удалась: {e}. Шумоподавление будет пропущено")
            self.rnnoise = None

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        dur_ms = int(self.config.get("chunk_duration", 2.0) * 1000)
        ovl_ms = int(self.config.get("overlap", 0.5) * 1000)
        target_rms_db = float(self.config.get("target_rms", -20.0))

        use_sox = (
            os.getenv("SOX_SKIP", "0") != "1"
            and shutil.which("sox") is not None
        )
        src = file_path
        prof = None
        sox_out = None

        if use_sox:
            prof = os.path.join(tempfile.gettempdir(), f"{task_id}.prof")
            sox_out = os.path.join(tempfile.gettempdir(), f"{task_id}.wav")
            try:
                subprocess.run(
                    ["sox", file_path, "-n", "trim", "0", "2", "noiseprof", prof],
                    check=True
                )
                subprocess.run(
                    ["sox", file_path, sox_out, "noisered", prof, "0.3", "gain", "-n", str(target_rms_db)],
                    check=True
                )
                src = sox_out
            except Exception as e:
                self.logger.warning(f"SoX processing failed, using original file: {e}", exc_info=True)
                src = file_path

        audio = AudioSegment.from_file(src)
        sr = audio.frame_rate
        segments = []
        pos = 0
        total = len(audio)

        # Chunking + RNNoise + DeepFilterNet
        while pos < total:
            end = min(pos + dur_ms, total)
            seg = audio[pos:end]

            if self.rnnoise is not None:
                try:
                    # If mock implements filter(), use it
                    if hasattr(self.rnnoise, "filter"):
                        seg = self.rnnoise.filter(seg)
                    else:
                        # Convert AudioSegment to float32 numpy array in [-1,1]
                        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                        samples /= np.iinfo(seg.array_type).max
                        if seg.channels > 1:
                            samples = samples.reshape(-1, seg.channels).mean(axis=1)

                        # Apply RNNoise frame by frame using correct API
                        denoised_frames = []
                        for _, frame in self.rnnoise.denoise_chunk(samples[np.newaxis, :]):
                            denoised_frames.append(frame)

                        if denoised_frames:
                            denoised = np.concatenate(denoised_frames, axis=0)
                            # Convert back to int16
                            denoised_int = (denoised * np.iinfo(seg.array_type).max).astype(seg.array_type)
                            seg = seg._spawn(denoised_int.tobytes())
                except Exception as e:
                    self.logger.warning(f"RNNoise denoising failed for chunk starting at {pos}ms: {e}", exc_info=True)

            # Normalize using DeepFilterNet2
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= np.iinfo(seg.array_type).max
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels).mean(axis=1)

            processed = enhance(self.model, self.df_state, samples)
            segments.append(processed)

            pos += dur_ms - ovl_ms
            if progress_callback:
                await progress_callback(min(100, int(100 * pos / total)), f"preprocess chunk {len(segments)}")

        if not segments:
            raise RuntimeError("Не удалось создать чанки")

        # Overlap-add
        out = segments[0]
        for arr in segments[1:]:
            overlap_samples = ovl_ms * sr // 1000
            out = np.concatenate((out, arr[overlap_samples:]), axis=0)

        # Final normalization if SoX used
        if use_sox:
            rms = np.sqrt(np.mean(out**2))
            target_lin = 10 ** (target_rms_db / 20)
            if rms > 0:
                out *= (target_lin / rms)

        out_path = file_path.replace(".wav", "_processed.wav")
        sf.write(out_path, out, sr, subtype="PCM_16")

        # Cleanup temp files
        if use_sox:
            for p in (prof, sox_out):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        return {"processed_path": out_path}
