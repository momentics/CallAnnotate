# -*- coding: utf-8 -*-
"""
sox_tools.py — опциональные вызовы SoX для статического шумоподавления.

Если sox недоступен или шаг завершается ошибкой, возвращаем исходный путь
без исключений (устойчивость на CI и в минимальных окружениях).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


def is_sox_available() -> bool:
    return shutil.which("sox") is not None


def apply_sox_noise_reduce(
    src_path: str,
    dst_dir: str,
    task_id: str,
    profile_duration: float,
    reduction: float,
    rms_gain_normalization: bool,
    target_rms_db: float
) -> str:
    """
    Применяет двухшаговый SoX:
    - формирование профиля шума с первых profile_duration секунд;
    - применение noisered + опциональная нормализация усиления.

    Возвращает путь к временному WAV или исходный src_path при ошибке.
    """
    if not is_sox_available():
        return src_path

    tmp_dir = Path(dst_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prof = tmp_dir / f"{task_id}.noiseprof"
    dst = tmp_dir / f"{task_id}_sox_tmp.wav"

    try:
        # Профиль шума
        subprocess.run(
            [
                "sox", src_path, "-n",
                "trim", "0", str(profile_duration),
                "noiseprof", str(prof)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Применение профиля
        cmd = ["sox", src_path, str(dst), "noisered", str(prof), str(reduction)]
        if rms_gain_normalization:
            # gain -n не принимает целевой RMS, но нормализует пики;
            # для RMS контролируем отдельно на конце пайплайна.
            cmd += ["gain", "-n"]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return str(dst)
    except Exception:
        # При любых ошибках просто возвращаем исходник
        return src_path
    finally:
        try:
            prof.unlink(missing_ok=True)
        except Exception:
            pass
