# -*- coding: utf-8 -*-
"""
denoise_dfn.py — интеграция DeepFilterNet2/3, с «тихим» отключением при отсутствии пакета.

Интерфейс унифицирован:
- init(model, device, target_sr)
- enhance(mono_f32, sr) -> mono_f32 (возможно с ресэмплингом)
"""
from __future__ import annotations

from typing import Optional, Tuple
import inspect
import numpy as np
import torch

from .io import resample_float


class DeepFilter:
    def __init__(self, enable: bool, model_name: str = "DeepFilterNet2", device: str = "cpu", target_sr: int = 48_000):
        self.enable = enable
        self.target_sr = target_sr
        self._df_model = None
        self._df_state = None
        self._enhance = None

        if not enable:
            return

        init_df = None
        enhance = None
        for modname in ("df.deepfilternet3", "df.deepfilternet2", "df.deepfilternet", "df.enhance"):
            try:
                mod = __import__(modname, fromlist=["init_df", "enhance"])
                init_df = getattr(mod, "init_df", None)
                enhance = getattr(mod, "enhance", None)
                if callable(init_df) and callable(enhance):
                    break
            except Exception:
                continue

        if not (init_df and enhance):
            # DFN недоступен — «мягко» отключаем
            self.enable = False
            return

        try:
            sig = inspect.signature(init_df)
            if len(sig.parameters) == 0:
                self._df_model, self._df_state, _ = init_df()  # type: ignore
            elif {"model", "device"}.issubset(sig.parameters.keys()):
                self._df_model, self._df_state, _ = init_df(model=model_name, device=device)  # type: ignore
            else:
                self._df_model, self._df_state, _ = init_df(model_name, device)  # type: ignore
            self._enhance = enhance
        except Exception:
            self.enable = False
            self._df_model = None
            self._df_state = None
            self._enhance = None

    def is_ready(self) -> bool:
        return self.enable and (self._df_model is not None) and (self._enhance is not None)

    def process(self, mono_f32: np.ndarray, sr: int) -> np.ndarray:
        """Возвращает обработанный массив или исходный при недоступности DFN."""
        if not self.is_ready():
            return mono_f32
        work = mono_f32
        if sr != self.target_sr:
            work = resample_float(mono_f32, sr, self.target_sr)

        t = torch.from_numpy(work).unsqueeze(0).float()
        try:
            out = self._enhance(self._df_model, self._df_state, t)  # type: ignore
            if isinstance(out, torch.Tensor):
                den = out.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                den = np.asarray(out, dtype=np.float32)
        except Exception:
            return mono_f32

        if self.target_sr != sr:
            den = resample_float(den, self.target_sr, sr)
        return den
