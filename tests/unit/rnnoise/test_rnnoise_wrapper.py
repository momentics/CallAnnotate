# tests/unit/rnnoise/test_rnnoise_wrapper.py

import ctypes
import pytest
import numpy as np
from app.rnnoise_wrapper import RNNoise, _PassthroughLib, FRAME_SIZE

def test_passthrough_behavior():
    lib = _PassthroughLib()
    state = lib.rnnoise_create()
    # input frame of zeros
    in_buf = (ctypes.c_float * FRAME_SIZE)(*[0.0] * FRAME_SIZE)
    out_buf = (ctypes.c_float * FRAME_SIZE)()
    prob = lib.rnnoise_process_frame(state, out_buf, in_buf)
    # Compare as float via .value
    assert prob.value == 0.0

def test_filter_noop_and_rate(monkeypatch):
    seg = pytest.importorskip("pydub").AudioSegment.silent(duration=10, frame_rate=48000)
    # force passthrough
    rn = RNNoise(allow_passthrough=True)
    out = rn.filter(seg)
    assert isinstance(out, type(seg))
    # test denoise_chunk generator
    data = np.zeros(FRAME_SIZE, dtype=float)
    frames = list(rn.denoise_chunk(data))
    assert all(isinstance(f, np.ndarray) for _, f in frames)
