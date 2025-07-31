import ctypes
import numpy as np
import pytest


# ---------------------------------------------------------------------
#  Вспомогательный стаб C-библиотеки RNNoise
# ---------------------------------------------------------------------
class _DummyLib:
    def __init__(self):
        self._state = ctypes.c_void_p(1)

    def rnnoise_create(self):
        return self._state

    def rnnoise_destroy(self, _state):
        return None

    def rnnoise_process_frame(self, _state, out_frame, in_frame):
        # копируем вход → выход + фиктивная вероятность речи
        ctypes.memmove(out_frame, in_frame,
                       480 * ctypes.sizeof(ctypes.c_float))
        return ctypes.c_float(0.42)


# ---------------------------------------------------------------------
#  Фикстура, подменяющая загрузку SO-библиотеки
# ---------------------------------------------------------------------
@pytest.fixture
def mock_rnnoise_lib(monkeypatch):
    monkeypatch.setattr("ctypes.util.find_library",
                        lambda name: "/usr/lib/librnnoise_mock.so")
    monkeypatch.setattr("ctypes.CDLL", lambda path: _DummyLib())
    yield


# ---------------------------------------------------------------------
#  Юнит-тесты
# ---------------------------------------------------------------------
def test_init_success(mock_rnnoise_lib):
    from app.rnnoise_wrapper import RNNoise

    rn = RNNoise()
    assert rn.sample_rate == 48_000


def test_denoise_chunk_identity(mock_rnnoise_lib):
    from app.rnnoise_wrapper import RNNoise, FRAME_SIZE

    rn = RNNoise()
    # два 10-мс фрейма тестового сигнала
    data = np.linspace(-1.0, 1.0, FRAME_SIZE * 2, dtype=np.float32)
    res = np.concatenate([frame for _, frame in rn.denoise_chunk(data)])

    # длина сохранена, сигнал не изменился
    assert res.shape == data.shape
    np.testing.assert_allclose(res, data, rtol=1e-6, atol=1e-6)
