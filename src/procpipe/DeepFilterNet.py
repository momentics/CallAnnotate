import torch
import torchaudio
from df import deepfilternet3 as _DFNet  # from https://github.com/Rikorose/DeepFilterNet


class DeepFilterNet:
    """
    Обёртка для DeepFilterNet (Rikorose), интегрируемая в пайплайн.
    Позволяет загружать предобученную модель, выполнять шумоподавление
    и переключаться в режим eval.
    """
    def __init__(
        self,
        model_path: str = None,
        device: str = None
    ):
        """
        Args:
            model_path: путь к файлу чекпоинта DeepFilterNet (.pt или .ckpt).
                        Если None, загружается дефолтная модель из пакета.
            device: "cuda" или "cpu". По умолчанию выбирается автоматически.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Инициализируем внутреннюю модель
        self.model = _DFNet()
        if model_path:
            checkpoint = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Применяет шумоподавление к входному сигналу.
        
        Args:
            waveform: тензор формы (batch, time) или (time,), dtype=float32
            sample_rate: частота дискретизации (например, 16000)
        
        Returns:
            clean: тензор той же формы и dtype, но очищенный от шума.
        """
        # DeepFilterNet ожидает (batch, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        # Приведение частоты, если модель требует 48 kHz
        if sample_rate != self.model.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.model.sample_rate
            )
        # inference
        clean = self.model(waveform)
        # обратно к оригинальному sample_rate
        if sample_rate != self.model.sample_rate:
            clean = torchaudio.functional.resample(
                clean, orig_freq=self.model.sample_rate, new_freq=sample_rate
            )
        return clean.squeeze(0)
