import soundfile as sf
import torch
from collections import deque
from pyannote.core import Segment, Annotation, SlidingWindow
from pyannote.audio import Inference
from pyannote.audio.pipelines.clustering import AHC

import DeepFilterNet

class IncrementalDiarizationTranscription:
    """
    Класс для пакетной диаризации и сегментной ASR с инкрементальным возвратом результатов.
    1) Читает аудио блоками.
    2) Применяет VAD → DeepFilterNet → эмбеддинг → кластеризацию → ASR.
    3) Возвращает аннотации и транскрипты по мере обработки.
    """

    def __init__(
        self,
        vad_model_name: str = "pyannote/voice-activity-detection",
        emb_model_name: str = "pyannote/embedding",
        asr_processor=None,
        asr_model=None,
        device: str = None,
        window_duration: float = 30.0,
        window_step: float = 15.0,
        cluster_threshold: float = 0.7,
        min_segment_duration: float = 0.5,
        collar: float = 0.1,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Модели pyannote.audio
        self.vad = Inference(vad_model_name, device=self.device)
        self.embedding = Inference(emb_model_name, device=self.device)
        # DeepFilterNet
        self.denoiser = DeepFilterNet().to(self.device)
        self.denoiser.eval()
        # ASR (WhisperProcessor + WhisperForConditionalGeneration)
        self.asr_processor = asr_processor
        self.asr_model = asr_model.to(self.device) if asr_model is not None else None
        # Слайдинг-окно и пост-обработка
        self.sw = SlidingWindow(duration=window_duration, step=window_step)
        self.cluster = AHC(metric="cosine", threshold=cluster_threshold)
        self.min_duration = min_segment_duration
        self.collar = collar
        # Буферы и результаты
        self._waveform_buffer = deque()
        self._timeline_buffer = Annotation()
        self._transcripts = []

    def apply_denoiser(self, raw: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.denoiser(raw)

    def process_chunk(self, chunk: torch.Tensor, sample_rate: int):
        """
        Обработка одного аудиоблока:
        - VAD → сегменты
        - DeepFilterNet → очистка
        - эмбеддинг → кластеризация
        - сохранение сегментов в Annotation
        - ASR по сегментам
        """
        # 1. VAD
        speech_tl = self.vad({"waveform": chunk.unsqueeze(0), "sample_rate": sample_rate})
        # 2. По сегментам речи
        for seg in speech_tl.support(collar=self.collar).segments:
            start_sample = int(seg.start * sample_rate)
            end_sample   = int(seg.end   * sample_rate)
            raw = chunk[0, start_sample:end_sample].unsqueeze(0).to(self.device)
            # 3. Denoise
            clean = self.apply_denoiser(raw)
            # 4. Embedding
            emb = self.embedding({"waveform": clean, "sample_rate": sample_rate}).squeeze(0).cpu().numpy()
            # 5. Сохранение для кластеризации
            self._timeline_buffer[seg] = emb
        # 6. Кластеризация и пост-обработка
        self._cluster_and_clean()
        # 7. ASR
        self._run_asr(sample_rate)

    def _cluster_and_clean(self):
        # Собрать фичи
        segments, embeddings = zip(*[
            (seg, emb) for seg, _, emb in self._timeline_buffer.itertracks(yield_label=True)
        ]) if len(self._timeline_buffer.get_timeline()) else ([], [])
        if not segments:
            return
        labels = self.cluster.predict(list(embeddings))
        diar = Annotation()
        for seg, lbl in zip(segments, labels):
            diar[seg] = f"speaker_{lbl}"
        merged = diar.support(collar=self.collar)
        # Фильтрация коротких
        cleaned = Annotation()
        for seg, track, spk in merged.itertracks(yield_label=True):
            if seg.duration >= self.min_duration:
                cleaned[seg, track] = spk
        self._diarization = cleaned

    def _run_asr(self, sample_rate: int):
        if self.asr_model is None or self.asr_processor is None:
            return
        for seg, _, spk in self._diarization.itertracks(yield_label=True):
            start, end = seg.start, seg.end
            # Извлечение из буфера или из исходного файла по таймкодам
            raw = self._get_waveform_segment(start, end)
            features = self.asr_processor(raw, sampling_rate=sample_rate,
                                          return_tensors="pt", padding="longest").input_features.to(self.device)
            ids = self.asr_model.generate(features)
            text = self.asr_processor.batch_decode(ids, skip_special_tokens=True).strip()
            self._transcripts.append({"speaker": spk, "segment": (start, end), "text": text})

    def _get_waveform_segment(self, start: float, end: float) -> torch.Tensor:
        """
        Получить сегмент из буфера или считать заново.
        Здесь – упрощённая реализация, предполагающая access к оригинальному файлу.
        """
        # Для больших файлов лучше читать прям по таймкодам из disk
        return torch.from_numpy(self._original_waveform[int(start*self._sr):int(end*self._sr)])

    def load_audio(self, path: str):
        waveform, sr = sf.read(path)
        self._original_waveform = waveform
        self._sr = sr

    def process(self, path: str):
        """
        Основной метод:
        - Загружает аудио
        - Разбивает на окна через SlidingWindow
        - Инкрементально process_chunk
        """
        self.load_audio(path)
        for window in self.sw(Segment(0, len(self._original_waveform)/self._sr), align_last=True):
            start, end = window.start, window.end
            start_sample = int(start * self._sr)
            end_sample   = int(end   * self._sr)
            chunk = torch.from_numpy(self._original_waveform[start_sample:end_sample]).unsqueeze(0)
            self.process_chunk(chunk, self._sr)
            yield {
                "diarization": self._diarization,
                "transcripts": list(self._transcripts)
            }
            self._transcripts.clear()
