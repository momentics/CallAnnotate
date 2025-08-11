import os
import threading
from typing import Callable, Optional
import numpy
import soundfile as sf
import torch
from collections import defaultdict, deque
from pyannote.core import Segment, Annotation, SlidingWindow
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering as AHC

from transformers import WhisperProcessor, WhisperForConditionalGeneration

import DeepFilterNet
import ThreadSafeIncrementalMerger


class ThreadSafeIncrementalDiarizationTranscription:
    """
    Потокобезопасный каркас для инкрементальной диаризации и сегментного ASR.
    Каждый поток может вызывать process(path), но модели и merger — общие.
    """
    _instance_lock = threading.Lock()
    _shared_state = {}

    def __init__(self,
                 vad_model: str = "pyannote/voice-activity-detection",
                 emb_model: str = "pyannote/embedding",
                 asr_processor: WhisperProcessor = None,
                 asr_model: WhisperForConditionalGeneration = None,
                 window_duration: float = 60.0,
                 window_step: float = 30.0,
                 cluster_threshold: float = 0.6,
                 min_segment_duration: float = 0.5,
                 collar: float = 0.2,
                 max_gap: float = 0.5):
        """
        Использует Borg-паттерн для шаринга весов и merger между экземплярами.
        """
        self.__dict__ = self._shared_state
        if not hasattr(self, "_initialized"):
            # инициализация моделей
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.vad = Inference(vad_model, device=device, use_auth_token = os.getenv("HF_TOKEN"))
            self.embedding = Inference(emb_model, device=device, use_auth_token = os.getenv("HF_TOKEN"))
            self.denoiser = DeepFilterNet().to(device)
            self.denoiser.eval()
            self.asr_processor = asr_processor
            self.asr_model = asr_model.to(device) if asr_model else None

            self.sw = SlidingWindow(duration=window_duration, step=window_step)
            self.cluster = AHC(metric="cosine", threshold=cluster_threshold)
            self.min_duration = min_segment_duration
            self.collar = collar

            self.merger = ThreadSafeIncrementalMerger(max_gap=max_gap)
            self._instance_lock = threading.Lock()
            self._initialized = True

    def _apply_denoise(self, raw: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.denoiser(raw)

    def _process_window(self, waveform: torch.Tensor, sr: int):
        """
        Обрабатывает один кадр окна: VAD -> Denoise -> Embedding -> Cluster -> ASR
        Возвращает локальную Annotation и словарь транскриптов для merger.
        """
        speech_tl = self.vad({"waveform": waveform.unsqueeze(0), "sample_rate": sr})
        feats = []
        for seg in speech_tl.support(collar=self.collar).segments:
            start_s, end_s = int(seg.start * sr), int(seg.end * sr)
            raw = waveform[0, start_s:end_s].unsqueeze(0)
            clean = self._apply_denoise(raw)
            emb = self.embedding({"waveform": clean, "sample_rate": sr}).squeeze(0).cpu().numpy()
            feats.append((seg, emb))

        # кластеризация
        if not feats:
            return Annotation(), {}
        segments, embeddings = zip(*feats)
        labels = self.cluster.predict(list(embeddings))
        diar = Annotation()
        for seg, lbl in zip(segments, labels):
            diar[seg] = f"speaker_{lbl}"
        merged = diar.support(collar=self.collar)
        cleaned = Annotation()
        for seg, track, spk in merged.itertracks(yield_label=True):
            if seg.duration >= self.min_duration:
                cleaned[seg, track] = spk

        # ASR
        transcripts = defaultdict(list)
        if self.asr_model and self.asr_processor:
            for seg, _, spk in cleaned.itertracks(yield_label=True):
                start, end = seg.start, seg.end
                raw_samples = waveform[0, int(start*sr):int(end*sr)].cpu().numpy()
                inputs = self.asr_processor(raw_samples, sampling_rate=sr,
                                            return_tensors="pt", padding="longest")
                input_feats = inputs.input_features.to(self.asr_model.device)
                ids = self.asr_model.generate(input_feats)
                text = self.asr_processor.batch_decode(ids, skip_special_tokens=True).strip()
                transcripts[spk].append({"segment": (start, end), "text": text})

        return cleaned, transcripts

    def process_file(self, path: str):
        """
        Генератор: инкрементально выдаёт полностью завершённые фрагменты
        диаризации и транскриптов по мере обработки окон.
        """
        waveform_np, sr = sf.read(path)
        return self.process_wave(waveform_np, sr)


    def process_wave(self, waveform_np: numpy.ndarray, samplerate: int):
        """
        Генератор: инкрементально выдаёт полностью завершённые фрагменты
        диаризации и транскриптов по мере обработки окон.
        """
        waveform = torch.from_numpy(waveform_np).unsqueeze(0).to(
            self.vad.device if hasattr(self.vad, "device") else "cpu"
        )
        for window in self.sw(Segment(0, len(waveform_np)/samplerate), align_last=True):
            start, end = window.start, window.end
            start_i, end_i = int(start * samplerate), int(end * samplerate)
            chunk = waveform[:, start_i:end_i]

            diar_part, transcripts_part = self._process_window(chunk, samplerate)

            # собираем завершённые транскрипты
            completed = self.merger.merge(transcripts_part)

            yield {
                "partial_diarization": diar_part,
                "completed_transcripts": completed
            }
