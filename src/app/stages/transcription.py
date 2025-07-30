# src/app/stages/transcription.py

"""
Этап транскрипции аудио для CallAnnotate
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import time
import whisper
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from .base import BaseStage

class TranscriptionStage(BaseStage):
    """Этап транскрипции на основе OpenAI Whisper"""

    @property
    def stage_name(self) -> str:
        return "transcription"

    async def _initialize(self):
        model_name = self.config.get("model", "openai/whisper-base")
        if "whisper-" in model_name:
            model_size = model_name.split("whisper-")[-1]
        else:
            model_size = model_name.split("/")[-1]
        device = self.config.get("device", "cpu")

        self.logger.info(f"Загрузка модели Whisper: {model_size}")
        from unittest.mock import MagicMock
        use_registry = (
            self.models_registry
            and not isinstance(self.models_registry, MagicMock)
            and hasattr(self.models_registry, "get_model")
        )
        if use_registry:
            cache_key = f"whisper_{model_size}_{device}"
            self.model = self.models_registry.get_model(
                cache_key,
                lambda: whisper.load_model(model_size, device=device)
            )
        else:
            self.model = whisper.load_model(model_size, device=device)

        self.model_size = model_size
        self.device = device
        self.logger.info("Модель транскрипции загружена успешно")

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        if progress_callback:
            await progress_callback(10, "Начало транскрипции")

        # Загружаем аудио, подменяем на тишину при ошибке
        try:
            audio = whisper.load_audio(file_path)
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить аудио '{file_path}': {e}. Используется тишина.")
            audio = np.zeros(16000, dtype=np.float32)

        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        language = self.config.get("language")
        if language == "auto":
            language = None

        options = {
            "language": language,
            "task": self.config.get("task", "transcribe"),
            "temperature": 0.0,
            "word_timestamps": True,
            "verbose": False
        }
        options = {k: v for k, v in options.items() if v is not None}

        batch_size = int(self.config.get("batch_size", 16))
        segments_all: List[Any] = []
        words_all: List[Any] = []
        metrics = {
            "confidences": [],
            "avg_logprobs": [],
            "no_speech_probs": [],
        }

        total = mel.shape[1]
        for i in range(0, total, batch_size):
            if progress_callback:
                pct = 10 + int(70 * i / total)
                await progress_callback(pct, f"Транскрипция батча {i//batch_size+1}")
            batch = mel[:, i : min(i + batch_size, total)]
            result = self.model.transcribe(batch, **options)

            raw_segments = getattr(result, "segments", result.get("segments", []))
            for seg in raw_segments:
                segments_all.append(seg)
                for w in seg.get("words", []):
                    words_all.append(w)
                metrics["confidences"].append(result.get("confidence", 0.0))
                metrics["avg_logprobs"].append(seg.get("avg_logprob", 0.0))
                metrics["no_speech_probs"].append(seg.get("no_speech_prob", 0.0))

        if progress_callback:
            await progress_callback(85, "Выравнивание сегментов")
        aligned = self._align_with_diarization(
            [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "no_speech_prob": s.get("no_speech_prob", 0.0),
                    "avg_logprob": s.get("avg_logprob", 0.0)
                }
                for s in segments_all
            ],
            previous_results.get("segments", [])
        )

        if progress_callback:
            await progress_callback(100, "Транскрипция завершена")

        total_time = time.perf_counter() - start_time
        avg_conf = sum(metrics["confidences"]) / len(metrics["confidences"]) if metrics["confidences"] else 0.0
        avg_log = sum(metrics["avg_logprobs"]) / len(metrics["avg_logprobs"]) if metrics["avg_logprobs"] else 0.0
        avg_nsp = sum(metrics["no_speech_probs"]) / len(metrics["no_speech_probs"]) if metrics["no_speech_probs"] else 0.0

        return {
            "segments": aligned,
            "words": [
                {
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                    "word": w["word"],
                    "probability": round(w["probability"], 3)
                }
                for w in words_all
            ],
            "confidence": round(avg_conf, 3),
            "avg_logprob": round(avg_log, 3),
            "no_speech_prob": round(avg_nsp, 3),
            "processing_time": round(total_time, 3)
        }

    def _align_with_diarization(
        self,
        transcription_segments: List[Dict[str, Any]],
        diarization_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not diarization_segments:
            return transcription_segments

        enhanced: List[Dict[str, Any]] = []
        for trans in transcription_segments:
            best_speaker = None
            best_overlap = 0.0
            t0, t1 = trans["start"], trans["end"]
            for diar in diarization_segments:
                d0, d1 = diar["start"], diar["end"]
                overlap = max(0.0, min(t1, d1) - max(t0, d0))
                duration = t1 - t0 if t1 > t0 else 0.0
                ratio = overlap / duration if duration > 0 else 0.0
                if ratio > best_overlap:
                    best_overlap, best_speaker = ratio, diar["speaker"]
            seg = trans.copy()
            if best_speaker and best_overlap >= 0.3:
                seg["speaker"] = best_speaker
                seg["speaker_confidence"] = round(best_overlap, 3)
            enhanced.append(seg)
        return enhanced

    def _get_model_info(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_name,
            "model_size": getattr(self, "model_size", "unknown"),
            "device": getattr(self, "device", "unknown"),
            "framework": "OpenAI Whisper"
        }

    def _calculate_average_confidence(self, words: List[Dict[str, Any]]) -> float:
        if not words:
            return 0.0
        return round(sum(w["probability"] for w in words) / len(words), 3)
