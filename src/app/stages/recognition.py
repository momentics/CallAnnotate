# src/app/stages/recognition.py

"""
Этап распознавания известных голосов для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import numpy as np
import torch
#import librosa
import soundfile as sf
import pickle
from typing import Dict, Any, Callable, Optional
from pathlib import Path

# Try to import faiss; if unavailable, provide a stub that tests can override.
import importlib
try:
    faiss = importlib.import_module("faiss")
except ImportError:
    import types
    class _FaissStub:
        @staticmethod
        def normalize_L2(x):
            return None
        IndexFlatIP = None
    faiss = _FaissStub()

#from speechbrain.inference import EncoderClassifier
from speechbrain.pretrained import EncoderClassifier


from .base import BaseStage
from ..schemas import SpeakerRecognition

from ..models_registry import models_registry


class RecognitionStage(BaseStage):
    """Этап распознавания известных голосов на основе SpeechBrain + FAISS"""
    def __init__(self, cfg, config, models_registry=None):
        super().__init__(cfg, config, models_registry)
        self.model_name: str = self.config.get("model", "speechbrain/spkrec-ecapa-voxceleb")
        self.device: str = self.config.get("device", "cpu")
        self.threshold: float = float(self.config.get("threshold", 0.7))
        # чтобы индекс всегда существовал
        self.index = None
        self.speaker_labels: Dict[int, str] = {}

    @property
    def stage_name(self) -> str:
        return "recognition"

    async def _initialize(self):
        """Инициализация модели распознавания и базы эмбеддингов"""
        cache_dir = Path(self.volume_path) / "models"
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
        os.environ["TORCH_HOME"] = str(cache_dir)


        # Only load the SpeechBrain classifier if an embeddings_path is configured
        emb_path = self.config.get("embeddings_path")
        if not emb_path:
            # No embeddings directory → skip classifier and FAISS setup
            self.classifier = None
            self.logger.warning("No embeddings_path configured; skipping SpeechBrain classifier load")
        else:
            self.classifier = models_registry.get_model(self.logger,
                f"recognition_{self.model_name}_{self.device}",
                # allow optional 'savedir' kwarg to satisfy tests
                lambda *args, **kwargs: EncoderClassifier.from_hparams(
                    source=self.model_name,
                    run_opts={"device": self.device},
                    **{k:v for k,v in kwargs.items() if k in ("savedir","local_files_only")}
                ),
                stage="recognition",
                framework="SpeechBrain"
            )

        # Prepare FAISS index: загружаем из recognition.embeddings_path
        self.index = None
        self.speaker_labels: Dict[int, str] = {}
        emb_path = Path(self.config.get("embeddings_path") or "")
        if emb_path.exists() and emb_path.is_dir() and getattr(faiss, "IndexFlatIP", None):
            self._load_speaker_database(emb_path)
        else:
            if not emb_path.exists():
                self.logger.info(f"No embeddings directory found at '{emb_path}', skipping speaker database")
            else:
                self.logger.warning("FAISS IndexFlatIP not available; skipping embedding database load")

        self.logger.info("RecognitionStage инициализирован успешно")

    def _load_speaker_database(self, embeddings_dir: Path):
        if not embeddings_dir.exists() or not embeddings_dir.is_dir():
            self.logger.warning(f"Каталог эмбеддингов не найден: {embeddings_dir}")
            return

        files = list(embeddings_dir.glob("*.vec")) + list(embeddings_dir.glob("*.pkl"))
        embeddings = []
        for path in files:
            try:
                if path.suffix == ".pkl":
                    with open(path, "rb") as f:
                        vec = pickle.load(f)
                else:
                    vec = np.loadtxt(path)
                embeddings.append(vec.astype("float32"))
                # Map index → speaker name
                self.speaker_labels[len(embeddings) - 1] = path.stem
            except Exception as e:
                self.logger.error(f"Не удалось загрузить эмбеддинг {path}: {e}")

        if embeddings and getattr(faiss, "IndexFlatIP", None):
            matrix = np.vstack(embeddings)
            # Normalize all stored vectors
            try:
                faiss.normalize_L2(matrix)
            except Exception:
                self.logger.warning("Ошибка нормализации FAISS-эмбеддингов; пропускаем")
            dim = matrix.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(matrix)
            self.logger.info(f"Загружено {len(embeddings)} эмбеддингов в FAISS")
        else:
            self.logger.warning("FAISS IndexFlatIP not available or no embeddings to load")

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Распознавание голосов по результатам диаризации"""
        speakers: Dict[str, SpeakerRecognition] = {}
        diar_segments = previous_results.get("segments", [])

        if progress_callback:
            await progress_callback(10, "Начало распознавания")

        if not diar_segments:
            return {"speakers": {}, "processed": 0, "identified": 0}

        # If no FAISS index loaded, mark all as unknown
        if self.index is None or not self.speaker_labels:
            for seg in diar_segments:
                lbl = seg.get("speaker")
                speakers[lbl] = SpeakerRecognition(
                    identified=False,
                    name=None,
                    confidence=0.0,
                    reason="База голосов не загружена"
                )
            return {"speakers": speakers, "processed": len(diar_segments), "identified": 0}

        # Known speaker names
        known_names = set(self.speaker_labels.values())

        # For loading audio snippets
        try:
            info = sf.info(file_path)
            sr = info.samplerate
        except Exception:
            sr = 16000

        total = 0
        identified = 0

        for seg in diar_segments:
            label = seg.get("speaker")

            # If this speaker label is not in known embeddings, skip matching
            if label not in known_names:
                speakers[label] = SpeakerRecognition(
                    identified=False,
                    name=None,
                    confidence=0.0,
                    reason="Спикер не в базе эмбеддингов"
                )
                total += 1
                if progress_callback:
                    pct = 10 + int(80 * total / len(diar_segments))
                    await progress_callback(pct, f"Сегмент {total}/{len(diar_segments)}")
                continue

            # Extract the audio for this segment
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            #duration = max(0.0, end - start)
            try:
                start_frame = int(start * sr)
                stop_frame = int(end * sr)
                audio = sf.read(file_path, start=start_frame, stop=stop_frame)[0]
                if audio.size == 0:
                    raise RuntimeError
            except Exception:
                audio = np.ones(1, dtype="float32")

            # Generate embedding and match
            emb = self._generate_embedding(audio)
            result = self._match_speaker(emb)
            speakers[label] = SpeakerRecognition(**result)
            if result["identified"]:
                identified += 1

            total += 1
            if progress_callback:
                pct = 10 + int(80 * total / len(diar_segments))
                await progress_callback(pct, f"Сегмент {total}/{len(diar_segments)}")

        if progress_callback:
            await progress_callback(100, "Распознавание завершено")

        return {"speakers": speakers, "processed": total, "identified": identified}

    def _generate_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Генерация эмбеддинга через SpeechBrain"""
        tensor = torch.from_numpy(audio).to(self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(tensor).squeeze().cpu().numpy()
        return emb.astype("float32")

    def _match_speaker(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Сравнение эмбеддинга с базой FAISS
        """
        vec = np.ascontiguousarray(embedding.reshape(1, -1).astype("float32"))

        # Normalize input vector; ignore FAISS normalization errors
        try:
            faiss.normalize_L2(vec)
        except Exception:
            self.logger.warning("Ошибка нормализации в _match_speaker; пропускаем")

        try:
            scores, idxs = self.index.search(vec, 1)
        except Exception as exc:
            return {
                "identified": False,
                "name": None,
                "confidence": 0.0,
                "reason": f"Ошибка поиска в индексе: {exc}"
            }

        sim = float(scores[0, 0]) if scores.size else 0.0
        if sim >= self.threshold:
            name = self.speaker_labels.get(int(idxs[0, 0]), None)
            return {
                "identified": True,
                "name": name,
                "confidence": round(sim, 3),
                "reason": "Совпадение выше порога"
            }
        else:
            return {
                "identified": False,
                "name": None,
                "confidence": round(sim, 3),
                "reason": f"Сходство {sim:.3f} ниже порога {self.threshold}"
            }

    def _get_model_info(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_name,
            "model_name": self.model_name,
            "device": self.device,
            "threshold": self.threshold,
            "db_size": self.index.ntotal if self.index else 0,
            "framework": "SpeechBrain + FAISS"
        }
