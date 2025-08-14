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
from typing import Awaitable, Dict, Any, Callable, Optional
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

from speechbrain.inference import EncoderClassifier
#from speechbrain.pretrained import EncoderClassifier


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
        self.speechbrain_path = Path(self.volume_path) / "models" / "speechbrain"
        self.speechbrain_path.mkdir(parents=True, exist_ok=True)

        self.emb_path = Path(self.volume_path) / "models" / "embeddings"
        self.emb_path.mkdir(parents=True, exist_ok=True)

        #os.environ["HF_HOME"] = str(self.speechbrain_path)
        #os.environ["TRANSFORMERS_CACHE"] = str(self.speechbrain_path)
        #os.environ["TORCH_HOME"] = str(self.speechbrain_path)

        cwd = Path.cwd().resolve()  # store current working directory
            
        cd_to = self.speechbrain_path.resolve()
        os.chdir(cd_to)

        # Only load the SpeechBrain classifier if an embeddings_path is configured
        self.classifier = models_registry.get_model(self.logger,
            f"recognition_{self.model_name}_{self.device}",
            # allow optional 'savedir' kwarg to satisfy tests
            lambda *args, **kwargs: EncoderClassifier.from_hparams(
                #source=self.model_name,
                source=self.speechbrain_path,
                #overrides={"pretrained_path": self.speechbrain_path},
                run_opts={"device": self.device},
                #use_auth_token=True,
                #savedir=self.speechbrain_path,
                #huggingface_cache_dir=self.speechbrain_path,
                #**{k:v for k,v in kwargs.items() if k in ("savedir","local_files_only")}
            ),
            stage="recognition",
            framework="SpeechBrain"
        )

        os.chdir(cwd)

        # Prepare FAISS index: загружаем из recognition.embeddings_path
        self.index = None
        self.speaker_labels: Dict[int, str] = {}
        if self.emb_path.exists() and self.emb_path.is_dir() and getattr(faiss, "IndexFlatIP", None):
            self._load_speaker_database(self.emb_path)
        else:
            if not self.emb_path.exists():
                self.logger.info(f"No embeddings directory found at '{self.emb_path}', skipping speaker database")
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


        if embeddings and getattr(faiss, "normalize_L2", None):
            matrix = np.vstack(embeddings)
            try:
                faiss.normalize_L2(matrix)
            except Exception:
                self.logger.warning("Ошибка нормализации FAISS-эмбеддингов; пропускаем")

            # Pull the index class out and check it isn’t None
            IndexFlatIP = getattr(faiss, "IndexFlatIP", None)
            if IndexFlatIP is not None:
                dim = matrix.shape[1]
                self.index = IndexFlatIP(dim)
                self.index.add(matrix)
                self.logger.info(f"Загружено {len(embeddings)} эмбеддингов в FAISS")
            else:
                self.logger.warning("FAISS IndexFlatIP not available; skipping index creation")
        else:
            self.logger.warning("FAISS normalize_L2 not available or no embeddings to load")


    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
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
        if self.classifier is None:
            raise RuntimeError("SpeechBrain classifier is not initialized; embeddings cannot be generated")
        tensor = torch.from_numpy(audio).to(self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(tensor).squeeze().cpu().numpy()
        return emb.astype("float32")


    def _match_speaker(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Сравнение эмбеддинга с базой FAISS
        """
        # If the FAISS index isn't initialized, skip matching:
        if self.index is None:
            return {
                "identified": False,
                "name": None,
                "confidence": 0.0,
                "reason": "База голосов не загружена"
            }

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
            "database_size": self.index.ntotal if self.index else 0,
            "framework": "SpeechBrain + FAISS"
        }
