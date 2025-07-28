# -*- coding: utf-8 -*-
"""
Этап распознавания известных голосов для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import numpy as np
import torch
import librosa
import pickle
from typing import Dict, Any, Optional, Callable
from pathlib import Path

try:
    import faiss
except ImportError:
    faiss = None

from speechbrain.inference import EncoderClassifier

from .base import BaseStage


class RecognitionStage(BaseStage):
    """Этап распознавания известных голосов на основе SpeechBrain"""
    
    @property
    def stage_name(self) -> str:
        return "recognition"
    
    async def _initialize(self):
        """Инициализация модели распознавания"""
        model_name = self.config.get("model", "speechbrain/spkrec-ecapa-voxceleb")
        device = self.config.get("device", "cpu")
        embeddings_path = self.config.get("embeddings_path")
        
        self.logger.info(f"Загрузка модели распознавания: {model_name}")
        
        if self.models_registry:
            cache_key = f"speaker_recognition_{model_name}_{device}"
            self.classifier = self.models_registry.get_model(
                cache_key,
                lambda: EncoderClassifier.from_hparams(
                    source=model_name,
                    run_opts={"device": device}
                )
            )
        else:
            self.classifier = EncoderClassifier.from_hparams(
                source=model_name,
                run_opts={"device": device}
            )
        
        # Загрузка базы известных голосов
        self.index = None
        self.speaker_labels = {}
        
        if embeddings_path and Path(embeddings_path).exists():
            self._load_speaker_database(embeddings_path)
        
        self.model_name = model_name
        self.device = device
        self.threshold = self.config.get("threshold", 0.7)
        
        self.logger.info("Модель распознавания загружена успешно")
    
    def _load_speaker_database(self, embeddings_path: str):
        """Загрузка базы данных голосовых эмбеддингов"""
        try:
            embeddings_dir = Path(embeddings_path)
            
            # Поиск файлов эмбеддингов
            embedding_files = list(embeddings_dir.glob("*.vec")) + list(embeddings_dir.glob("*.pkl"))
            
            if not embedding_files:
                self.logger.warning(f"Не найдены файлы эмбеддингов в {embeddings_path}")
                return
            
            embeddings = []
            labels = []
            
            for emb_file in embedding_files:
                try:
                    if emb_file.suffix == '.pkl':
                        with open(emb_file, 'rb') as f:
                            embedding = pickle.load(f)
                    else:  # .vec файл
                        embedding = np.loadtxt(emb_file)
                    
                    # Имя спикера из имени файла
                    speaker_name = emb_file.stem
                    
                    embeddings.append(embedding)
                    labels.append(speaker_name)
                    self.speaker_labels[len(labels) - 1] = speaker_name
                    
                except Exception as e:
                    self.logger.error(f"Ошибка загрузки эмбеддинга {emb_file}: {e}")
            
            # Создание FAISS индекса
            if faiss and embeddings:
                embeddings_matrix = np.vstack(embeddings).astype('float32')
                dimension = embeddings_matrix.shape[1]
                
                self.index = faiss.IndexFlatIP(dimension)  # Cosine similarity
                faiss.normalize_L2(embeddings_matrix)
                self.index.add(embeddings_matrix)
                
                self.logger.info(f"Загружено {len(embeddings)} эмбеддингов голосов")
            else:
                self.logger.warning("FAISS не доступен или нет эмбеддингов")
                
        except Exception as e:
            self.logger.error(f"Ошибка загрузки базы голосов: {e}")
    

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Выполнение распознавания голосов"""

        if progress_callback:
            await progress_callback(10, "Начало распознавания голосов")

        # Получение сегментов диаризации
        diarization_segments = previous_results.get("segments", [])
        if not diarization_segments:
            self.logger.warning("Отсутствуют сегменты диаризации для распознавания")
            return {"speakers": {}, "total_processed": 0, "identified_count": 0}

        # Если база не загружена, создаём результат для каждого спикера
        if self.index is None or not self.speaker_labels:
            speakers = {}
            for seg in diarization_segments:
                speaker = seg.get("speaker")
                speakers[speaker] = {
                    "identified": False,
                    "name": None,
                    "confidence": 0.0,
                    "reason": "База голосов не загружена"
                }
            return {
                "speakers": speakers,
                "total_processed": len(speakers),
                "identified_count": 0
            }




    def _generate_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Генерация голосового эмбеддинга"""
        # Подготовка аудио для SpeechBrain
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        
        # Генерация эмбеддинга
        with torch.no_grad():
            embedding = self.classifier.encode_batch(audio_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def _find_speaker_match(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Поиск совпадения в базе известных голосов"""
        if self.index is None or not self.speaker_labels:
            return {
                "identified": False,
                "name": None,
                "confidence": 0.0,
                "reason": "База голосов не загружена"
            }
        
        try:
            # Нормализация для косинусного сходства
            embedding = embedding.reshape(1, -1).astype('float32')
            if faiss:
                faiss.normalize_L2(embedding)
            
            # Поиск ближайшего соседа
            scores, indices = self.index.search(embedding, 1)
            
            if len(scores[0]) > 0:
                similarity = float(scores[0][0])
                speaker_idx = int(indices[0][0])
                
                if similarity >= self.threshold:
                    speaker_name = self.speaker_labels.get(speaker_idx, f"Unknown_{speaker_idx}")
                    return {
                        "identified": True,
                        "name": speaker_name,
                        "confidence": round(similarity, 3),
                        "reason": "Найдено совпадение в базе"
                    }
                else:
                    return {
                        "identified": False,
                        "name": None,
                        "confidence": round(similarity, 3),
                        "reason": f"Сходство {similarity:.3f} ниже порога {self.threshold}"
                    }
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска совпадения: {e}")
        
        return {
            "identified": False,
            "name": None,
            "confidence": 0.0,
            "reason": "Ошибка поиска в базе"
        }
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Информация о модели распознавания"""
        return {
            "stage": self.stage_name,
            "model_name": getattr(self, 'model_name', 'unknown'),
            "device": getattr(self, 'device', 'unknown'),
            "threshold": getattr(self, 'threshold', 0.7),
            "database_size": self.index.ntotal if self.index else 0,
            "framework": "SpeechBrain + FAISS"
        }
