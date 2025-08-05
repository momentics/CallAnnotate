# -*- coding: utf-8 -*-
"""
Реестр моделей для CallAnnotate с ленивой загрузкой и оффлайн-кешем на volume/models.
"""

import logging
import threading
from typing import Dict, Any, Callable
import torch
from pathlib import Path
import os

class ModelRegistry:
    """Singleton реестр для ленивой загрузки и кеширования ML-моделей"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> 'ModelRegistry':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return

        self._models: Dict[str, Any] = {}
        self._model_info: Dict[str, Dict[str, Any]] = {}
        self._load_lock = threading.RLock()
        self._initialized = True

    def get_model(self, logger, key: str, loader: Callable[[], Any], **metadata) -> Any:
        
        with self._load_lock:
            if key not in self._models:
                logger.info(f"Загрузка модели: {key}")
                try:
                    model = loader()
                    self._models[key] = model
                    mem = self._estimate_model_memory(model)
                    self._model_info[key] = {"memory_usage": mem, **metadata}
                    logger.info(f"Модель {key} загружена, память {mem:.1f}MB")
                except Exception as e:
                    logger.error(f"Ошибка загрузки модели {key}: {e}")
                    raise
            return self._models[key]

    def unload_model(self, logger, key: str) -> bool:
        with self._load_lock:
            if key in self._models:
                m = self._models.pop(key)
                if hasattr(m, 'cpu'):
                    m.cpu()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                self._model_info.pop(key, None)
                logger.info(f"Модель {key} выгружена")
                return True
            return False

    def clear_all(self, logger) -> None:
        """Полное удаление всех кэшированных моделей и метаданных"""
        with self._load_lock:
            for key, m in list(self._models.items()):
                if hasattr(m, "cpu"):
                    try:
                        m.cpu()
                    except Exception:
                        pass
                if hasattr(m, "close"):
                    try:
                        m.close()
                    except Exception:
                        pass
            self._models.clear()
            self._model_info.clear()
            if hasattr(torch.cuda, "empty_cache"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            logger.info("Весь кэш моделей полностью очищен (clear_all)")

    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        with self._load_lock:
            return self._model_info.copy()

    def get_memory_usage(self) -> Dict[str, Any]:
        with self._load_lock:
            total_mem = sum(info["memory_usage"] for info in self._model_info.values())
            return {
                "total_models": len(self._model_info),
                "total_memory_mb": total_mem,
                "models": self._model_info.copy(),
            }


    def _estimate_model_memory(self, model: Any) -> float:
        try:
            params = sum(p.numel() for p in model.parameters())
            return params * 4 / (1024**2)
        except Exception:
            return 0.0

# Глобальный экземпляр
models_registry = ModelRegistry()
