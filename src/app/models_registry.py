# -*- coding: utf-8 -*-
"""
Реестр моделей для CallAnnotate с ленивой загрузкой и кешированием

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import logging
import threading
from typing import Dict, Any, Callable
import torch


class ModelRegistry:
    """Реестр для ленивой загрузки и кеширования ML-моделей"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ModelRegistry':
        """Singleton паттерн для глобального реестра"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.logger = logging.getLogger(__name__)
            self._models: Dict[str, Any] = {}
            self._model_info: Dict[str, Dict[str, Any]] = {}
            self._load_lock = threading.RLock()
            self._initialized = True
            
            self.logger.info("ModelRegistry инициализирован")
    
    def get_model(self, key: str, loader: Callable[[], Any], **metadata) -> Any:
        """
        Получение модели с ленивой загрузкой
        
        Args:
            key: Уникальный ключ модели
            loader: Функция загрузки модели
            **metadata: Дополнительные метаданные модели
            
        Returns:
            Загруженная модель
        """
        with self._load_lock:
            if key not in self._models:
                self.logger.info(f"Загрузка модели: {key}")
                
                try:
                    model = loader()
                    self._models[key] = model
                    self._model_info[key] = {
                        "memory_usage": self._estimate_model_memory(model),
                        **metadata
                    }
                    
                    self.logger.info(f"Модель {key} загружена успешно")
                    
                except Exception as e:
                    self.logger.error(f"Ошибка загрузки модели {key}: {e}")
                    raise
            else:
                self.logger.debug(f"Модель {key} получена из кеша")
            
            return self._models[key]
    
    def unload_model(self, key: str) -> bool:
        """
        Выгрузка модели из памяти
        
        Args:
            key: Ключ модели
            
        Returns:
            True если модель была выгружена
        """
        with self._load_lock:
            if key in self._models:
                model = self._models[key]
                
                # Освобождение GPU памяти для PyTorch моделей
                if hasattr(model, 'cpu'):
                    model.cpu()
                
                if hasattr(model, 'to') and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                del self._models[key]
                if key in self._model_info:
                    del self._model_info[key]
                
                self.logger.info(f"Модель {key} выгружена")
                return True
            
            return False
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Получение информации о загруженных моделях"""
        with self._load_lock:
            return self._model_info.copy()
    
    def clear_all(self):
        """Очистка всех моделей"""
        with self._load_lock:
            keys_to_clear = list(self._models.keys())
            for key in keys_to_clear:
                self.unload_model(key)
            
            self.logger.info("Все модели выгружены")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Получение информации об использовании памяти"""
        memory_info = {
            "total_models": len(self._models),
            "models": {}
        }
        
        total_memory = 0
        
        for key, info in self._model_info.items():
            model_memory = info.get("memory_usage", 0)
            memory_info["models"][key] = {
                "memory_mb": model_memory
            }
            total_memory += model_memory
        
        memory_info["total_memory_mb"] = total_memory
        
        # Информация о системной памяти
        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "cached_mb": torch.cuda.memory_reserved() / 1024 / 1024
            }
        
        return memory_info
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Оценка потребления памяти моделью в MB"""
        try:
            if hasattr(model, 'parameters'):
                # PyTorch модель
                total_params = sum(p.numel() for p in model.parameters())
                # Приблизительно 4 байта на параметр (float32)
                return (total_params * 4) / (1024 * 1024)
            
            elif hasattr(model, '__sizeof__'):
                # Общий случай
                return model.__sizeof__() / (1024 * 1024)
            
            else:
                return 0.0
                
        except Exception:
            return 0.0


# Глобальный экземпляр реестра
models_registry = ModelRegistry()
