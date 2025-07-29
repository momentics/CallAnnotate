# src/app/stages/base.py

# -*- coding: utf-8 -*-
"""
Базовый класс для всех этапов обработки аудио

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

@dataclass
class StageResult:
    """Результат выполнения этапа обработки"""
    stage_name: str
    processing_time: float
    model_info: Dict[str, Any]
    payload: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None

class BaseStage(ABC):
    """Базовый абстрактный класс для всех этапов обработки"""

    def __init__(self, config: Dict[str, Any], models_registry=None):
        self.config = config
        self.models_registry = models_registry
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Название этапа для логирования и отслеживания"""
        pass

    @abstractmethod
    async def _initialize(self):
        """Инициализация моделей и ресурсов (вызывается один раз)"""
        pass

    @abstractmethod
    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Основная логика обработки (должна быть реализована в подклассах)"""
        pass

    async def process(
        self,
        file_path: str,
        task_id: str,
        previous_results: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> StageResult:
        start_time = time.perf_counter()
        try:
            if not self._initialized:
                self.logger.info(f"Инициализация этапа {self.stage_name}")
                await self._initialize()
                self._initialized = True

            payload = await self._process_impl(
                file_path, task_id, previous_results or {}, progress_callback
            )
            processing_time = time.perf_counter() - start_time
            return StageResult(
                stage_name=self.stage_name,
                processing_time=processing_time,
                model_info=self._get_model_info(),
                payload=payload,
                success=True
            )
        except Exception as e:
            processing_time = time.perf_counter() - start_time
            error_msg = f"Ошибка в этапе {self.stage_name}: {e}"
            self.logger.error(error_msg, exc_info=True)
            return StageResult(
                stage_name=self.stage_name,
                processing_time=processing_time,
                model_info=self._get_model_info(),
                payload={},
                success=False,
                error=error_msg
            )

    def _get_model_info(self) -> Dict[str, Any]:
        return {
            "stage": self.stage_name,
            "config": self.config
        }

    async def cleanup(self):
        """Очистка ресурсов (может быть переопределено в подклассах)"""
        pass
