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
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass

from ..config import AppSettings
from ..utils import setup_logging

@dataclass
class StageResult:
    stage_name: str
    processing_time: float
    model_info: Dict[str, Any]
    payload: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None

class BaseStage(ABC):
    def __init__(self, cfg: AppSettings, config: Dict[str, Any], models_registry=None):
        setup_logging(cfg)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.volume_path = cfg.queue.volume_path

        self.config = config
        self.models_registry = models_registry
        self._initialized = False

    @property
    @abstractmethod
    def stage_name(self) -> str:
        pass

    @abstractmethod
    async def _initialize(self):
        pass

    @abstractmethod
    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        pass

    async def process(self, file_path, task_id, previous_results=None, progress_callback=None):
        start = time.perf_counter()
        if not self._initialized:
            await self._initialize()
            self._initialized = True
        try:
            payload = await self._process_impl(file_path, task_id, previous_results or {}, progress_callback)
            duration = time.perf_counter() - start
            return StageResult(self.stage_name, duration, self._get_model_info(), payload, True, None)
        except Exception as e:
            duration = time.perf_counter() - start
            msg = str(e)
            self.logger.error(f"Error in stage {self.stage_name}: {msg}", exc_info=True)
            return StageResult(self.stage_name, duration, self._get_model_info(), {}, False, msg)

    def _get_model_info(self) -> Dict[str, Any]:
        return {"stage": self.stage_name, "config": self.config}
