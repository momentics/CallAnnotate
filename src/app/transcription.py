# -*- coding: utf-8 -*-
"""
Менеджер очереди задач для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import logging
from typing import Any, Dict


class TranscriptionService:
    """
    Выполняет транскрипцию с таймкодами.
    Возвращает:
    [
      {"start": 0.0, "end": 1.2, "text": "Hello"},
      {"start": 1.5, "end": 3.5, "text": "world"},
      ...
    ]
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
