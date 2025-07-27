# -*- coding: utf-8 -*-
"""
Менеджер очереди задач для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import logging
from typing import Any, Dict

class DiarizationService:
    """
    Выполняет сегментацию говорящих.
    Возвращает список сегментов:
    [
      {"speaker_id": 0, "start": 0.0, "end": 5.0},
      {"speaker_id": 1, "start": 5.0, "end": 10.0},
      ...
    ]
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

