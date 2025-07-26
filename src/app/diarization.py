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
    """Сервис диаризации"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
