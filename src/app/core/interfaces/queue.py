# -*- coding: utf-8 -*-
"""
Абстрактный контракт очереди задач CallAnnotate
"""
from __future__ import annotations
import abc
from typing import Awaitable, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class TaskResultProtocol(Protocol):
    task_id: str
    status: str
    message: str
    progress: int
    result: Optional[Dict]


class QueueService(abc.ABC):
    """Контракт сервиса очереди."""

    @abc.abstractmethod
    async def add_task(self, job_id: str, metadata: Dict) -> bool: ...

    @abc.abstractmethod
    async def cancel_task(self, job_id: str) -> bool: ...

    @abc.abstractmethod
    async def get_task_result(self, job_id: str) -> Optional[TaskResultProtocol]: ...

    @abc.abstractmethod
    async def get_queue_info(self) -> Dict: ...

    @abc.abstractmethod
    async def start(self) -> Awaitable[None]: ...

    @abc.abstractmethod
    async def stop(self) -> Awaitable[None]: ...

    @abc.abstractmethod
    async def subscribe_to_task(self, job_id: str, client_id: str): ...
