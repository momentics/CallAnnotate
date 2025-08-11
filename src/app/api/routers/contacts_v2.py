# src/app/api/routers/contacts_v2.py
# -*- coding: utf-8 -*-
"""
REST-роутер управления контактами через CardDAV для CallAnnotate.
Поддерживает CRUD-операции и фильтрацию по имени, телефону, email.
Автор: akoodoy@capilot.ru
Лицензия: Apache-2.0
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import inspect

from ...config import load_settings
from ...stages.carddav_stage import CardDAVStage
from ...schemas import ContactCreate, ContactInfo, ContactUpdate

router = APIRouter(prefix="/api/v1/contacts", tags=["Contacts"])

class ContactFilter(BaseModel):
    name: Optional[str] = Field(None, description="Поиск по полному или частичному имени")
    phone: Optional[str] = Field(None, description="Поиск по номеру телефона")
    email: Optional[str] = Field(None, description="Поиск по email")

async def get_carddav_stage() -> CardDAVStage:
    cfg = load_settings()
    config = load_settings().carddav.dict()

    stage = CardDAVStage(cfg, config)
    await stage._initialize()
    return stage

@router.get("/", response_model=List[ContactInfo])
async def list_contacts(
    name: Optional[str] = Query(None),
    phone: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    stage: CardDAVStage = Depends(get_carddav_stage),
):
    """
    Список контактов. По умолчанию возвращает все, при указании параметров
    фильтрует по имени, телефону или email.
    """
    if any((name, phone, email)):
        result = stage.search_contact(name=name, phone=phone, email=email) # type: ignore
    else:
        result = stage.list_contacts()
    if inspect.isawaitable(result):
        result = await result
    return result

@router.get("/{uid}", response_model=ContactInfo)
async def get_contact(
    uid: str,
    stage: CardDAVStage = Depends(get_carddav_stage),
):
    """
    Получение контакта по UID.
    """
    result = stage.get_contact(uid)
    if inspect.isawaitable(result):
        result = await result
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Контакт не найден")
    return result

@router.post("/", response_model=ContactInfo, status_code=status.HTTP_201_CREATED)
async def create_contact(
    data: ContactCreate,
    stage: CardDAVStage = Depends(get_carddav_stage),
):
    """
    Создание нового контакта.
    """
    result = stage.create_contact(data)
    if inspect.isawaitable(result):
        result = await result
    if not result:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Не удалось создать контакт"
        )
    return result

@router.put("/{uid}", response_model=ContactInfo)
async def update_contact(
    uid: str,
    data: ContactUpdate,
    stage: CardDAVStage = Depends(get_carddav_stage),
):
    """
    Обновление существующего контакта по UID.
    """
    result = stage.update_contact(uid, data)
    if inspect.isawaitable(result):
        result = await result
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Контакт не найден")
    return result

@router.delete("/{uid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_contact(
    uid: str,
    stage: CardDAVStage = Depends(get_carddav_stage),
):
    """
    Удаление контакта по UID.
    """
    result = stage.delete_contact(uid)
    if inspect.isawaitable(result):
        result = await result
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Контакт не найден")
    return
