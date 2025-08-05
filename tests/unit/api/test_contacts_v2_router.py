# tests/unit/api/test_contacts_v2_router.py

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from app.config import load_settings
from app.stages.carddav_stage import CardDAVStage
from app.schemas import ContactCreate, ContactInfo
from app.api.routers.contacts_v2 import router, ContactFilter, list_contacts, get_contact, create_contact, update_contact, delete_contact

import pytest
from typing import List, Optional
from pydantic import BaseModel, Field

client = TestClient(router)


class ContactFilterModel(BaseModel):
    name: Optional[str] = Field(None, description="Поиск по полному или частичному имени")
    phone: Optional[str] = Field(None, description="Поиск по номеру телефона")
    email: Optional[str] = Field(None, description="Поиск по email")


@pytest.fixture
async def carddav_stage():
    settings = load_settings()
    cfg = settings.carddav.dict()
    stage = CardDAVStage(settings, cfg)
    await stage._initialize()
    return stage


def test_router_prefix_and_tags():
    assert router.prefix == "/api/v1/contacts"
    assert "Contacts" in router.tags


@pytest.mark.asyncio
async def test_list_contacts_all(monkeypatch, carddav_stage):
    contacts = [ContactInfo(uid="1", full_name="John Doe", phones=["+123"], emails=[], organization=None)]
    monkeypatch.setattr(carddav_stage, "list_contacts", lambda: contacts)
    response = await list_contacts(name=None, phone=None, email=None, stage=carddav_stage)
    assert response == contacts


@pytest.mark.asyncio
async def test_list_contacts_filtered(monkeypatch, carddav_stage):
    contacts = [
        ContactInfo(uid="1", full_name="John Doe", phones=["+123"], emails=[], organization=None),
        ContactInfo(uid="2", full_name="Jane Smith", phones=["+987"], emails=[], organization=None)
    ]
    monkeypatch.setattr(carddav_stage, "search_contact", lambda name, phone, email: [contacts[0]])
    response = await list_contacts(name="John", phone=None, email=None, stage=carddav_stage)
    assert response == [contacts[0]]


@pytest.mark.asyncio
async def test_get_contact_found(monkeypatch, carddav_stage):
    contact = ContactInfo(uid="1", full_name="John Doe", phones=["+123"], emails=[], organization=None)
    monkeypatch.setattr(carddav_stage, "get_contact", lambda uid: contact)
    result = await get_contact(uid="1", stage=carddav_stage)
    assert result == contact


@pytest.mark.asyncio
async def test_get_contact_not_found(monkeypatch, carddav_stage):
    monkeypatch.setattr(carddav_stage, "get_contact", lambda uid: None)
    with pytest.raises(HTTPException) as excinfo:
        await get_contact(uid="unknown", stage=carddav_stage)
    assert excinfo.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_create_contact_success(monkeypatch, carddav_stage):
    data = ContactCreate(full_name="Jane Smith", phones=["+987"], emails=["jane@example.com"], organization=None)
    created = ContactInfo(uid="2", full_name="Jane Smith", phones=["+987"], emails=["jane@example.com"], organization=None)
    monkeypatch.setattr(carddav_stage, "create_contact", lambda d: created)
    result = await create_contact(data=data, stage=carddav_stage)
    assert result == created


@pytest.mark.asyncio
async def test_create_contact_failure(monkeypatch, carddav_stage):
    data = ContactCreate(full_name="Jane Smith", phones=["+987"], emails=["jane@example.com"], organization=None)
    monkeypatch.setattr(carddav_stage, "create_contact", lambda d: None)
    with pytest.raises(HTTPException) as excinfo:
        await create_contact(data=data, stage=carddav_stage)
    assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_update_contact_success(monkeypatch, carddav_stage):
    data = ContactCreate(full_name="John Updated", phones=["+123"], emails=["john@example.com"], organization=None)
    updated = ContactInfo(uid="1", full_name="John Updated", phones=["+123"], emails=["john@example.com"], organization=None)
    monkeypatch.setattr(carddav_stage, "update_contact", lambda uid, d: updated)
    result = await update_contact(uid="1", data=data, stage=carddav_stage)
    assert result == updated


@pytest.mark.asyncio
async def test_update_contact_not_found(monkeypatch, carddav_stage):
    data = ContactCreate(full_name="John Updated", phones=["+123"], emails=["john@example.com"], organization=None)
    monkeypatch.setattr(carddav_stage, "update_contact", lambda uid, d: None)
    with pytest.raises(HTTPException) as excinfo:
        await update_contact(uid="unknown", data=data, stage=carddav_stage)
    assert excinfo.value.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_delete_contact_success(monkeypatch, carddav_stage):
    monkeypatch.setattr(carddav_stage, "delete_contact", lambda uid: True)
    result = await delete_contact(uid="1", stage=carddav_stage)
    assert result is None  # No content


@pytest.mark.asyncio
async def test_delete_contact_not_found(monkeypatch, carddav_stage):
    monkeypatch.setattr(carddav_stage, "delete_contact", lambda uid: False)
    with pytest.raises(HTTPException) as excinfo:
        await delete_contact(uid="unknown", stage=carddav_stage)
    assert excinfo.value.status_code == status.HTTP_404_NOT_FOUND
