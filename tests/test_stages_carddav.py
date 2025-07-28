import asyncio
import pytest
from unittest.mock import patch, MagicMock
from app.stages.carddav import CardDAVStage

@pytest.mark.asyncio
async def test_carddav_disabled(tmp_path, monkeypatch):
    cfg = {"enabled": False}
    stage = CardDAVStage(cfg, None)
    result = await stage.process("file", "job")
    assert result.payload == {"speakers": {}, "contacts_found": 0}

@pytest.mark.asyncio
async def test_search_contact_found(monkeypatch):
    # Настраиваем успешный PROPFIND и vCard
    fake_vcard = "FN:John Doe\nTEL:+1234567\nEMAIL:john@example.com"
    response = MagicMock(status_code=207, text=f"<xml><address-data>{fake_vcard}</address-data></xml>")
    monkeypatch.setenv("CARDDAV_URL", "https://fake")
    monkeypatch.setenv("CARDDAV_USERNAME", "u")
    monkeypatch.setenv("CARDDAV_PASSWORD", "p")
    from app.stages.carddav import requests
    monkeypatch.setattr(requests, "request", lambda *args, **kw: response)

    cfg = {"enabled": True, "url": "u", "username": "u", "password": "p", "timeout": 1}
    stage = CardDAVStage(cfg, None)
    await stage._initialize()
    out = await stage._process_impl("file", "job", {"speakers": {"spk": {"identified": True, "name": "John"}}})
    assert out["contacts_found"] == 1
    assert "John" in out["speakers"]["spk"]["contact"]["full_name"]
