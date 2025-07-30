# tests/test_stages_carddav.py
# -*- coding: utf-8 -*-
"""
Unit-тест для CardDAVStage

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.stages.carddav_stage import CardDAVStage
from app.schemas import ContactCreate, ContactInfo, ContactUpdate


@pytest.mark.asyncio
async def test_carddav_stage_disabled():
    """Тест CardDAV стадии в выключенном состоянии"""
    config = {"enabled": False}
    stage = CardDAVStage(config, models_registry=None)
    
    await stage._initialize()
    
    # Проверяем что клиент не инициализирован
    assert stage.client is None
    assert not stage.enabled


@pytest.mark.asyncio
async def test_carddav_stage_enabled_no_credentials():
    """Тест CardDAV стадии с включенным состоянием но без учетных данных"""
    config = {
        "enabled": True,
        "url": None,
        "username": None,
        "password": None,
        "timeout": 30,
        "verify_ssl": True
    }
    stage = CardDAVStage(config, models_registry=None)
    
    await stage._initialize()
    
    # Проверяем что клиент не инициализирован из-за отсутствия URL
    assert stage.client is None
    assert stage.enabled


@pytest.mark.asyncio
async def test_carddav_stage_initialization_with_credentials():
    """Тест инициализации CardDAV стадии с учетными данными"""
    config = {
        "enabled": True,
        "url": "https://example.com/carddav",
        "username": "testuser",
        "password": "testpass",
        "timeout": 30,
        "verify_ssl": True
    }
    
    with patch('app.stages.carddav_stage.httpx.AsyncClient') as mock_client:
        stage = CardDAVStage(config, models_registry=None)
        await stage._initialize()
        
        # Проверяем что клиент создан с правильными параметрами
        mock_client.assert_called_once_with(
            timeout=30,
            verify=True,
            auth=("testuser", "testpass"),
            headers={"User-Agent": "CallAnnotate-CardDAV/1.0"}
        )
        assert stage.enabled
        assert stage.url == "https://example.com/carddav"


@pytest.mark.asyncio
async def test_list_contacts_disabled():
    """Тест получения списка контактов при выключенной стадии"""
    config = {"enabled": False}
    stage = CardDAVStage(config, models_registry=None)
    await stage._initialize()
    
    contacts = await stage.list_contacts()
    assert contacts == []


@pytest.mark.asyncio
async def test_list_contacts_success():
    """Тест успешного получения списка контактов"""
    config = {
        "enabled": True,
        "url": "https://example.com/carddav",
        "username": "user",
        "password": "pass"
    }
    
    # Мокаем XML ответ CardDAV сервера
    mock_xml_response = """<?xml version="1.0" encoding="UTF-8"?>
    <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">
        <d:response>
            <d:href>/carddav/contact1.vcf</d:href>
            <c:address-data>BEGIN:VCARD
VERSION:3.0
FN:John Doe
N:Doe;John;;;
TEL:+1234567890
EMAIL:john@example.com
END:VCARD</c:address-data>
        </d:response>
    </d:multistatus>"""
    
    mock_response = MagicMock()
    mock_response.status_code = 207
    mock_response.text = mock_xml_response
    
    mock_client = AsyncMock()
    mock_client.request.return_value = mock_response
    
    with patch('app.stages.carddav_stage.httpx.AsyncClient') as mock_client_class:
        mock_client_class.return_value = mock_client
        
        stage = CardDAVStage(config, models_registry=None)
        await stage._initialize()
        
        contacts = await stage.list_contacts()
        
        # Проверяем что запрос был выполнен
        mock_client.request.assert_called_once_with(
            "PROPFIND", "https://example.com/carddav", headers={"Depth": "1"}
        )
        
        # Проверяем результат
        assert len(contacts) == 1
        assert contacts[0].uid == "contact1"
        assert contacts[0].full_name == "John Doe"
        assert "+1234567890" in contacts[0].phones
        assert "john@example.com" in contacts[0].emails


@pytest.mark.asyncio
async def test_search_contact_by_name():
    """Тест поиска контакта по имени"""
    config = {"enabled": True, "url": "https://example.com/carddav"}
    
    # Создаем мок контактов
    contact1 = ContactInfo(uid="1", full_name="John Doe", phones=[], emails=[])
    contact2 = ContactInfo(uid="2", full_name="Jane Smith", phones=[], emails=[])
    
    stage = CardDAVStage(config, models_registry=None)
    stage.enabled = True
    stage.client = AsyncMock()
    
    # Мокаем list_contacts для возврата тестовых контactов
    with patch.object(stage, 'list_contacts', return_value=[contact1, contact2]):
        results = await stage.search_contact(name="john")
        
        assert len(results) == 1
        assert results[0].full_name == "John Doe"


@pytest.mark.asyncio
async def test_search_contact_by_phone():
    """Тест поиска контакта по телефону"""
    config = {"enabled": True, "url": "https://example.com/carddav"}
    
    contact1 = ContactInfo(uid="1", full_name="John Doe", phones=["+1234567890"], emails=[])
    contact2 = ContactInfo(uid="2", full_name="Jane Smith", phones=["+0987654321"], emails=[])
    
    stage = CardDAVStage(config, models_registry=None)
    stage.enabled = True
    stage.client = AsyncMock()
    
    with patch.object(stage, 'list_contacts', return_value=[contact1, contact2]):
        results = await stage.search_contact(phone="123456")
        
        assert len(results) == 1
        assert results[0].full_name == "John Doe"


@pytest.mark.asyncio
async def test_create_contact_disabled():
    """Тест создания контакта при выключенной стадии"""
    config = {"enabled": False}
    stage = CardDAVStage(config, models_registry=None)
    await stage._initialize()
    
    contact_data = ContactCreate(full_name="Test User", phones=["+1234567890"])
    result = await stage.create_contact(contact_data)
    
    assert result is None


@pytest.mark.asyncio
async def test_create_contact_success():
    """Тест успешного создания контакта"""
    config = {
        "enabled": True,
        "url": "https://example.com/carddav",
        "username": "user",
        "password": "pass"
    }
    
    contact_data = ContactCreate(
        full_name="Test User",
        phones=["+1234567890"],
        emails=["test@example.com"]
    )
    
    mock_client = AsyncMock()
    mock_client.put.return_value.status_code = 201
    
    # Мокаем get_contact для возврата созданного контакта
    expected_contact = ContactInfo(
        uid="Test_User",
        full_name="Test User",
        phones=["+1234567890"],
        emails=["test@example.com"]
    )
    
    with patch('app.stages.carddav_stage.httpx.AsyncClient') as mock_client_class:
        mock_client_class.return_value = mock_client
        
        stage = CardDAVStage(config, models_registry=None)
        await stage._initialize()
        
        with patch.object(stage, 'get_contact', return_value=expected_contact):
            result = await stage.create_contact(contact_data)
            
            # Проверяем что PUT запрос был выполнен
            mock_client.put.assert_called_once()
            
            # Проверяем результат
            assert result is not None
            assert result.full_name == "Test User"
            assert "+1234567890" in result.phones


@pytest.mark.asyncio
async def test_process_impl_disabled():
    """Тест _process_impl при выключенной стадии"""
    config = {"enabled": False}
    stage = CardDAVStage(config, models_registry=None)
    await stage._initialize()
    
    result = await stage._process_impl(
        "test.wav", 
        "job123", 
        {"speakers": {"spk1": {"name": "John Doe"}}},
        progress_callback=None
    )
    
    expected = {"speakers": {}, "contacts_found": 0}
    assert result == expected


@pytest.mark.asyncio
async def test_process_impl_with_speakers():
    """Тест _process_impl с спикерами"""
    config = {
        "enabled": True,
        "url": "https://example.com/carddav",
        "username": "user",
        "password": "pass"
    }
    
    # Создаем мок контакт
    mock_contact = ContactInfo(
        uid="1",
        full_name="John Doe",
        phones=["+1234567890"],
        emails=["john@example.com"]
    )
    
    stage = CardDAVStage(config, models_registry=None)
    stage.enabled = True
    stage.client = AsyncMock()
    
    # Мокаем list_contacts
    with patch.object(stage, 'list_contacts', return_value=[mock_contact]):
        result = await stage._process_impl(
            "test.wav",
            "job123", 
            {"speakers": {"spk1": {"name": "John"}}},
            progress_callback=None
        )
        
        assert "speakers" in result
        assert "contacts_found" in result
        assert result["contacts_found"] == 1
        assert result["speakers"]["spk1"]["contact"] is not None
        assert result["speakers"]["spk1"]["contact"]["full_name"] == "John Doe"


@pytest.mark.asyncio
async def test_vcard_parsing():
    """Тест парсинга vCard"""
    config = {"enabled": True}
    stage = CardDAVStage(config, models_registry=None)
    
    vcard_data = """BEGIN:VCARD
VERSION:3.0
FN:John Doe
N:Doe;John;;;
TEL:+1234567890
EMAIL:john@example.com
ORG:Example Corp
END:VCARD"""
    
    parsed = stage._parse_vcard(vcard_data)
    
    assert parsed.full_name == "John Doe"
    assert parsed.last_name == "Doe"
    assert parsed.first_name == "John"
    assert "+1234567890" in parsed.phones
    assert "john@example.com" in parsed.emails
    assert parsed.organization == "Example Corp"


@pytest.mark.asyncio
async def test_vcard_generation():
    """Тест генерации vCard"""
    config = {"enabled": True}
    stage = CardDAVStage(config, models_registry=None)
    
    contact_data = ContactCreate(
        full_name="Jane Smith",
        phones=["+0987654321", "+1111111111"],
        emails=["jane@example.com"],
        organization="Test Corp"
    )
    
    vcard = stage._to_vcard(contact_data)
    
    assert "FN:Jane Smith" in vcard
    assert "TEL:+0987654321" in vcard
    assert "TEL:+1111111111" in vcard
    assert "EMAIL:jane@example.com" in vcard
    assert "ORG:Test Corp" in vcard
    assert "BEGIN:VCARD" in vcard
    assert "END:VCARD" in vcard
