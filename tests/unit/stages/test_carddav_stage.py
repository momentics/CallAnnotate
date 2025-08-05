# tests/unit/stages/test_carddav_stage.py

import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.stages.carddav_stage import CardDAVStage
from app.schemas import ContactCreate, ContactInfo

@pytest.fixture(autouse=True)
def disable_setup_logging(monkeypatch):
    # Disable setup_logging to avoid filesystem interactions
    monkeypatch.setattr("app.utils.setup_logging", lambda cfg: None)

class TestCardDAVStage:
    @pytest.fixture
    def config(self):
        return {
            "enabled": True,
            "url": "https://carddav.example.com/contacts/",
            "username": "user",
            "password": "pass",
            "timeout": 30,
            "verify_ssl": True
        }

    @pytest.fixture
    def app_config(self, tmp_path):
        cfg = Mock()
        # Provide a valid string path for volume_path
        cfg.queue = Mock()
        cfg.queue.volume_path = str(tmp_path / "volume")
        return cfg

    @pytest.fixture
    def stage(self, app_config, config):
        return CardDAVStage(app_config, config, None)

    @pytest.mark.asyncio
    async def test_initialize_creates_httpx_client(self, stage):
        """Проверяет создание HTTP клиента при инициализации"""
        with patch('app.stages.carddav_stage.httpx.AsyncClient') as mock_client:
            await stage._initialize()
            assert stage.client is not None
            mock_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_contacts_success(self, stage):
        """Проверяет получение списка контактов"""
        mock_response = Mock()
        mock_response.status_code = 207
        mock_response.text = '''<?xml version="1.0" encoding="UTF-8"?>
            <d:multistatus xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:carddav">
                <d:response>
                    <d:href>/contacts/contact1.vcf</d:href>
                    <c:address-data>BEGIN:VCARD
VERSION:3.0
FN:John Doe
TEL:+1234567890
EMAIL:john@example.com
END:VCARD</c:address-data>
                </d:response>
            </d:multistatus>'''

        stage.client = Mock()
        stage.client.request = AsyncMock(return_value=mock_response)
        stage.enabled = True
        stage.url = "https://example.com/contacts/"

        contacts = await stage.list_contacts()

        assert len(contacts) == 1
        assert contacts[0].full_name == "John Doe"
        assert "+1234567890" in contacts[0].phones
        assert "john@example.com" in contacts[0].emails

    @pytest.mark.asyncio
    async def test_create_contact_success(self, stage):
        """Проверяет создание нового контакта"""
        contact_data = ContactCreate(
            full_name="Jane Smith",
            phones=["+9876543210"],
            emails=["jane@example.com"]
        )

        # Мок успешного создания
        mock_put_response = Mock()
        mock_put_response.status_code = 201

        # Мок получения созданного контакта
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.text = '''BEGIN:VCARD
VERSION:3.0
FN:Jane Smith
TEL:+9876543210
EMAIL:jane@example.com
END:VCARD'''

        stage.client = Mock()
        stage.client.put = AsyncMock(return_value=mock_put_response)
        stage.client.get = AsyncMock(return_value=mock_get_response)
        stage.enabled = True
        stage.url = "https://example.com/contacts/"

        created_contact = await stage.create_contact(contact_data)

        assert created_contact is not None
        assert created_contact.full_name == "Jane Smith"

    @pytest.mark.asyncio
    async def test_search_contact_by_phone(self, stage):
        """Проверяет поиск контакта по телефону"""
        # Мок списка контактов
        contacts = [
            ContactInfo(uid="1", full_name="John Doe", phones=["+1234567890"], emails=[], organization=None),
            ContactInfo(uid="2", full_name="Jane Smith", phones=["+9876543210"], emails=[], organization=None)
        ]

        stage.list_contacts = AsyncMock(return_value=contacts)

        results = await stage.search_contact(phone="+123456")

        assert len(results) == 1
        assert results[0].full_name == "John Doe"

