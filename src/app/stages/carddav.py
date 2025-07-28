# -*- coding: utf-8 -*-
"""
Этап связывания с CardDAV для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import re
import asyncio
from typing import Dict, Any, Optional, Callable
import requests
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET

from .base import BaseStage


class CardDAVStage(BaseStage):
    """Этап связывания распознанных голосов с контактами через CardDAV"""
    
    @property
    def stage_name(self) -> str:
        return "carddav"
    
    async def _initialize(self):
        """Инициализация CardDAV клиента"""
        self.server_url = self.config.get("url")
        self.username = self.config.get("username")
        self.password = self.config.get("password")
        self.timeout = self.config.get("timeout", 30)
        
        if not all([self.server_url, self.username, self.password]):
            self.logger.warning("CardDAV конфигурация неполная, этап будет пропущен")
            self.enabled = False
        else:
            self.enabled = self.config.get("enabled", True)
            self.auth = HTTPBasicAuth(self.username, self.password)
            self.logger.info("CardDAV клиент инициализирован")
    
    async def _process_impl(
        self, 
        file_path: str, 
        task_id: str, 
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Выполнение связывания с CardDAV"""
        
        if not self.enabled:
            return {"speakers": {}, "contacts_found": 0}
        
        if progress_callback:
            await progress_callback(10, "Начало поиска контактов")
        
        # Получение результатов распознавания
        recognition_results = previous_results.get("speakers", {})
        if not recognition_results:
            return {"speakers": {}, "contacts_found": 0}
        
        enhanced_speakers = {}
        contacts_found = 0
        total_speakers = len(recognition_results)
        processed = 0
        
        for speaker, recognition_data in recognition_results.items():
            if progress_callback:
                progress = 10 + int((processed / total_speakers) * 80)
                await progress_callback(progress, f"Поиск контакта для {speaker}")
            
            enhanced_data = recognition_data.copy()
            
            # Поиск контакта для идентифицированного спикера
            if recognition_data.get("identified") and recognition_data.get("name"):
                contact_info = await self._search_contact_by_name(recognition_data["name"])
                if contact_info:
                    enhanced_data["contact"] = contact_info
                    contacts_found += 1
            
            enhanced_speakers[speaker] = enhanced_data
            processed += 1
        
        if progress_callback:
            await progress_callback(100, "Поиск контактов завершен")
        
        return {
            "speakers": enhanced_speakers,
            "contacts_found": contacts_found,
            "total_searched": total_speakers
        }
    
    async def _search_contact_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Поиск контакта по имени в CardDAV"""
        try:
            # Формирование PROPFIND запроса для поиска контактов
            propfind_body = '''<?xml version="1.0" encoding="utf-8"?>
            <D:propfind xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:carddav">
                <D:prop>
                    <D:getetag/>
                    <C:address-data/>
                </D:prop>
            </D:propfind>'''
            
            headers = {
                'Content-Type': 'application/xml; charset=utf-8',
                'Depth': '1'
            }
            
            # Выполнение запроса асинхронно
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None, 
                lambda: requests.request(
                    'PROPFIND',
                    self.server_url,
                    auth=self.auth,
                    headers=headers,
                    data=propfind_body,
                    timeout=self.timeout
                )
            )
            
            if response.status_code != 207:  # Multi-Status
                self.logger.warning(f"CardDAV PROPFIND error: {response.status_code}")
                return None
            
            # Парсинг XML ответа
            root = ET.fromstring(response.text)
            
            # Поиск контактов, содержащих имя
            for address_data_elem in root.findall('.//address-data'):
                if address_data_elem.text:
                    contact_info = self._parse_vcard(address_data_elem.text)
                    if contact_info and self._name_matches(name, contact_info):
                        return contact_info

            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска контакта {name}: {e}")
            return None
    
    def _parse_vcard(self, vcard_data: str) -> Optional[Dict[str, Any]]:
        """Парсинг vCard данных"""
        try:
            contact_info = {
                "full_name": None,
                "first_name": None,
                "last_name": None,
                "phones": [],
                "emails": [],
                "organization": None
            }
            
            lines = vcard_data.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('FN:'):
                    contact_info["full_name"] = line[3:].strip()
                elif line.startswith('N:'):
                    # Формат: Фамилия;Имя;Отчество;Префикс;Суффикс
                    name_parts = line[2:].split(';')
                    if len(name_parts) >= 2:
                        contact_info["last_name"] = name_parts[0].strip()
                        contact_info["first_name"] = name_parts[1].strip()
                elif line.startswith('TEL:') or 'TEL;' in line:
                    phone_match = re.search(r':([\+\d\-\(\)\s]+)', line)
                    if phone_match:
                        phone = self._normalize_phone(phone_match.group(1))
                        if phone:
                            contact_info["phones"].append(phone)
                elif line.startswith('EMAIL:') or 'EMAIL;' in line:
                    email_match = re.search(r':([^:]+@[^:]+)', line)
                    if email_match:
                        contact_info["emails"].append(email_match.group(1).strip())
                elif line.startswith('ORG:'):
                    contact_info["organization"] = line[4:].strip()
            
            # Проверяем, что контакт содержит минимальную информацию
            if contact_info["full_name"] or (contact_info["first_name"] and contact_info["last_name"]):
                return contact_info
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка парсинга vCard: {e}")
            return None
    
    def _normalize_phone(self, phone: str) -> Optional[str]:
        """Нормализация номера телефона"""
        # Удаление всех символов кроме цифр и +
        phone = re.sub(r'[^\d\+]', '', phone)
        
        # Минимальная длина номера
        if len(phone) < 7:
            return None
        
        return phone
    
    def _name_matches(self, search_name: str, contact_info: Dict[str, Any]) -> bool:
        """Проверка соответствия имени"""
        search_name_lower = search_name.lower().strip()
        
        # Проверка полного имени
        if contact_info.get("full_name"):
            if search_name_lower in contact_info["full_name"].lower():
                return True
        
        # Проверка отдельных частей имени
        first_name = contact_info.get("first_name", "").lower()
        last_name = contact_info.get("last_name", "").lower()
        
        if search_name_lower in first_name or search_name_lower in last_name:
            return True
        
        # Проверка на точное совпадение частей
        search_parts = search_name_lower.split()
        name_parts = [first_name, last_name]
        
        for search_part in search_parts:
            if len(search_part) > 2:  # Игнорируем слишком короткие части
                for name_part in name_parts:
                    if search_part in name_part or name_part in search_part:
                        return True
        
        return False
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Информация о CardDAV соединении"""
        return {
            "stage": self.stage_name,
            "server_url": self.server_url if self.enabled else None,
            "enabled": self.enabled,
            "framework": "CardDAV/vCard"
        }
