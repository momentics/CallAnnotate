# src/app/stages/carddav_stage.py
# -*- coding: utf-8 -*-
"""
CardDAV Stage для CallAnnotate: асинхронное управление контактами, гибкая фильтрация,
поиск по имени, телефону или email. Соответствует архитектуре stages, строгая обработка ошибок.
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET
import re

import httpx

from .base import BaseStage
from ..schemas import ContactCreate, ContactInfo, ContactUpdate

class CardDAVStage(BaseStage):
    @property
    def stage_name(self) -> str:
        return "carddav"

    async def _initialize(self):
        cfg = self.config
        self.enabled: bool = cfg.get("enabled", True)
        self.url: Optional[str] = cfg.get("url")
        self.username: Optional[str] = cfg.get("username")
        self.password: Optional[str] = cfg.get("password")
        self.timeout: int = int(cfg.get("timeout", 30))
        self.verify_ssl: bool = cfg.get("verify_ssl", True)
        if self.enabled and self.url:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.verify_ssl,
                auth=(self.username, self.password) if self.username and self.password else None,
                headers={"User-Agent": "CallAnnotate-CardDAV/1.0"}
            )
        else:
            self.client = None

    async def list_contacts(self) -> List[ContactInfo]:
        if not (self.enabled and self.client and self.url):
            return []
        # CardDAV multistatus запрос
        resp = await self.client.request(
            "PROPFIND", self.url, headers={"Depth": "1"}
        )
        if resp.status_code != 207:
            return []
        return self._parse_multistatus(resp.text)

    async def get_contact(self, uid: str) -> Optional[ContactInfo]:
        if not (self.enabled and self.client and self.url):
            return None
        # uid — это файл .vcf или уникальный ID
        href = f"{self.url.rstrip('/')}/{uid}.vcf" if not uid.endswith('.vcf') else uid
        resp = await self.client.get(href)
        if resp.status_code == 200 and resp.text:
            info = self._parse_vcard(resp.text)
            info.uid = uid.replace(".vcf", "")
            return info
        return None

    async def create_contact(self, data: ContactCreate) -> Optional[ContactInfo]:
        if not (self.enabled and self.client and self.url):
            return None
        vcard = self._to_vcard(data)
        # Имя создаваемого контакта = full_name без пробелов (fallback: anon)
        contact_uid = (data.full_name or "anon").strip().replace(" ", "_")
        href = f"{self.url.rstrip('/')}/{contact_uid}.vcf"
        resp = await self.client.put(
            href,
            content=vcard,
            headers={"Content-Type": "text/vcard"},
        )
        if resp.status_code not in (200, 201, 204):
            return None
        return await self.get_contact(contact_uid)

    async def update_contact(self, uid: str, data: ContactUpdate) -> Optional[ContactInfo]:
        if not (self.enabled and self.client and self.url):
            return None
        vcard = self._to_vcard(data)
        href = f"{self.url.rstrip('/')}/{uid}.vcf" if not uid.endswith('.vcf') else uid
        resp = await self.client.put(
            href,
            content=vcard,
            headers={"Content-Type": "text/vcard"},
        )
        if resp.status_code not in (200, 201, 204):
            return None
        return await self.get_contact(uid.replace(".vcf", ""))

    async def delete_contact(self, uid: str) -> bool:
        if not (self.enabled and self.client and self.url):
            return False
        href = f"{self.url.rstrip('/')}/{uid}.vcf" if not uid.endswith('.vcf') else uid
        resp = await self.client.delete(href)
        return resp.status_code in (200, 204)

    async def search_contact(
        self, *, name: str = None, phone: str = None, email: str = None
    ) -> List[ContactInfo]:
        """Гибкий поиск по имени, телефону или email."""
        contacts = await self.list_contacts()
        result = []
        for c in contacts:
            matched = False
            if name and c.full_name and name.lower() in c.full_name.lower():
                matched = True
            if phone and any(phone in p for p in c.phones):
                matched = True
            if email and any(email in e for e in c.emails):
                matched = True
            if matched:
                result.append(c)
        return result

    def _parse_multistatus(self, xml_text: str) -> List[ContactInfo]:
        # CardDAV multistatus → vCard parse
        root = ET.fromstring(xml_text.replace(";", ""))
        ns = {"D": "DAV:", "C": "urn:ietf:params:xml:ns:carddav"}
        out: List[ContactInfo] = []
        for resp in root.findall("D:response", ns):
            href = resp.findtext("D:href", default="", namespaces=ns)
            adata = resp.find(".//C:address-data", ns)
            vcard_text = adata.text if adata is not None and adata.text else None
            if not vcard_text:
                continue
            info = self._parse_vcard(vcard_text)
            info.uid = href.rstrip("/").split("/")[-1].replace(".vcf", "")
            out.append(info)
        return out

    def _parse_vcard(self, vcard: str) -> ContactInfo:
        # Простая, но строгая реализация парсинга vCard 3.0
        info = ContactInfo(uid="")
        lines = [l.strip() for l in vcard.replace("\r\n", "\n").split("\n") if l.strip()]
        for line in lines:
            if line.startswith("FN:"):
                info.full_name = line[3:].strip()
            elif line.startswith("N:"):
                parts = line[2:].split(";")
                # стандарт: N:Фамилия;Имя;Отчество;;
                info.last_name = parts[0].strip() if len(parts) > 0 else None
                info.first_name = parts[1].strip() if len(parts) > 1 else None
            elif line.startswith("TEL"):
                m = re.match(r"TEL(;.+?)?:(?P<val>.+)", line)
                if m:
                    phone = m.group("val").strip()
                    info.phones.append(phone)
            elif line.startswith("EMAIL"):
                m = re.match(r"EMAIL(;.+?)?:(?P<em>.+)", line)
                if m:
                    email = m.group("em").strip()
                    info.emails.append(email)
            elif line.startswith("ORG:"):
                info.organization = line[4:].strip()
        return info

    def _to_vcard(self, data: Union[ContactCreate, ContactUpdate]) -> str:
        # Формируем vCard 3.0
        lines = ["BEGIN:VCARD", "VERSION:3.0"]
        if hasattr(data, "full_name") and data.full_name:
            lines.append(f"FN:{data.full_name}")
        if getattr(data, "last_name", None) or getattr(data, "first_name", None):
            ln = getattr(data, "last_name", "") or ""
            fn = getattr(data, "first_name", "") or ""
            lines.append(f"N:{ln};{fn};;;")
        phones = getattr(data, "phones", []) or []
        for tel in phones:
            lines.append(f"TEL:{tel}")
        emails = getattr(data, "emails", []) or []
        for em in emails:
            lines.append(f"EMAIL:{em}")
        if getattr(data, "organization", None):
            lines.append(f"ORG:{getattr(data,'organization')}")
        lines.append("END:VCARD")
        return "\r\n".join(lines)

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Связывание известных спикеров и контактов CardDAV по имени/телефону/email"""
        # Ожидается: previous_results содержит speakers: {label: meta, ...}
        if not (self.enabled and self.client and self.url):
            return {"speakers": {}, "contacts_found": 0}
        speakers_in = previous_results.get("speakers", {})
        contacts = await self.list_contacts()
        out: Dict[str, Any] = {}
        found = 0
        for label, meta in speakers_in.items():
            match = None
            # Сначала точное совпадение по name, затем — по телефонам (если есть номер)
            name = meta.get("name")
            phone = meta.get("phone")
            email = meta.get("email")
            for c in contacts:
                if name and c.full_name and name.lower() in c.full_name.lower():
                    match = c
                    break
                if phone and phone in c.phones:
                    match = c
                    break
                if email and email in c.emails:
                    match = c
                    break
            if match:
                found += 1
            out[label] = {"contact": match.model_dump() if match else None}
        return {"speakers": out, "contacts_found": found}
