# src/app/stages/carddav_stage.py
# -*- coding: utf-8 -*-
"""
CardDAV Stage для CallAnnotate: асинхронное управление контактами, гибкая фильтрация,
поиск по имени, телефону или email. Соответствует архитектуре stages, строгая обработка ошибок.
Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from typing import Any, Callable, Dict, List, Optional, Union
import xml.etree.ElementTree as ET
import re

import httpx

from .base import BaseStage
from ..schemas import ContactCreate, ContactInfo, ContactUpdate

class CardDAVStage(BaseStage):
    def __init__(self, config, models_registry=None):
        super().__init__(config, models_registry)
        # Защита для тестов, когда _initialize не вызывается
        self.enabled: bool = False
        self.url: Optional[str] = None
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self.timeout: int = 30
        self.verify_ssl: bool = True
        self.client: Optional[httpx.AsyncClient] = None

    @property
    def stage_name(self) -> str:
        return "carddav"

    async def _initialize(self):
        cfg = self.config
        self.enabled = cfg.get("enabled", True)
        self.url = cfg.get("url")
        self.username = cfg.get("username")
        self.password = cfg.get("password")
        self.timeout = int(cfg.get("timeout", 30))
        self.verify_ssl = cfg.get("verify_ssl", True)
        if self.enabled and self.url:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                verify=self.verify_ssl,
                auth=(self.username, self.password) if self.username and self.password else None,
                headers={"User-Agent": "CallAnnotate-CardDAV/1.0"},
            )
        else:
            self.client = None

    async def list_contacts(self) -> List[ContactInfo]:
        if not (self.enabled and self.client and self.url):
            return []
        resp = await self.client.request("PROPFIND", self.url, headers={"Depth": "1"})
        if resp.status_code != 207:
            return []
        return self._parse_multistatus(resp.text)

    async def get_contact(self, uid: str) -> Optional[ContactInfo]:
        if not (self.enabled and self.client and self.url):
            return None
        href = f"{self.url.rstrip('/')}/{uid}.vcf" if not uid.endswith(".vcf") else uid
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
        contact_uid = (data.full_name or "anon").strip().replace(" ", "_")
        href = f"{self.url.rstrip('/')}/{contact_uid}.vcf"
        resp = await self.client.put(href, content=vcard, headers={"Content-Type": "text/vcard"})
        if resp.status_code not in (200, 201, 204):
            return None
        return await self.get_contact(contact_uid)

    async def update_contact(self, uid: str, data: ContactUpdate) -> Optional[ContactInfo]:
        if not (self.enabled and self.client and self.url):
            return None
        vcard = self._to_vcard(data)
        href = f"{self.url.rstrip('/')}/{uid}.vcf" if not uid.endswith(".vcf") else uid
        resp = await self.client.put(href, content=vcard, headers={"Content-Type": "text/vcard"})
        if resp.status_code not in (200, 201, 204):
            return None
        return await self.get_contact(uid.replace(".vcf", ""))

    async def delete_contact(self, uid: str) -> bool:
        if not (self.enabled and self.client and self.url):
            return False
        href = f"{self.url.rstrip('/')}/{uid}.vcf" if not uid.endswith(".vcf") else uid
        resp = await self.client.delete(href)
        return resp.status_code in (200, 204)

    async def search_contact(self, *, name: str = None, phone: str = None, email: str = None) -> List[ContactInfo]:
        contacts = await self.list_contacts()
        results: List[ContactInfo] = []
        for c in contacts:
            matched = False
            if name and c.full_name and self._match_name(name, c.full_name):
                matched = True
            if phone and any(phone in p for p in c.phones):
                matched = True
            if email and any(email in e for e in c.emails):
                matched = True
            if matched:
                results.append(c)
        return results

    def _parse_multistatus(self, xml_text: str) -> List[ContactInfo]:
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
        info = ContactInfo(uid="")
        lines = [l.strip() for l in vcard.replace("\r\n", "\n").split("\n") if l.strip()]
        for line in lines:
            if line.startswith("FN:"):
                info.full_name = line[3:].strip()
            elif line.startswith("N:"):
                parts = line[2:].split(";")
                info.last_name = parts[0].strip() if len(parts) > 0 else None
                info.first_name = parts[1].strip() if len(parts) > 1 else None
            elif line.startswith("TEL"):
                m = re.match(r"TEL(;.+?)?:(?P<val>.+)", line)
                if m:
                    info.phones.append(m.group("val").strip())
            elif line.startswith("EMAIL"):
                m = re.match(r"EMAIL(;.+?)?:(?P<em>.+)", line)
                if m:
                    info.emails.append(m.group("em").strip())
            elif line.startswith("ORG:"):
                info.organization = line[4:].strip()
        return info

    def _to_vcard(self, data: Union[ContactCreate, ContactUpdate]) -> str:
        lines = ["BEGIN:VCARD", "VERSION:3.0"]
        if getattr(data, "full_name", None):
            lines.append(f"FN:{data.full_name}")
        if getattr(data, "last_name", None) or getattr(data, "first_name", None):
            ln = getattr(data, "last_name", "") or ""
            fn = getattr(data, "first_name", "") or ""
            lines.append(f"N:{ln};{fn};;;")
        for tel in getattr(data, "phones", []) or []:
            lines.append(f"TEL:{tel}")
        for em in getattr(data, "emails", []) or []:
            lines.append(f"EMAIL:{em}")
        if getattr(data, "organization", None):
            lines.append(f"ORG:{data.organization}")
        lines.append("END:VCARD")
        return "\r\n".join(lines)

    def _match_name(self, speaker_name: str, contact_full_name: str) -> bool:
        if not speaker_name or not contact_full_name:
            return False
        return speaker_name.strip().lower() in contact_full_name.strip().lower()

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict:
        if not (self.enabled and self.client):
            return {"speakers": {}, "contacts_found": 0}

        recognition = previous_results.get("speakers", {})
        contacts = await self.list_contacts()

        out = {}
        found = 0
        for label, meta in recognition.items():
            match = None
            name = meta.get("name")
            if name:
                for c in contacts:
                    if c.full_name and self._match_name(name, c.full_name):
                        match = c
                        break

            if not match:
                phone = meta.get("phone")
                email = meta.get("email")
                for c in contacts:
                    if phone and phone in c.phones:
                        match = c
                        break
                    if email and email in c.emails:
                        match = c
                        break

            if match:
                found += 1
                out[label] = {"contact": match.model_dump() if hasattr(match, "model_dump") else match.dict()}
            else:
                out[label] = {"contact": None}

        return {"speakers": out, "contacts_found": found}
