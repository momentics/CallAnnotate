#!/usr/bin/env python3
"""
Apache-2.0 License
Author: akoodoy@capilot.ru
Repository: https://github.com/momentics/CallAnnot

Скрипт для проверки полноты файла NOTICE в проекте CallAnnotate.
Он сравнивает список ключевых зависимостей из requirements.txt
с содержимым NOTICE и выводит отсутствующие записи.
Exit code 0 при отсутствии ошибок, 1 при обнаружении проблем.
"""

import sys
import re
from pathlib import Path

# Путь к проекту (корень репозитория)
ROOT = Path(__file__).parent.parent
REQUIREMENTS = ROOT / "src" / "requirements.txt"
NOTICE = ROOT / "NOTICE"

# Список обязательных зависимостей для включения в NOTICE
# Включает все зависимости из requirements.txt, требующие лицензионного уведомления
# согласно требованиям Apache License 2.0

REQUIRED_PACKAGES = [
    "fastapi",
    "starlette",
    "uvicorn[standard]",
    "python-multipart",
    "websockets",
    "aiofiles",
    "httpx",
    "anyio",
    "openai-whisper",
    "speechbrain",
    "pyannote.audio",
    "faiss-cpu",
    "torch",
    "torchaudio",
    "librosa",
    "soundfile",
    "numpy",
    "scipy",
    "deepfilternet",
    "pydub",
    "transformers",
    "pytorch-lightning",
    "caldav",
    "vobject",
    "requests",
    "pyyaml",
    "python-dotenv",
    "pydantic",
    "pydantic-settings",
    "python-dateutil",
    "pytz",
    "click",
    "pytest",
    "pytest-asyncio",
    "pytest-mock",
]

def load_requirements(path: Path) -> set[str]:
    pkgs = set()
    if not path.exists():
        print(f"Файл requirements.txt не найден: {path}", file=sys.stderr)
        sys.exit(1)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg = re.split(r"[<>=]", line)[0].lower()
        pkgs.add(pkg)
    return pkgs

def load_notice(path: Path) -> str:
    if not path.exists():
        print(f"Файл NOTICE не найден: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text(encoding="utf-8").lower()

def main():
    reqs = load_requirements(REQUIREMENTS)
    notice_text = load_notice(NOTICE)

    missing = []
    for pkg in REQUIRED_PACKAGES:
        key = pkg.lower()
        # Если пакет есть в requirements, то проверяем в NOTICE
        if key in reqs:
            if key not in notice_text:
                missing.append(pkg)

    if missing:
        print("В NOTICE отсутствуют следующие зависимости:")
        for pkg in missing:
            print(f"  - {pkg}")
        sys.exit(1)
    else:
        print("NOTICE содержит все обязательные зависимости.")
        sys.exit(0)

if __name__ == "__main__":
    main()
