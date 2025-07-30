# Создано файл: scripts/whisper_cache_manager.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Менеджер кэша Whisper моделей для CallAnnotate

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0

Утилита для управления локальным кэшем моделей OpenAI Whisper:
  - list      — показать доступные локальные модели
  - available — показать все поддерживаемые размеры моделей Whisper
  - download  — загрузить модель в кэш
  - size      — вывести размер кэша (в человекочитаемом формате)
  - clear     — удалить кэш моделей
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import whisper


def human_readable_size(num_bytes: int) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"


def cmd_list(cache_dir: Path):
    """Показать локально закэшированные модели"""
    if not cache_dir.exists():
        print("Кэш не найден.")
        return
    models = [p.name for p in cache_dir.iterdir() if p.is_dir()]
    if not models:
        print("В кэше нет моделей.")
    else:
        print("Локальные модели:")
        for m in sorted(models):
            print(f"  {m}")


def cmd_available():
    """Показать все поддерживаемые Whisper размеры"""
    sizes = whisper.available_models()
    print("Доступные модели Whisper:")
    for sz in sizes:
        print(f"  {sz}")


def cmd_download(cache_dir: Path, model_size: str):
    """Загрузить модель Whisper в кэш"""
    print(f"Загрузка модели '{model_size}' в {cache_dir} ...")
    # load_model сам кэширует в стандартный HF_CACHE
    os.environ["XDG_CACHE_HOME"] = str(cache_dir.parent)
    model = whisper.load_model(model_size, device="cpu")
    # проверка наличия папки после загрузки
    target = cache_dir / model_size
    if target.exists():
        print(f"Модель '{model_size}' успешно загружена.")
    else:
        print(f"Не удалось найти каталог {target} после загрузки.", file=sys.stderr)


def cmd_size(cache_dir: Path):
    """Показать общий размер кэша"""
    if not cache_dir.exists():
        print("Кэш не найден.")
        return
    total = 0
    for root, _, files in os.walk(cache_dir):
        for f in files:
            fp = Path(root) / f
            total += fp.stat().st_size
    print(f"Размер кэша: {human_readable_size(total)}")


def cmd_clear(cache_dir: Path):
    """Удалить каталог кэша целиком"""
    if not cache_dir.exists():
        print("Кэш не найден.")
        return
    confirm = input(f"Удалить весь кэш моделей в {cache_dir}? [y/N]: ").lower()
    if confirm == "y":
        shutil.rmtree(cache_dir)
        print("Кэш удалён.")
    else:
        print("Операция отменена.")


def main():
    parser = argparse.ArgumentParser(
        description="Менеджер кэша Whisper моделей"
    )
    parser.add_argument(
        "--cache-dir", "-c",
        default=os.getenv("WHISPER_CACHE_DIR", os.path.expanduser("~/.cache/whisper")),
        help="Путь к каталогу кэша моделей"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Список локальных моделей")
    sub.add_parser("available", help="Доступные размеры моделей Whisper")
    dl = sub.add_parser("download", help="Загрузить модель в кэш")
    dl.add_argument("model", choices=whisper.available_models(), help="Размер модели")

    sub.add_parser("size", help="Показать размер кэша")
    sub.add_parser("clear", help="Удалить кэш")

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)

    if args.command == "list":
        cmd_list(cache_dir)
    elif args.command == "available":
        cmd_available()
    elif args.command == "download":
        cache_dir.mkdir(parents=True, exist_ok=True)
        cmd_download(cache_dir, args.model)
    elif args.command == "size":
        cmd_size(cache_dir)
    elif args.command == "clear":
        cmd_clear(cache_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
