#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт предзагрузки моделей Whisper для CallAnnotate Docker контейнера

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import argparse
import os
import sys
import whisper
import logging

def preload_model(model_size: str, cache_dir: str):
    """
    Загружает модель Whisper указанного размера и сохраняет её в кэш.
    """
    # Настроить окружение для кеша HuggingFace/transformers
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["TORCH_HOME"] = cache_dir

    logging.info(f"Предзагрузка модели Whisper '{model_size}' в каталог {cache_dir}")
    try:
        # whisper.load_model сам кэширует веса в ~/.cache/whisper или в указанном cache_dir
        model = whisper.load_model(model_size, download_root=cache_dir)
        available = whisper.available_models()
        logging.info(f"Успешно предзагружены модели: {available}")
    except Exception as e:
        logging.error(f"Ошибка при предзагрузке модели Whisper: {e}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Предзагрузка моделей OpenAI Whisper в локальный кэш"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="small",
        help="Размер модели Whisper (tiny, base, small, medium, large)"
    )
    parser.add_argument(
        "--cache-dir",
        "-c",
        type=str,
        default=os.path.expanduser("~/.cache/whisper"),
        help="Каталог для кеширования весов модели"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Создать директорию кэша, если не существует
    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Настроить простое логирование
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    preload_model(args.model, cache_dir)

if __name__ == "__main__":
    main()
