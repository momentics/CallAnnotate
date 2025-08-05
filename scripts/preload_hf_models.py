#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Предзагрузка любых моделей Hugging Face для CallAnnotate.

Пример использования (в Dockerfile или локально):

    python preload_hf_models.py \
        --cache-dir /volume/models/cache \
        pyannote/speaker-diarization-3.1 \
        speechbrain/spkrec-ecapa-voxceleb \
        openai/whisper-small

Автор: akoodoy@capilot.ru
Лицензия: Apache-2.0
"""

import argparse
import os
import sys
import shutil
import textwrap
from pathlib import Path
from typing import List, Tuple

# --- вспомогательные импорты без «тяжёлых» библиотек ---
try:
    from transformers import AutoModel, AutoConfig, AutoProcessor
except ImportError:
    AutoModel = AutoConfig = AutoProcessor = None  # отложенная проверка

try:
    from pyannote.audio import Pipeline as PaPipeline
except ImportError:
    PaPipeline = None

try:
    from speechbrain.pretrained import SpeakerRecognition
except ImportError:
    SpeakerRecognition = None

import torch  # почти всегда уже установлен в контейнере


# --------------------------------------------------------------------------- #
#  CLI                                                                       #
# --------------------------------------------------------------------------- #
def cli() -> Tuple[Path, List[str]]:
    parser = argparse.ArgumentParser(
        prog="preload_hf_models.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Загружает одну или несколько моделей HF в локальный кэш.
            Поддерживаются:
              • модели Transformers (AutoModel/Config/Processor);
              • pyannote.audio Pipeline;
              • SpeechBrain SpeakerRecognition.
            """
        ),
    )
    parser.add_argument(
        "models",
        metavar="MODEL_ID",
        nargs="+",
        help="ID модели на Hugging Face Hub (любого типа)",
    )
    parser.add_argument(
        "--cache-dir",
        "-c",
        type=str,
        default=os.getenv("HF_HOME", "~/.cache/huggingface"),
        help="Каталог для оффлайн-кэша",
    )
    args = parser.parse_args()
    cache_dir = Path(os.path.expanduser(args.cache_dir)).resolve()
    return cache_dir, args.models


# --------------------------------------------------------------------------- #
#  Базовая инициализация окружения                                           #
# --------------------------------------------------------------------------- #
def setup_env(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Объявляем все стандартные переменные → один общий кэш
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["TORCH_HOME"] = str(cache_dir)
    # Pyannote/torch.hub используют TORCH_HOME
    print(f"[ℹ] Используется кэш: {cache_dir}")


# --------------------------------------------------------------------------- #
#  Унифицированная загрузка модели                                           #
# --------------------------------------------------------------------------- #
def download_model(model_id: str) -> None:
    """
    Пробует последовательно:
      1. Transformers (универсально для большинства моделей);
      2. pyannote.audio Pipeline;
      3. SpeechBrain SpeakerRecognition.

    Если модель уже есть в кэше, лишней загрузки не будет.
    """

    # --- Transformers ---
    if AutoConfig is not None:
        try:
            print(f"[→] Transformers: {model_id}")
            AutoConfig.from_pretrained(model_id)
            # не все модели требуют процессор; пробуем, но игнорируем ошибки
            try:
                AutoProcessor.from_pretrained(model_id)
            except Exception:
                pass
            # веса могут быть большими; грузить полностью не обязательно
            AutoModel.from_pretrained(model_id)
            print(f"[✓] Transformers OK: {model_id}")
            return
        except Exception as e:
            print(f"[…] Не Transformers: {e}")

    # --- pyannote.audio ---
    if PaPipeline is not None:
        try:
            print(f"[→] pyannote.audio: {model_id}")
            pipe = PaPipeline.from_pretrained(
                model_id,
                use_auth_token=os.getenv("HF_TOKEN"),
                cache_dir=os.getenv("HF_HOME"),
            )
            # принудительно выгружаем, чтобы не держать в памяти
            del pipe
            torch.cuda.empty_cache()
            print(f"[✓] pyannote.audio OK: {model_id}")
            return
        except Exception as e:
            print(f"[…] Не pyannote.audio: {e}")

    # --- SpeechBrain ---
    if SpeakerRecognition is not None:
        try:
            print(f"[→] SpeechBrain: {model_id}")
            sr = SpeakerRecognition.from_hparams(
                source=model_id,
                savedir=os.getenv("HF_HOME"),
                run_opts={"device": "cpu"},
                use_auth_token=os.getenv("HF_TOKEN"),
            )
            del sr
            print(f"[✓] SpeechBrain OK: {model_id}")
            return
        except Exception as e:
            print(f"[…] Не SpeechBrain: {e}")

    print(f"[✗] Не удалось определить тип модели для «{model_id}» — пропущена.")


# --------------------------------------------------------------------------- #
#  Главная точка входа                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    cache_dir, model_ids = cli()
    setup_env(cache_dir)

    failed = []
    for mid in model_ids:
        try:
            download_model(mid)
        except Exception as exc:
            failed.append((mid, str(exc)))

    # --- итоговая сводка ---
    print("\n=== Сводка ===")
    total = len(model_ids)
    ok = total - len(failed)
    print(f"Успешно: {ok}/{total}")
    if failed:
        print("Не удалось загрузить:")
        for mid, err in failed:
            print(f" • {mid}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
