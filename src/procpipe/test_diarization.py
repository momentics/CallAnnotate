#!/usr/bin/env python
# ────────────────────────────────────────────────────────────────
# Тест-запуск ThreadSafeIncrementalDiarizationTranscription,
# ThreadSafeIncrementalMerger и обёртки DeepFilterNet.
#
# Пример:
#     python test_diarization.py --audio sample.wav --hf_token YOUR_TOKEN
#
# Требования (установите один раз):
#     pip install pyannote.audio deepfilternet transformers soundfile torchaudio
# ────────────────────────────────────────────────────────────────

import argparse
import os
from collections import defaultdict

import torch
import soundfile as sf
from pyannote.core import Annotation
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ------------------------------------------------------------------
# импорт собственных классов (предполагается, что они лежат в том же
# каталоге или установлены как пакет)
# ------------------------------------------------------------------
from ThreadSafeIncrementalDiarizationTranscription import ThreadSafeIncrementalDiarizationTranscription
#import ThreadSafeIncrementalMerger


def main():
    parser = argparse.ArgumentParser(
        description="Простой тестовый запуск инкрементального "
                    "диаризации + ASR с подавлением шума DeepFilterNet"
    )
    parser.add_argument(
        "--audio", "-a", required=True,
        help="Путь к WAV/FLAC/MP3 файлу (моно или стерео)"
    )
    parser.add_argument(
        "--window", type=float, default=60.0,
        help="Длина окна, с (по умолчанию 60)"
    )
    parser.add_argument(
        "--step", type=float, default=30.0,
        help="Шаг окна, с (по умолчанию 30, т.е. 50 % перекрытие)"
    )
    parser.add_argument(
        "--hf_token", default=os.getenv("HF_TOKEN"),
        help="Токен Hugging Face (если модели pyannote требуют авторизации)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        parser.error(f"Файл {args.audio} не найден")

    # ------------------------------------------------------------------
    # Инициализируем Whisper-ASR (можно заменить модель)
    # ------------------------------------------------------------------
    print("⇢ Загружаем Whisper-модель ‘openai/whisper-small’ …")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="ru", task="transcribe"
    )
    asr_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Создаём потокобезопасный конвейер
    # ------------------------------------------------------------------
    print("⇢ Создаём инкрементальный пайплайн …")
    pipeline = ThreadSafeIncrementalDiarizationTranscription(
        vad_model="pyannote/voice-activity-detection",
        emb_model="pyannote/embedding",
        asr_processor=processor,
        asr_model=asr_model,
        window_duration=args.window,
        window_step=args.step,
        cluster_threshold=0.6,
        min_segment_duration=0.5,
        collar=0.2,
        max_gap=0.5,
    )

    # ------------------------------------------------------------------
    # Буферы для итогов
    # ------------------------------------------------------------------
    global_diar = Annotation()
    global_transcripts = defaultdict(list)

    # ------------------------------------------------------------------
    # Инкрементальная обработка
    # ------------------------------------------------------------------
    print(f"⇢ Обрабатываем файл «{args.audio}» …")
    for idx, result in enumerate(pipeline.process(args.audio)):
        part_diar = result["partial_diarization"]
        part_texts = result["completed_transcripts"]

        # обновляем совокупную разметку
        global_diar.update(part_diar, intersect=True)

        # накапливаем завершённые транскрипты
        for spk, entries in part_texts.items():
            global_transcripts[spk].extend(entries)

        # выводим промежуточный отчёт каждые два окна
        if idx % 2 == 1:
            print(f"\n─── Промежуточный отчёт (после окна #{idx}) ───")
            for seg, _, spk in part_diar.itertracks(yield_label=True):
                print(f"{seg.start:7.1f}-{seg.end:7.1f} → {spk}")
            for spk, entries in part_texts.items():
                for ent in entries:
                    st, ed = ent["segment"]
                    print(f"{spk}: {st:7.1f}-{ed:7.1f} | {ent['text']}")

    # ------------------------------------------------------------------
    # Финальный вывод
    # ------------------------------------------------------------------
    print("\n=== Итоговая диаризация ===")
    for seg, _, spk in global_diar.itertracks(yield_label=True):
        print(f"{seg.start:7.1f}-{seg.end:7.1f} s → {spk}")

    print("\n=== Итоговые транскрипты ===")
    for spk, entries in global_transcripts.items():
        for ent in entries:
            st, ed = ent["segment"]
            print(f"{spk}: {st:7.1f}-{ed:7.1f} s → {ent['text']}")


if __name__ == "__main__":
    main()
