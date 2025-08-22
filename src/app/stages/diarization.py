# src/app/stages/diarization.py
# -*- coding: utf-8 -*-
"""
Этап диаризации для CallAnnotate с исправленным определением уникальных спикеров
и передачей метаданных модели в payload для корректного заполнения processing_info.

Дополнительно:
- Добавлен постпроцессинг сегментов: фильтрация сверхкоротких, слияние соседних сегментов одного спикера.
  Это устраняет артефактные микро-сегменты (например, 0.017с), которые дублируют начало следующего сегмента
  и приводят к некорректному увеличению количества сегментов в итоговом JSON.

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment, SlidingWindow

from ..config import AppSettings
from .base import BaseStage


class DiarizationStage(BaseStage):
    CONFIG_FILE = "pyannote_diarization_config.yaml"

    @property
    def stage_name(self) -> str:
        return "diarization"

    async def _initialize(self) -> None:
        model_id = str(self.config.get("model"))
        use_auth = self.config.get("use_auth_token")
        device = self.config.get("device", "cpu")

        # Кэш для локальной конфигурации pyannote
        self.cache_path = Path(self.volume_path).expanduser().resolve() / "models" / "pyannote"
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_path / self.CONFIG_FILE

        def loader():
            # Подгружаем pipeline. Если используется локальный конфиг — загружаем из файла.
            cwd = Path.cwd().resolve()
            cd_to = self.cache_path.parent.resolve()
            os.chdir(cd_to)

            # Если у вас есть локальный конфиг self.cache_path — используем его,
            # иначе можно раскомментировать загрузку с HuggingFace:
            # pipe = Pipeline.from_pretrained(model_id, use_auth_token=use_auth)
            pipe = Pipeline.from_pretrained(self.cache_path)

            try:
                pipe.to(device)
            except Exception:
                pass

            os.chdir(cwd)
            return pipe

        if self.models_registry:
            self.pipeline = self.models_registry.get_model(
                self.logger,
                f"diarization:{model_id}",
                loader,
                model_name=model_id,
                framework="pyannote.audio"
            )
        else:
            self.pipeline = loader()

        self._initialized = True
        self.logger.info("DiarizationStage инициаллизирована")

    async def _process_impl(
        self,
        file_path: str,
        task_id: str,
        previous_results: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        audio_path = Path(file_path)
        with sf.SoundFile(str(audio_path)) as f:
            sr = f.samplerate
            duration = len(f) / sr

        if progress_callback:
            await progress_callback(10, "Начало диаризации")

        # Используем настройки оконного режима из конфига (не перезаписываем насильно)
        window_enabled = bool(self.config.get("window_enabled", False))
        window_size = float(self.config.get("window_size", duration))
        hop_size = float(self.config.get("hop_size", window_size))
        min_spk = int(self.config.get("min_speakers", 1))
        max_spk = int(self.config.get("max_speakers", 2))

        if window_enabled and window_size < duration:
            annotation = await self._process_with_sliding_window(
                audio_path, duration, window_size, hop_size, min_spk, max_spk, 
                progress_callback                
            )
        else:
            if progress_callback:
                await progress_callback(10, "Диаризация файла целиком")
            annotation = self.pipeline(str(audio_path), min_speakers=min_spk, max_speakers=max_spk)
            if progress_callback:
                await progress_callback(70, "Диаризация завершена")

        if progress_callback:
            await progress_callback(80, "Извлечение сегментов")

        raw_segments = self._extract_segments(annotation, duration)

        # Пост-обработка: фильтрация коротких и слияние соседних сегментов одного спикера.
        # Порог минимальной длительности берём из конфигурации транскрипции (если доступен),
        # чтобы согласовать логику с min_segment_duration.
        # Если конфиг недоступен — используем дефолт 0.2 секунды.
        try:
            min_seg = float(self.app_config.transcription.min_segment_duration)  # type: ignore[attr-defined]
        except Exception:
            min_seg = 0.2

        # Порог для объединения "стыкующихся" сегментов одного спикера: 0.05с
        merge_gap = 0.05

        segments = self._merge_and_filter_segments(raw_segments, min_duration=min_seg, merge_gap=merge_gap)

        # Собираем уникальные метки спикеров из отфильтрованных сегментов
        labels = []
        for seg in segments:
            if seg["speaker"] not in labels:
                labels.append(seg["speaker"])

        # Если меньше двух, добавляем 'unknown'
        if len(labels) == 0:
            labels = ["speaker_1", "unknown"]
        elif len(labels) == 1:
            labels.append("unknown")
        # Если больше двух — оставляем как есть (в реальных кейсах допускается >2),
        # а выбор топ-2 по длительности был удалён, чтобы не терять реальных спикеров.

        processing_time = round(time.perf_counter() - start_time, 3)
        self.logger.info(
            f"Диаризация завершена за {processing_time} сек: speakers={labels}, segments={len(segments)}"
        )

        # Информация о модели для заполнения processing_info
        model_info = {
            "model_name": str(self.config.get("model")),
            "framework": "pyannote.audio"
        }

        return {
            "segments": segments,
            "speakers": labels,
            "total_segments": len(segments),
            "total_speakers": len(labels),
            "processing_time": processing_time,
            "model_info": model_info
        }

    async def _process_with_sliding_window(
        self,
        audio_path: Path,
        duration: float,
        window_size: float,
        hop_size: float,
        min_spk: int, max_spk: int,
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None
    ) -> Annotation:
        combined = Annotation(uri=audio_path.stem)
        sw = SlidingWindow(duration=window_size, step=hop_size)
        windows = list(sw(Segment(0.0, duration), align_last=False))
        total = len(windows)
        if progress_callback:
            await progress_callback(10, f"Разбиваем на окна Окон={total} duration={window_size} step={hop_size}")
        for idx, window in enumerate(windows, start=1):
            ann = self.pipeline({
                "uri": audio_path.stem,
                "audio": str(audio_path),
                "segment": window
                },
                min_speakers=min_spk,
                max_speakers=max_spk)
            if progress_callback:
                pct = int(11 + 60 * idx / total)
                await progress_callback(pct, f"Обработка окна {idx}/{total}")
            for segment, track, label in ann.itertracks(yield_label=True):
                start = segment.start + window.start
                end = min(segment.end + window.start, duration)
                if end > start:
                    combined[Segment(start, end), track] = label
        return combined

    def _extract_segments(self, annotation: Annotation, duration: float) -> List[Dict[str, Any]]:
        """
        Преобразует pyannote Annotation в список сегментов.
        """
        raw: List[Dict[str, Any]] = []
        for segment, _, label in annotation.itertracks(yield_label=True):  # type: ignore
            start = round(segment.start, 3)
            end = round(min(segment.end, duration), 3)
            if end <= start:
                continue
            raw.append({
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "speaker": str(label),
                "confidence": 0.0
            })
        raw.sort(key=lambda x: x["start"])
        return raw

    def _merge_and_filter_segments(
        self,
        segments: List[Dict[str, Any]],
        min_duration: float = 0.2,
        merge_gap: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Фильтрует сверхкороткие сегменты и сливает соседние сегменты одного спикера,
        если разрыв между ними меньше merge_gap.

        - min_duration: минимальная длительность сегмента (сегменты короче отбрасываются).
        - merge_gap: если следующий сегмент того же спикера начинается в пределах этого зазора,
          они объединяются.

        Возвращает упорядоченный список сегментов.
        """
        if not segments:
            return []

        # Предварительно: избавимся от совсем коротких кусочков, которые часто являются артефактами.
        pre = [s for s in segments if float(s["duration"]) >= float(min_duration)]

        if not pre:
            return []

        pre.sort(key=lambda x: (x["speaker"], x["start"]))
        merged: List[Dict[str, Any]] = []
        current = pre[0].copy()

        for s in pre[1:]:
            same_spk = (s["speaker"] == current["speaker"])
            gap = float(s["start"]) - float(current["end"])
            if same_spk and gap >= -1e-6 and gap <= merge_gap:
                # Сливаем сегменты одного спикера, которые «стыкуются» или почти соприкасаются
                current["end"] = max(float(current["end"]), float(s["end"]))
                current["duration"] = round(float(current["end"]) - float(current["start"]), 3)
            else:
                merged.append(current)
                current = s.copy()
        merged.append(current)

        # Финальная сортировка по времени
        merged.sort(key=lambda x: x["start"])
        return merged
