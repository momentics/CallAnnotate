# tests/test_preprocessing_windowed_overlap.py
# -*- coding: utf-8 -*-
"""
Дополнительные тесты для проверки режима «windowed» в _merge_chunks
модуля PreprocessingStage.

Автор: akoodoy@capilot.ru
Лицензия: Apache-2.0
"""

import numpy as np
import pytest

from app.stages.preprocessing import PreprocessingStage


@pytest.mark.asyncio
async def test_merge_chunks_windowed_weights():
    """
    Проверяет, что «windowed»-режим:
      1. Возвращает массив той же длины, что и «linear».
      2. Модифицирует значения в области перекрытия
         в соответствии с Хэннинговым окном.
    """
    # Создаём два простых чанка разной «амплитуды»
    a = np.ones(5, dtype=np.float32)          # [1, 1, 1, 1, 1]
    b = np.ones(5, dtype=np.float32) * 2      # [2, 2, 2, 2, 2]

    overlap_ms = 2
    sample_rate = 1_000                       # → overlap_samples = 2

    stage = PreprocessingStage({}, models_registry=None)

    # Склейка без окна
    merged_linear = await stage._merge_chunks(
        [a.copy(), b.copy()],
        overlap_ms=overlap_ms,
        sample_rate=sample_rate,
        method="linear",
    )

    # Склейка с использованием оконного сглаживания
    merged_windowed = await stage._merge_chunks(
        [a.copy(), b.copy()],
        overlap_ms=overlap_ms,
        sample_rate=sample_rate,
        method="windowed",
    )

    # 1. Длина должна совпадать
    assert merged_linear.shape == merged_windowed.shape

    # 2. Значения в области перекрытия должны отличаться
    assert not np.allclose(merged_linear, merged_windowed)

    # 3. Проверяем, что весовые коэффициенты Хэннинга применены корректно
    overlap_samples = int(overlap_ms * sample_rate / 1000)
    window = np.hanning(overlap_samples * 2)

    # Формируем ожидаемый результат вручную
    expected = a.copy()
    expected[-overlap_samples:] *= window[:overlap_samples]
    expected_full = np.concatenate([expected, b[overlap_samples:]])

    np.testing.assert_allclose(
        merged_windowed,
        expected_full,
        rtol=1e-6,
        atol=1e-6,
        err_msg="windowed merge не соответствует ожидаемому результату с окном Хэннинга",
    )
