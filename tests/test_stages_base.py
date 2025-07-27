# -*- coding: utf-8 -*-
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import pytest
import asyncio
from app.stages.base import BaseStage, StageResult

class DummyStage(BaseStage):
    @property
    def stage_name(self):
        return "dummy"
    async def _initialize(self):
        # simulate init
        await asyncio.sleep(0)
    async def _process_impl(self, file_path, task_id, prev, progress_callback=None):
        if file_path == "err":
            raise RuntimeError("fail")
        return {"ok": True}

@pytest.mark.asyncio
async def test_process_success(tmp_path):
    stage = DummyStage({}, None)
    res = await stage.process("file", "tid")
    assert isinstance(res, StageResult)
    assert res.success
    assert res.payload["ok"]

@pytest.mark.asyncio
async def test_process_error(tmp_path):
    stage = DummyStage({}, None)
    res = await stage.process("err", "tid")
    assert not res.success
    assert "fail" in res.error

@pytest.mark.asyncio
async def test_timing_and_model_info(tmp_path):
    cfg = {"foo": "bar"}
    stage = DummyStage(cfg, None)
    # ensure model_info contains config
    res = await stage.process("file", "tid")
    assert res.model_info["config"] == cfg
    assert res.processing_time >= 0
