# tests/test_annotation_service_build.py

import asyncio
import pytest

import app.annotation as ann_mod
from app.schemas import AnnotationResult, AudioMetadata

class DummyStage:
    def __init__(self, name, payload, success=True, error_msg=None):
        self.stage_name = name
        self._payload = payload
        self._success = success
        self._error_msg = error_msg

    async def process(self, file_path, task_id, context, progress_callback=None):
        # эмулируем прогресс
        if progress_callback:
            await progress_callback(50, f"{self.stage_name} midway")
        await asyncio.sleep(0)
        if not self._success:
            class R: pass
            r = R()
            r.stage_name = self.stage_name
            r.processing_time = 0.0
            r.model_info = {}
            r.payload = {}
            r.success = False
            r.error = self._error_msg or "error"
            return r
        class R: pass
        r = R()
        r.stage_name = self.stage_name
        r.processing_time = 0.0
        r.model_info = {}
        r.payload = self._payload
        r.success = True
        r.error = None
        return r

@pytest.mark.asyncio
async def test_build_final_annotation_basic(tmp_path, monkeypatch):
    # Подготовка фабрики аудиофайла и метаданных
    fake_file = tmp_path / "a.wav"
    fake_file.write_bytes(b"\x00"*256)
    fake_meta = AudioMetadata(
        filename="a.wav", duration=2.0, sample_rate=2, channels=1,
        format="wav", bitrate=None, size_bytes=256
    )
    monkeypatch.setattr(ann_mod, "extract_audio_metadata", lambda p: fake_meta)

    svc = ann_mod.AnnotationService({"diarization": {}, "transcription": {}, "recognition": {}, "carddav": {}})
    # все этапы успешны
    svc.stages = [
        DummyStage("diarization", {"segments": [{"start":0.0,"end":1.0,"duration":1.0,"speaker":"spk1"}]}),
        DummyStage("transcription", {"words": [{"start":0.2,"end":0.3,"word":"hi","probability":0.9}], "segments": []}),
        DummyStage("recognition", {"speakers": {"spk1":{"identified":True,"name":"Alice","confidence":0.8}}}),
        DummyStage("carddav", {"speakers": {"spk1":{"contact":{"uid": "alice_001", "full_name":"Alice"}}}, "contacts_found":1})
    ]

    calls = []
    def prog_cb(p, m):
        calls.append(p)
    res_dict = await svc.process_audio(str(fake_file), "jid", progress_callback=prog_cb)
    res = AnnotationResult(**res_dict)
    assert res.task_id == "jid"
    # проверяем, что статистика совпадает
    assert res.statistics.total_speakers == 1
    assert any(p == 100 for p in calls)
