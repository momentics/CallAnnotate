# tests/test_annotation_service.py

import asyncio
import pytest

import app.annotation as ann_mod
from app.schemas import AnnotationResult, AudioMetadata

class DummyStage:
    def __init__(self, name, payload, success=True, error=None):
        self.stage_name = name
        self._payload = payload
        self._success = success
        self._error = error
        self.initialized = False

    async def process(self, file_path, task_id, previous_results, progress_callback=None):
        # simulate initialization call
        if not self.initialized:
            self.initialized = True
        # simulate a bit of async work
        await asyncio.sleep(0)
        if not self._success:
            return type("R", (), {
                "stage_name": self.stage_name,
                "processing_time": 0.0,
                "model_info": {},
                "payload": {},
                "success": False,
                "error": self._error or "error"
            })()
        return type("R", (), {
            "stage_name": self.stage_name,
            "processing_time": 0.0,
            "model_info": {},
            "payload": self._payload,
            "success": True,
            "error": None
        })()

@pytest.mark.asyncio
async def test_process_audio_all_stages_success(tmp_path, monkeypatch):
    # create a dummy wav file
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"\x00" * 1024)
    # monkeypatch extract_audio_metadata
    fake_meta = AudioMetadata(
        filename="test.wav",
        duration=1.0,
        sample_rate=2,
        channels=1,
        format="wav",
        bitrate=None,
        size_bytes=1024
    )
    monkeypatch.setattr(ann_mod, "extract_audio_metadata", lambda p: fake_meta)

    # build a service with dummy stages
    service = ann_mod.AnnotationService({"diarization":{}, "transcription":{}, "recognition":{}, "carddav":{}})
    service.stages = [
        DummyStage("diarization", {"segments": []}),
        DummyStage("transcription", {"segments": [], "words": []}),
        DummyStage("recognition", {"speakers": {}}),
        DummyStage("carddav", {"speakers": {}, "contacts_found": 0}),
    ]

    calls = []
    def prog_cb(pct, msg):
        calls.append((pct, msg))

    result = await service.process_audio(str(audio_path), "jid", progress_callback=prog_cb)
    assert isinstance(result, dict)
    # should finish at 100%
    assert calls and calls[-1][0] == 100
    ann = AnnotationResult(**result)
    assert ann.task_id == "jid"
    assert ann.audio_metadata.filename == "test.wav"

@pytest.mark.asyncio
async def test_process_audio_stage_error(tmp_path, monkeypatch):
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"\x00" * 512)
    fake_meta = AudioMetadata(
        filename="test.wav", duration=0.5, sample_rate=2,
        channels=1, format="wav", bitrate=None, size_bytes=512
    )
    monkeypatch.setattr(ann_mod, "extract_audio_metadata", lambda p: fake_meta)

    service = ann_mod.AnnotationService({"diarization":{}, "transcription":{}, "recognition":{}, "carddav":{}})
    # one stage will fail
    service.stages = [
        DummyStage("diarization", {"segments": []}),
        DummyStage("transcription", {}, success=False, error="fail"),
        DummyStage("recognition", {"speakers": {}}),
        DummyStage("carddav", {"speakers": {}, "contacts_found": 0}),
    ]

    # even if a stage fails, overall returns dict
    result = await service.process_audio(str(audio_path), "jid2")
    assert isinstance(result, dict)
    ann = AnnotationResult(**result)
    # ensure segments and statistics exist
    assert hasattr(ann, "segments")
    assert isinstance(ann.statistics.total_speakers, int)
