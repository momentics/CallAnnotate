# tests/test_annotation_service_unit.py

import asyncio
import pytest
from pathlib import Path
from app.annotation import AnnotationService
from app.schemas import AnnotationResult, AudioMetadata
from app.stages.base import StageResult

class DummyStage:
    def __init__(self, name, payload, success=True, error=None):
        self.stage_name = name
        self._payload = payload
        self._success = success
        self._error = error
        self._initialized = False

    async def process(self, file_path, task_id, previous_results, progress_callback=None):
        # simulate one-time initialization
        if not self._initialized:
            self._initialized = True
        # simulate async
        await asyncio.sleep(0)
        if not self._success:
            return StageResult(
                stage_name=self.stage_name,
                processing_time=0.0,
                model_info={},
                payload={},
                success=False,
                error=self._error or "error"
            )
        return StageResult(
            stage_name=self.stage_name,
            processing_time=0.0,
            model_info={},
            payload=self._payload,
            success=True,
            error=None
        )

@pytest.mark.asyncio
async def test_annotation_service_success(tmp_path, monkeypatch):
    # Prepare dummy audio file
    wav = tmp_path / "call.wav"
    wav.write_bytes(b"\x00" * 1024)

    # Monkey-patch extract_audio_metadata
    fake_meta = AudioMetadata(
        filename="call.wav",
        duration=2.0,
        sample_rate=16000,
        channels=1,
        format="wav",
        bitrate=None,
        size_bytes=1024
    )
    monkeypatch.setattr("app.annotation.extract_audio_metadata", lambda p: fake_meta)

    # Instantiate service with dummy stages
    svc = AnnotationService({
        "diarization": {},
        "transcription": {},
        "recognition": {},
        "carddav": {}
    })
    svc.stages = [
        DummyStage("diarization", {"segments": [{"start": 0.0, "end": 1.0, "duration": 1.0, "speaker": "spk1", "confidence": 1.0}]}),
        DummyStage("transcription", {"segments": [], "words": []}),
        DummyStage("recognition", {"speakers": {}}),
        DummyStage("carddav", {"speakers": {}, "contacts_found": 0}),
    ]

    # Collect progress updates
    progress = []
    def cb(pct, msg):
        progress.append((pct, msg))

    result = await svc.process_audio(str(wav), "job123", progress_callback=cb)
    ann = AnnotationResult(**result)

    # Validate result structure
    assert ann.task_id == "job123"
    assert ann.audio_metadata.filename == "call.wav"
    assert ann.statistics.total_speakers == 1
    # Last progress update should be 100%
    assert progress and progress[-1][0] == 100

@pytest.mark.asyncio
async def test_annotation_service_stage_failure(tmp_path, monkeypatch):
    # Prepare dummy audio file
    wav = tmp_path / "err.wav"
    wav.write_bytes(b"\x00" * 512)

    # Monkey-patch extract_audio_metadata
    fake_meta = AudioMetadata(
        filename="err.wav",
        duration=1.0,
        sample_rate=8000,
        channels=1,
        format="wav",
        bitrate=None,
        size_bytes=512
    )
    monkeypatch.setattr("app.annotation.extract_audio_metadata", lambda p: fake_meta)

    svc = AnnotationService({
        "diarization": {},
        "transcription": {},
        "recognition": {},
        "carddav": {}
    })
    # Second stage fails
    svc.stages = [
        DummyStage("diarization", {"segments": []}),
        DummyStage("transcription", {}, success=False, error="transcribe failed"),
        DummyStage("recognition", {"speakers": {}}),
        DummyStage("carddav", {"speakers": {}, "contacts_found": 0}),
    ]

    result = await svc.process_audio(str(wav), "job_err")
    ann = AnnotationResult(**result)

    # Even on failure, service returns a valid AnnotationResult
    assert isinstance(ann, AnnotationResult)
    # No segments extracted on failure
    assert ann.segments == []
    # Unknown speakers count matches
    assert ann.statistics.unknown_speakers >= 0
