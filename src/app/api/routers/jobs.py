import os
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, status, Depends, Request
from ...schemas import CreateJobRequest, CreateJobResponse, JobStatusResponse, TaskStatus
from ...utils import validate_audio_file_path, create_task_metadata
from ..deps import get_queue

router = APIRouter(tags=["Jobs"])

@router.post(
    "/api/v1/jobs",
    response_model=CreateJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_job(req: CreateJobRequest, request: Request, q=Depends(get_queue)):
    base_volume = Path(os.getenv("VOLUME_PATH", q._cfg["queue"]["volume_path"])).resolve()
    file_path = base_volume / "incoming" / req.filename

    vr = validate_audio_file_path(str(file_path))
    if not vr.is_valid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, vr.error)

    job_id = str(uuid.uuid4())
    meta = create_task_metadata(job_id, str(file_path), req.filename, req.priority)
    await q.add_task(job_id, meta)

    return CreateJobResponse(
        job_id=job_id,
        status=TaskStatus.QUEUED,
        message="queued",
        created_at=datetime.utcnow(),
        file_info={"filename": req.filename, "path": str(file_path)},
        progress_url=str(request.url_for("job_status", job_id=job_id)),
        result_url=str(request.url_for("job_result", job_id=job_id)),
    )

@router.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, name="job_status")
async def job_status(job_id: str, q=Depends(get_queue)):
    tr = await q.get_task_result(job_id)
    if not tr:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    status_str = tr.status if isinstance(tr.status, str) else tr.status.value
    return JobStatusResponse(
        job_id=tr.task_id,
        status=status_str,
        message=tr.message,
        progress={"percentage": tr.progress, "current_stage": ""},
        timestamps={
            "created_at": tr.created_at,
            "started_at": getattr(tr, "started_at", None),
            "completed_at": getattr(tr, "completed_at", None),
        },
        file_info=None
    )

@router.get("/api/v1/jobs/{job_id}/result", name="job_result")
async def job_result(job_id: str, q=Depends(get_queue)):
    tr = await q.get_task_result(job_id)
    if not tr:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    return tr.result or {}

@router.delete("/api/v1/jobs/{job_id}")
async def job_cancel(job_id: str, q=Depends(get_queue)):
    if not await q.cancel_task(job_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    return "", status.HTTP_204_NO_CONTENT
