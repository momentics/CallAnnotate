# src/app.py
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import os
import uuid
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from queue_manager import QueueManager

SUPPORTED_FORMATS = {"wav", "mp3", "flac", "opus"}

# Глобальная переменная для совместимости с тестами
queue: QueueManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    volume_path = os.getenv("VOLUME_PATH", "/app/volume")
    global queue
    queue = QueueManager(volume_path=volume_path)
    app.state.queue = queue
    yield
    # graceful shutdown при необходимости

app = FastAPI(
    title="CallAnnotate API",
    version="1.0.0",
    root_path="",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    progress_url: str
    result_url: str

@app.get("/health")
async def health_check():
    q = globals().get("queue") or app.state.queue
    return {
        "status": "healthy",
        "version": app.version,
        "uptime": time.time() - q.start_time,
        "queue_length": q.length(),
    }

@app.get("/info")
async def info():
    return {
        "service": "CallAnnotate",
        "version": app.version,
        "supported_formats": list(SUPPORTED_FORMATS),
        "max_file_size": int(os.getenv("MAX_FILE_SIZE", "1073741824")),
        "processing_mode": "asynchronous",
        "api_endpoints": {"rest": "/api/v1"},
    }

@app.post("/jobs", status_code=201, response_model=JobCreateResponse)
async def create_job(file: UploadFile = File(...)):
    max_size = int(os.getenv("MAX_FILE_SIZE", "1073741824"))
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(status_code=413, detail="File too large")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    job_id = str(uuid.uuid4())
    q = globals().get("queue") or app.state.queue
    incoming_dir = q.incoming_dir(job_id)
    os.makedirs(incoming_dir, exist_ok=True)
    filepath = os.path.join(incoming_dir, file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)

    q.enqueue(job_id, filepath)
    meta = q.get(job_id)
    return JobCreateResponse(
        job_id=job_id,
        status="queued",
        created_at=meta["created_at"],
        progress_url=f"/api/v1/jobs/{job_id}",
        result_url=f"/api/v1/jobs/{job_id}/result",
    )

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    q = globals().get("queue") or app.state.queue
    job = q.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    q = globals().get("queue") or app.state.queue
    job = q.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    result_path = q.result_file(job_id)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(result_path, media_type="application/json")

@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    q = globals().get("queue") or app.state.queue
    if not q.delete(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return Response(status_code=204)
