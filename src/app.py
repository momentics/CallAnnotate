# src/app.py
# Автор: akoodoy@ilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import os
import uuid
import shutil
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from queue_manager import QueueManager, JobStatus
from starlette.middleware.cors import CORSMiddleware

# Константы из конфигурации
VOLUME_PATH = os.getenv("VOLUME_PATH", "/app/volume")
SUPPORTED_FORMATS = {"wav", "mp3", "flac", "opus"}

app = FastAPI(
    title="CallAnnotate API",
    version="1.0.0",
    openapi_prefix="/api/v1",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

queue = QueueManager(volume_path=VOLUME_PATH)

class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    progress_url: str
    result_url: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": app.version,
        "uptime": time.time() - queue.start_time,
        "queue_length": queue.length(),
    }

@app.get("/info")
async def info():
    return {
        "service": "CallAnnotate",
        "version": app.version,
        "supported_formats": list(SUPPORTED_FORMATS),
        "max_file_size": int(os.getenv("MAX_FILE_SIZE", "1073741824")),
        "processing_mode": "asynchronous",
        "api_endpoints": {
            "rest": "/api/v1",
        },
    }

@app.post("/jobs", status_code=201, response_model=JobCreateResponse)
async def create_job(file: UploadFile = File(...)):
    # Динамически считываем ограничение из окружения
    max_size = int(os.getenv("MAX_FILE_SIZE", "1073741824"))
    contents = await file.read()
    size = len(contents)
    if size > max_size:
        raise HTTPException(status_code=413, detail="File too large")
    # Проверка формата
    ext = file.filename.split(".")[-1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=415, detail="Unsupported media type")
    # Сохранение файла во входящую папку
    job_id = str(uuid.uuid4())
    incoming_dir = queue.incoming_dir(job_id)
    os.makedirs(incoming_dir, exist_ok=True)
    filepath = os.path.join(incoming_dir, file.filename)
    with open(filepath, "wb") as f:
        f.write(contents)
    # Создание задания
    queue.enqueue(job_id, filepath)
    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.QUEUED.value,
        created_at=queue.jobs[job_id]["created_at"],
        progress_url=f"/api/v1/jobs/{job_id}",
        result_url=f"/api/v1/jobs/{job_id}/result",
    )

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    job = queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")
    # Отдаём файл результата
    result_path = queue.result_file(job_id)
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(result_path, media_type="application/json")

@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    if not queue.delete(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return Response(status_code=204)

# запуск приложения: uvicorn src.app:app --host 0.0.0.0 --port 8000
