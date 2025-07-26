# tests/test_queue_manager.py
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

import os
import json
import shutil
import threading
import time
from enum import Enum

class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class QueueManager:
    def __init__(self, volume_path: str):
        self.volume = volume_path
        self._lock = threading.Lock()
        self.jobs = {}  # job_id -> metadata dict
        self.start_time = time.time()
        # Создать папки
        os.makedirs(self.volume, exist_ok=True)

    def incoming_dir(self, job_id: str) -> str:
        return os.path.join(self.volume, job_id, "incoming")

    def processing_dir(self, job_id: str) -> str:
        return os.path.join(self.volume, job_id, "processing")

    def completed_dir(self, job_id: str) -> str:
        return os.path.join(self.volume, job_id, "completed")

    def result_file(self, job_id: str) -> str:
        return os.path.join(self.completed_dir(job_id), "result.json")

    def enqueue(self, job_id: str, filepath: str):
        with self._lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.QUEUED.value,
                "created_at": time.time(),
                "updated_at": time.time(),
                "file_info": {
                    "filename": os.path.basename(filepath),
                    "size": os.path.getsize(filepath),
                },
                "progress": {"percentage": 0},
            }
        # Запустить обработчик в фоне
        thread = threading.Thread(target=self._worker, args=(job_id, filepath), daemon=True)
        thread.start()

    def _worker(self, job_id: str, filepath: str):
        try:
            with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = JobStatus.PROCESSING.value
                    self.jobs[job_id]["updated_at"] = time.time()
                else:
                    return  # Задача была удалена
            
            # Переместить файл в processing
            proc_dir = self.processing_dir(job_id)
            os.makedirs(proc_dir, exist_ok=True)
            dest = os.path.join(proc_dir, os.path.basename(filepath))
            
            if os.path.exists(filepath):
                shutil.move(filepath, dest)
            
            # TODO: здесь будут вызовы diarization, transcription и т.д.
            time.sleep(2)  # заглушка обработки
            
            result = {
                "job_id": job_id,
                "speakers": [],
                "metadata": {"processing_time": 2},
            }
            
            # Запись результата
            comp_dir = self.completed_dir(job_id)
            os.makedirs(comp_dir, exist_ok=True)
            with open(self.result_file(job_id), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = JobStatus.COMPLETED.value
                    self.jobs[job_id]["updated_at"] = time.time()
                    self.jobs[job_id]["result"] = result
                    
        except Exception as e:
            with self._lock:
                if job_id in self.jobs:
                    self.jobs[job_id]["status"] = JobStatus.FAILED.value
                    self.jobs[job_id]["updated_at"] = time.time()
                    self.jobs[job_id]["error"] = str(e)

    def get(self, job_id: str):
        with self._lock:
            return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self.jobs:
                return False
            del self.jobs[job_id]
        
        # Удалить папку с файлами
        path = os.path.join(self.volume, job_id)
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        return True

    def length(self) -> int:
        with self._lock:
            return len(self.jobs)
