"""
Apache-2.0 License
Author: akoodoy@capilot.ru
Repository: https://github.com/momentics/CallAnnotate

Файловая очередь обработки аудиофайлов с восстановлением после сбоя.
Основано на рекомендациях файловых очередей из предыдущих обсуждений.
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class FileQueueManager:
    """Менеджер файловой очереди с атомарными операциями и восстановлением."""
    
    def __init__(self, volume_path: str):
        self.volume_path = Path(volume_path)
        self.queue_path = self.volume_path / "queue"
        self.state_file = self.volume_path / "config" / "queue_state.json"
        
        # Создаём необходимые директории
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Создание структуры папок очереди."""
        dirs = [
            self.queue_path / "incoming",
            self.queue_path / "processing", 
            self.queue_path / "completed",
            self.queue_path / "failed",
            self.queue_path / "archived",
            self.volume_path / "intermediate" / "diarization",
            self.volume_path / "intermediate" / "transcription",
            self.volume_path / "intermediate" / "recognition", 
            self.volume_path / "intermediate" / "carddav",
            self.volume_path / "outputs" / "pending",
            self.volume_path / "outputs" / "delivered",
            self.volume_path / "logs" / "tasks",
            self.volume_path / "logs" / "system",
            self.volume_path / "logs" / "queue"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def get_next_task(self) -> Optional[str]:
        """Получение следующей задачи из очереди incoming/."""
        incoming_path = self.queue_path / "incoming"
        
        for file_name in os.listdir(incoming_path):
            if file_name.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                return self._move_to_processing(file_name)
        
        return None
        
    def _move_to_processing(self, file_name: str) -> str:
        """Атомарное перемещение файла в processing/ с таймстампом."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processing_name = f"{file_name}_PROCESSING_{timestamp}"
        
        src = self.queue_path / "incoming" / file_name
        dst = self.queue_path / "processing" / processing_name
        
        os.rename(src, dst)
        logger.info(f"Moved to processing: {file_name} -> {processing_name}")
        
        return processing_name
        
    def mark_completed(self, processing_file: str, result_data: Dict[str, Any]):
        """Перемещение в completed/ после успешной обработки."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        completed_name = processing_file.replace("_PROCESSING_", f"_COMPLETED_{timestamp}_")
        
        src = self.queue_path / "processing" / processing_file
        dst = self.queue_path / "completed" / completed_name
        
        os.rename(src, dst)
        
        # Сохраняем результат обработки
        result_file = self.volume_path / "outputs" / "pending" / f"{completed_name}.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Task completed: {processing_file}")
        
    def mark_failed(self, processing_file: str, error: str):
        """Перемещение в failed/ при ошибке обработки."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_name = processing_file.replace("_PROCESSING_", f"_FAILED_{timestamp}_")
        
        src = self.queue_path / "processing" / processing_file
        dst = self.queue_path / "failed" / failed_name
        
        os.rename(src, dst)
        
        # Логируем ошибку
        error_log = self.volume_path / "logs" / "queue" / f"{failed_name}_error.log"
        with open(error_log, 'w') as f:
            f.write(f"Error: {error}\nTimestamp: {datetime.now()}\n")
            
        logger.error(f"Task failed: {processing_file} - {error}")
        
    def recover_processing_files(self):
        """Восстановление незавершённых задач при запуске."""
        processing_path = self.queue_path / "processing"
        
        for file_name in os.listdir(processing_path):
            if "_PROCESSING_" in file_name:
                # Возвращаем в incoming/ для повторной обработки
                original_name = file_name.split("_PROCESSING_")[0]
                retry_name = f"{original_name}_RETRY_{int(time.time())}"
                
                src = processing_path / file_name
                dst = self.queue_path / "incoming" / retry_name
                
                os.rename(src, dst)
                logger.warning(f"Recovered processing file: {file_name} -> {retry_name}")
