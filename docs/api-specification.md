# Спецификация REST/JSON и WebSocket API для Docker Image CallAnnotate

## Метаданные документа

**Автор:** akoodoy@capilot.ru
**Проект:** https://github.com/momentics/CallAnnotate
**Лицензия:** Apache-2.0
**Версия API:** 1.0.0
**Дата создания:** 26 июля 2025
**Место хранения:** `docs/api-specification.md`

## 1. Обзор системы

### 1.1 Архитектурная концепция

CallAnnotate представляет собой контейнеризованный сервис автоматической аннотации аудиофайлов, работающий на базе образа `python:3.11-slim-bullseye`. Система реализует **асинхронную модель обработки одного файла за раз** с использованием внешнего подключенного volume `/volume` для входящих файлов.

### 1.2 Ключевые принципы проектирования

1. **Функциональная эквивалентность API**: REST и WebSocket интерфейсы предоставляют идентичную функциональность
2. **Асинхронная обработиент получает JobID и периодически опрашивает статус выполнения
3. **Отказоустойчивость**: Защита от сбоев, перезагрузок и автоматическая очистка повисших файлов
4. **Масштабируемость**: Обработка файлов до 1 гигабайта с ротацией логов

## 2. Технические требования

### 2.1 Системные ограничения

| Параметр | Значение | Обоснование |
| :-- | :-- | :-- |
| Максимальный размер файла | 1 ГБ | Предотвращение DoS-атак |
| Одновременная обработка | 1 файл | Оптимизация ресурсов CPU/RAM |
| Поддерживаемые форматы | WAV, MP3, FLAC, OPUS | Современные аудиокодеки |
| Порт контейнера | 8000 | Стандартный порт для HTTP/WebSocket |

### 2.2 Инфраструктурные зависимости

Контейнер использует следующие системные зависимости:

- `libsndfile1-dev` - обработка аудиофайлов
- `portaudio19-dev` - аудио I/O
- `ffmpeg` - декодирование форматов


## 3. REST/JSON API

### 3.1 Базовая информация

**Base URL:** `http://localhost:8000/api/v1`
**Content-Type:** `application/json`
**Аутентификация:** Не требуется (анонимный доступ)

### 3.2 HTTP статус-коды

В соответствии с лучшими практиками REST API дизайна, система использует стандартные HTTP коды:

#### Успешные ответы (2xx)

- `200 OK` - Успешное получение данных
- `201 Created` - Задача создана в очереди
- `202 Accepted` - Запрос принят к асинхронной обработке
- `204 No Content` - Успешное удаление без содержимого


#### Клиентские ошибки (4xx)

- `400 Bad Request` - Некорректный формат запроса
- `404 Not Found` - Ресурс не найден
- `413 Payload Too Large` - Размер файла превышает 1 ГБ
- `415 Unsupported Media Type` - Неподдерживаемый формат файла
- `429 Too Many Requests` - Превышение лимита запросов


#### Серверные ошибки (5xx)

- `500 Internal Server Error` - Внутренняя ошибка сервера
- `503 Service Unavailable` - Сервис временно недоступен


### 3.3 Основные эндпоинты

#### 3.3.1 Информация о сервисе

**GET /health**

```json
{
  "status": "healthy",
  "version": "1.0.0", 
  "uptime": 3600,
  "queue_status": "active"
}
```

**GET /info**

```json
{
  "service": "CallAnnotate",
  "version": "1.0.0",
  "supported_formats": ["wav", "mp3", "flac", "opus"],
  "max_file_size": 1073741824,
  "processing_mode": "asynchronous",
  "api_endpoints": {
    "rest": "/api/v1",
    "websocket": "/ws"
  }
}
```


#### 3.3.2 Управление задачами обработки

**POST /jobs**

Создание новой задачи обработки аудиофайла.

*Request:*

```http
POST /api/v1/jobs
Content-Type: multipart/form-data

file=@audio.wav
metadata={"description": "Phone call recording", "priority": 1}
```

*Response (201 Created):*

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "created_at": "2025-07-26T18:00:00Z",
  "estimated_duration": 300,
  "file_info": {
    "filename": "audio.wav",
    "size": 1048576,
    "format": "wav",
    "duration": 120.5
  },
  "progress_url": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000",
  "websocket_url": "/ws/550e8400-e29b-41d4-a716-446655440000"
}
```

**GET /jobs/{job_id}**

Получение статуса задачи с детальной информацией о прогрессе.

*Response (200 OK):*

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": {
    "percentage": 45,
    "current_stage": "diarization",
    "stages_completed": ["validation", "preprocessing"],
    "stages_remaining": ["transcription", "recognition", "annotation"]
  },
  "created_at": "2025-07-26T18:00:00Z",
  "started_at": "2025-07-26T18:01:00Z",
  "estimated_completion": "2025-07-26T18:06:00Z",
  "file_info": {
    "filename": "audio.wav",
    "size": 1048576,
    "format": "wav",
    "duration": 120.5
  }
}
```


#### 3.3.3 Статусы задач

Система поддерживает следующие статусы задач:


| Статус | Описание | Следующий статус |
| :-- | :-- | :-- |
| `queued` | Задача в очереди на обработку | `processing` |
| `processing` | Активная обработка файла | `completed`, `failed` |
| `completed` | Обработка завершена успешно | `archived` |
| `failed` | Обработка завершена с ошибкой | `archived` |
| `archived` | Результаты удалены из системы | - |

**GET /jobs/{job_id}/result**

Получение результатов обработки для завершенной задачи.

*Response (200 OK):*

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "completed_at": "2025-07-26T18:05:30Z",
  "processing_time": 330,
  "result": {
    "speakers": [
      {
        "speaker_id": "speaker_1",
        "identified_as": "Иван Петров",
        "confidence": 0.92,
        "voice_embedding_source": "/app/embeddings/ivan.vec",
        "segments": [
          {
            "start_time": 0.0,
            "end_time": 15.2,
            "text": "Добро пожаловать в нашу компанию",
            "confidence": 0.89
          }
        ]
      },
      {
        "speaker_id": "speaker_2", 
        "identified_as": "unknown",
        "confidence": 0.0,
        "carddav_search_performed": true,
        "segments": [
          {
            "start_time": 15.3,
            "end_time": 28.7,
            "text": "Спасибо за приглашение на интервью",
            "confidence": 0.94
          }
        ]
      }
    ],
    "metadata": {
      "total_duration": 120.5,
      "speaker_count": 2,
      "word_count": 156,
      "confidence_average": 0.91,
      "processing_stages": {
        "diarization": "completed",
        "transcription": "completed", 
        "recognition": "completed",
        "carddav_lookup": "completed",
        "annotation": "completed"
      }
    }
  }
}
```

**DELETE /jobs/{job_id}**

Удаление задачи и связанных с ней результатов.

*Response (204 No Content):*

```
HTTP/1.1 204 No Content
```


#### 3.3.4 Управление очередью

**GET /queue/status**

Получение общего состояния очереди обработки.

*Response (200 OK):*

```json
{
  "queue_length": 3,
  "processing_job": "550e8400-e29b-41d4-a716-446655440000",
  "average_processing_time": 280,
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440001",
      "status": "queued",
      "position": 1,
      "estimated_start": "2025-07-26T18:10:00Z"
    },
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440002", 
      "status": "queued",
      "position": 2,
      "estimated_start": "2025-07-26T18:15:00Z"
    }
  ]
}
```


### 3.4 Обработка ошибок

В соответствии с принципами надежного API дизайна, все ошибки возвращаются в стандартизированном формате:

```json
{
  "error": {
    "code": "FILE_TOO_LARGE",
    "message": "Размер файла превышает максимально допустимый (1 ГБ)",
    "details": {
      "file_size": 1073741825,
      "max_allowed": 1073741824
    },
    "timestamp": "2025-07-26T18:00:00Z",
    "request_id": "req_550e8400-e29b-41d4-a716-446655440000"
  }
}
```


## 4. WebSocket API

### 4.1 Соединение и аутентификация

**URL подключения:** `ws://localhost:8000/ws/{job_id}`
**Протокол:** WebSocket RFC 6455
**Аутентификация:** Не требуется (анонимный доступ)

### 4.2 Жизненный цикл соединения

В соответствии с лучшими практиками WebSocket архитектуры, соединение поддерживает следующие паттерны:

#### 4.2.1 Установка соединения

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/550e8400-e29b-41d4-a716-446655440000');

socket.onopen = function(event) {
    console.log('WebSocket connected');
    // Соединение установлено
};
```


#### 4.2.2 Обработка сообщений

```javascript
socket.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};
```


### 4.3 Типы сообщений

#### 4.3.1 Статус задачи (Server → Client)

```json
{
  "type": "status_update",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": {
    "percentage": 45,
    "current_stage": "diarization",
    "stage_progress": 78
  },
  "timestamp": "2025-07-26T18:03:00Z"
}
```


#### 4.3.2 Результат обработки (Server → Client)

```json
{
  "type": "result",
  "job_id": "550e8400-e29b-41d4-a716-446655440000", 
  "status": "completed",
  "result": {
    "speakers": [...],
    "metadata": {...}
  },
  "timestamp": "2025-07-26T18:05:30Z"
}
```


#### 4.3.3 Ошибка обработки (Server → Client)

```json
{
  "type": "error",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "error": {
    "code": "PROCESSING_FAILED",
    "message": "Ошибка при диаризации аудиофайла",
    "stage": "diarization"
  },
  "timestamp": "2025-07-26T18:03:30Z"
}
```


#### 4.3.4 Управляющие команды (Client → Server)

```json
{
  "type": "ping",
  "timestamp": "2025-07-26T18:04:00Z"
}
```

*Response:*

```json
{
  "type": "pong", 
  "timestamp": "2025-07-26T18:04:00Z"
}
```


### 4.4 Обработка отключений

В соответствии с рекомендациями по надежности WebSocket соединений, система реализует:

- **Автоматическое переподключение** при разрыве связи
- **Heartbeat механизм** (ping/pong каждые 30 секунд)
- **Буферизация сообщений** при временной недоступности
- **Graceful shutdown** при завершении обработки


## 5. Модель данных

### 5.1 Структура задачи (Job)

```yaml
Job:
  type: object
  required: [job_id, status, created_at, file_info]
  properties:
    job_id:
      type: string
      format: uuid
      description: Уникальный идентификатор задачи
    status:
      type: string
      enum: [queued, processing, completed, failed, archived]
    created_at:
      type: string
      format: date-time
    started_at:
      type: string
      format: date-time
      nullable: true
    completed_at:
      type: string
      format: date-time
      nullable: true
    progress:
      $ref: '#/components/schemas/Progress'
    file_info:
      $ref: '#/components/schemas/FileInfo'
    result:
      $ref: '#/components/schemas/Result'
      nullable: true
```


### 5.2 Информация о файле (FileInfo)

```yaml
FileInfo:
  type: object
  required: [filename, size, format]
  properties:
    filename:
      type: string
      description: Имя исходного файла
    size:
      type: integer
      minimum: 1
      maximum: 1073741824
      description: Размер файла в байтах
    format:
      type: string
      enum: [wav, mp3, flac, opus]
    duration:
      type: number
      minimum: 0
      description: Длительность в секундах
    sample_rate:
      type: integer
      description: Частота дискретизации
    channels:
      type: integer
      description: Количество каналов
```


### 5.3 Прогресс обработки (Progress)

```yaml
Progress:
  type: object
  properties:
    percentage:
      type: integer
      minimum: 0
      maximum: 100
    current_stage:
      type: string
      enum: [validation, preprocessing, diarization, transcription, recognition, carddav_lookup, annotation]
    stage_progress:
      type: integer
      minimum: 0
      maximum: 100
      description: Прогресс текущего этапа
    stages_completed:
      type: array
      items:
        type: string
    stages_remaining:
      type: array
      items:
        type: string
    estimated_completion:
      type: string
      format: date-time
```


### 5.4 Результат обработки (Result)

```yaml
Result:
  type: object
  required: [speakers, metadata]
  properties:
    speakers:
      type: array
      items:
        $ref: '#/components/schemas/Speaker'
    metadata:
      $ref: '#/components/schemas/Metadata'

Speaker:
  type: object
  required: [speaker_id, confidence, segments]
  properties:
    speaker_id:
      type: string
      description: Идентификатор спикера
    identified_as:
      type: string
      nullable: true
      description: Имя идентифицированного спикера
    confidence:
      type: number
      minimum: 0
      maximum: 1
      description: Уверенность в идентификации
    voice_embedding_source:
      type: string
      nullable: true
      description: Путь к файлу эмбеддингов
    carddav_search_performed:
      type: boolean
      description: Выполнялся ли поиск в CardDAV
    segments:
      type: array
      items:
        $ref: '#/components/schemas/Segment'

Segment:
  type: object
  required: [start_time, end_time, text, confidence]
  properties:
    start_time:
      type: number
      minimum: 0
      description: Время начала в секундах
    end_time:
      type: number
      minimum: 0
      description: Время окончания в секундах
    text:
      type: string
      description: Транскрибированный текст
    confidence:
      type: number
      minimum: 0
      maximum: 1
      description: Уверенность в транскрипции
    words:
      type: array
      items:
        $ref: '#/components/schemas/Word'

Word:
  type: object
  required: [word, start_time, end_time, confidence]
  properties:
    word:
      type: string
    start_time:
      type: number
      minimum: 0
    end_time:
      type: number
      minimum: 0  
    confidence:
      type: number
      minimum: 0
      maximum: 1
```


## 6. Безопасность и надежность

### 6.1 Защита от атак

В соответствии с принципами безопасного API дизайна:

#### 6.1.1 Ограничения размера файлов

- Максимальный размер: 1 ГБ
- Проверка MIME-типов и расширений
- Валидация структуры аудиофайлов


#### 6.1.2 Rate Limiting

```http
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 8
X-RateLimit-Reset: 1627321200
```


#### 6.1.3 Санитизация входных данных

- Фильтрация имен файлов от path traversal
- Валидация JSON полей
- Ограничение длины строковых параметров


### 6.2 Мониторинг и логирование

Система ведет структурированные логи в формате JSON:

```json
{
  "timestamp": "2025-07-26T18:00:00Z",
  "level": "INFO",
  "service": "callannotate",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "event": "job_started",
  "file_info": {
    "filename": "audio.wav",
    "size": 1048576
  },
  "request_id": "req_550e8400-e29b-41d4-a716-446655440000"
}
```


### 6.3 Восстановление после сбоев

Система реализует механизмы отказоустойчивости:

1. **Автоматическое восстановление** повисших задач при запуске
2. **Откат файлов** из processing в incoming при сбое
3. **Ротация логов** для предотвращения переполнения диска
4. **Health checks** для мониторинга состояния

## 7. Развертывание и эксплуатация

### 7.1 Docker Compose конфигурация

```yaml
version: '3.8'
services:
  callannotate:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: callannotate
    ports:
      - "8000:8000"
    volumes:
      - ./volume:/app/volume
      - ./config:/app/config:ro
    environment:
      - PYTHONPATH=/app/src
      - CONFIG_PATH=/app/config/default.yaml
      - VOLUME_PATH=/app/volume
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```


### 7.2 Переменные окружения

| Переменная | Значение по умолчанию | Описание |
| :-- | :-- | :-- |
| `CONFIG_PATH` | `/app/config/default.yaml` | Путь к конфигурационному файлу |
| `VOLUME_PATH` | `/app/volume` | Путь к volume с файлами |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `MAX_FILE_SIZE` | `1073741824` | Максимальный размер файла |
| `QUEUE_POLL_INTERVAL` | `5` | Интервал проверки очереди (сек) |

### 7.3 Health Check

Эндпоинт `/health` возвращает статус компонентов системы:

```json
{
  "status": "healthy",
  "components": {
    "queue_manager": "healthy",
    "volume_access": "healthy",
    "audio_processing": "healthy",
    "memory_usage": {
      "used": "512MB",
      "available": "1536MB",
      "percentage": 25
    },
    "disk_usage": {
      "volume_free": "50GB",
      "logs_size": "100MB"
    }
  },
  "uptime": 3600,
  "version": "1.0.0"
}
```


## 8. Примеры использования

### 8.1 Полный цикл REST API

```bash
# 1. Отправка файла на обработку
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "file=@recording.wav" \
  -F 'metadata={"description":"Interview recording"}'

# Response: {"job_id": "550e8400-e29b-41d4-a716-446655440000", ...}

# 2. Проверка статуса
curl http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000

# 3. Получение результата
curl http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result

# 4. Удаление задачи
curl -X DELETE http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000
```


### 8.2 WebSocket клиент (JavaScript)

```javascript
class CallAnnotateClient {
  constructor(jobId) {
    this.jobId = jobId;
    this.socket = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    this.socket = new WebSocket(`ws://localhost:8000/ws/${this.jobId}`);
    
    this.socket.onopen = () => {
      console.log('Connected to CallAnnotate');
      this.reconnectAttempts = 0;
      this.startHeartbeat();
    };

    this.socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };

    this.socket.onclose = () => {
      console.log('Disconnected from CallAnnotate');
      this.stopHeartbeat();
      this.attemptReconnect();
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  handleMessage(message) {
    switch (message.type) {
      case 'status_update':
        this.onProgress(message.progress);
        break;
      case 'result':
        this.onComplete(message.result);
        break;
      case 'error':
        this.onError(message.error);
        break;
      case 'pong':
        // Heartbeat response
        break;
    }
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({
          type: 'ping',
          timestamp: new Date().toISOString()
        }));
      }
    }, 30000);
  }

  onProgress(progress) {
    console.log(`Progress: ${progress.percentage}% - ${progress.current_stage}`);
  }

  onComplete(result) {
    console.log('Processing completed:', result);
    this.socket.close();
  }

  onError(error) {
    console.error('Processing error:', error);
  }
}
```


### 8.3 Python клиент

```python
import requests
import websocket
import json
import time
import threading

class CallAnnotateClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        
    def submit_job(self, audio_file_path, metadata=None):
        """Отправка файла на обработку"""
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            data = {'metadata': json.dumps(metadata or {})}
            
            response = requests.post(
                f"{self.api_url}/jobs",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
    
    def get_job_status(self, job_id):
        """Получение статуса задачи"""
        response = requests.get(f"{self.api_url}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def get_result(self, job_id):
        """Получение результата"""
        response = requests.get(f"{self.api_url}/jobs/{job_id}/result")
        response.raise_for_status()
        return response.json()
    
    def poll_until_complete(self, job_id, poll_interval=5):
        """Polling до завершения задачи"""
        while True:
            status = self.get_job_status(job_id)
            print(f"Status: {status['status']}")
            
            if status['status'] in ['completed', 'failed']:
                break
                
            time.sleep(poll_interval)
        
        if status['status'] == 'completed':
            return self.get_result(job_id)
        else:
            raise Exception(f"Job failed: {status}")

# Использование
client = CallAnnotateClient()

# Отправка файла
job = client.submit_job("recording.wav", {"description": "Test recording"})
print(f"Job created: {job['job_id']}")

# Ожидание завершения
result = client.poll_until_complete(job['job_id'])
print("Result:", json.dumps(result, indent=2, ensure_ascii=False))
```


## 9. Тестирование и отладка

### 9.1 Коллекция Postman

Для удобства тестирования API рекомендуется создать коллекцию Postman со следующими запросами:

1. **Health Check** - `GET /health`
2. **Service Info** - `GET /info`
3. **Submit Job** - `POST /jobs` (с файлом)
4. **Job Status** - `GET /jobs/{{job_id}}`
5. **Job Result** - `GET /jobs/{{job_id}}/result`
6. **Delete Job** - `DELETE /jobs/{{job_id}}`
7. **Queue Status** - `GET /queue/status`

### 9.2 Автоматическое тестирование

```python
import pytest
import requests
import tempfile
import wave
import numpy as np

class TestCallAnnotateAPI:
    BASE_URL = "http://localhost:8000/api/v1"
    
    def create_test_audio(self, duration=5, sample_rate=16000):
        """Создание тестового аудиофайла"""
        samples = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        samples = (samples * 32767).astype(np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            with wave.open(f.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(samples.tobytes())
            return f.name
    
    def test_health_check(self):
        response = requests.get(f"{self.BASE_URL.replace('/api/v1', '')}/health")
        assert response.status_code == 200
        assert response.json()['status'] == 'healthy'
    
    def test_submit_job(self):
        audio_file = self.create_test_audio()
        
        with open(audio_file, 'rb') as f:
            response = requests.post(
                f"{self.BASE_URL}/jobs",
                files={'file': f},
                data={'metadata': '{"test": true}'}
            )
        
        assert response.status_code == 201
        job = response.json()
        assert 'job_id' in job
        assert job['status'] == 'queued'
        
        return job['job_id']
    
    def test_job_status(self):
        job_id = self.test_submit_job()
        
        response = requests.get(f"{self.BASE_URL}/jobs/{job_id}")
        assert response.status_code == 200
        
        status = response.json()
        assert status['job_id'] == job_id
        assert status['status'] in ['queued', 'processing', 'completed', 'failed']
```


## 10. Документация OpenAPI 3.0

В соответствии с лучшими практиками API документации, полная спецификация должна быть представлена в формате OpenAPI 3.0:

```yaml
openapi: 3.0.3
info:
  title: CallAnnotate API
  description: API для автоматической аннотации аудиофайлов с диاризацией и распознаванием голосов
  version: 1.0.0
  contact:
    name: CallAnnotate Support
    email: akoodoy@capilot.ru
    url: https://github.com/momentics/CallAnnotate
  license:
    name: Apache 2.0
    url: https://www.apache.org/licenses/LICENSE-2.0

servers:
  - url: http://localhost:8000/api/v1
    description: Development server

paths:
  /health:
    get:
      summary: Проверка состояния сервиса
      tags: [System]
      responses:
        '200':
          description: Сервис работает нормально
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthStatus'

  /jobs:
    post:
      summary: Создание новой задачи обработки
      tags: [Jobs]
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                  description: Аудиофайл для обработки
                metadata:
                  type: string
                  description: JSON метаданные
      responses:
        '201':
          description: Задача создана
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Job'

components:
  schemas:
    HealthStatus:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, unhealthy]
        version:
          type: string
        uptime:
          type: integer
        queue_status:
          type: string
    
    Job:
      type: object
      required: [job_id, status, created_at, file_info]
      properties:
        job_id:
          type: string
          format: uuid
        status:
          type: string
          enum: [queued, processing, completed, failed, archived]
        created_at:
          type: string
          format: date-time
        file_info:
          $ref: '#/components/schemas/FileInfo'
```


## 11. Заключение

Данная спецификация API предоставляет детальное описание внешних интерфейсов Docker Image CallAnnotate, обеспечивая:

1. **Функциональную эквивалентность** REST и WebSocket API
2. **Асинхронную модель обработки** с поддержкой polling и push-уведомлений
3. **Надежность и отказоустойчивость** системы
4. **Соответствие современным стандартам** API дизайна
5. **Полную документированность** для разработчиков

Спецификация готова для:

- Реализации серверной части на FastAPI
- Создания клиентских SDK и библиотек
- Автоматического тестирования и валидации
- Интеграции в CI/CD процессы

**Рекомендуемое местоположение в проекте:** `docs/api-specification.md`

Документ будет использоваться как основа для разработки серверных компонентов и последующей контейнеризации в Docker Image CallAnnotate.
