# Спецификация REST/JSON и WebSocket API для Docker Image CallAnnotate

## Метаданные документа

- **Автор:** akoodoy@capilot.ru
- **Лицензия:** Apache-2.0
- **Версия API:** 1.0.0
- **Проект:** https://github.com/momentics/CallAnnotate
- **План:** /docs/development-phases.md


## 1. Обзор системы

CallAnnotate представляет собой контейнеризованный сервис автоматической аннотации аудиофайлов.
Асинхронная модель обработки с этапами: preprocess → diarization → transcription → recognition → carddav.

## 2. Технические требования

- CPU: 4 vCPU
- RAM: 3 GB
- Диск: 30 GB NVMe

| Параметр | Значение |
| :-- | :-- |
| Максимальный размер файла | 1 GB |
| Одновременная обработка | до 2 задач |
| Поддерживаемые форматы | wav, mp3, flac, ogg, aac, m4a, mp4 |
| Порт контейнера | 8000 |



## 3. REST/JSON API

**Base URL:** `http://localhost:8000/api/v1`
**Content-Type:** `application/json`
**Аутентификация:** Не требуется

### 3.1 Пример POST /jobs

**Request:**

```json
{
  "filename": "recording.wav",
  "priority": 5
}
```

**Response (201 Created):**

```json
{
  "job_id": "uuid-v4",
  "status": "queued",
  "message": "queued",
  "created_at": "2025-07-30T03:00:00Z",
  "file_info": {
    "filename": "recording.wav",
    "path": "/volume/incoming/recording.wav"
  },
  "progress_url": "/api/v1/jobs/{job_id}",
  "result_url": "/api/v1/jobs/{job_id}/result"
}
```


### 3.2 GET /jobs/{job_id}

Получение статуса задачи:

```json
{
  "job_id": "...",
  "status": "processing",
  "message": "processing",
  "progress": {
    "percentage": 65,
    "current_stage": "transcription"
  },
  "timestamps": {
    "created_at": "2025-07-30T03:00:00Z",
    "started_at": "2025-07-30T03:00:05Z",
    "completed_at": null
  }
}
```


### 3.3 GET /jobs/{job_id}/result

Получение полного JSON-результата аннотации.
В разделе `transcription` возвращаются:

- `full_text`
- `confidence`
- `language`
- `segments` (с полями `start`, `end`, `text`, `speaker`, `speaker_confidence`, `no_speech_prob`, `avg_logprob`)
- `words` (с полями `start`, `end`, `word`, `probability`, `speaker`)

**Пример ответа:**

```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "version": "1.0.0",
  "created_at": "2025-07-30T03:00:10Z",
  "audio_metadata": { /* ... */ },
  "processing_info": { /* ... */ },
  "speakers": [ /* ... */ ],
  "segments": [ /* ... */ ],
  "transcription": {
    "full_text": "[speaker_01]: Здравствуйте!",
    "confidence": 0.93,
    "language": "ru",
    "segments": [
      {
        "start": 0.000,
        "end": 3.500,
        "text": "Здравствуйте!",
        "speaker": "speaker_01",
        "speaker_confidence": 1.000,
        "no_speech_prob": 0.001,
        "avg_logprob": -0.25
      }
    ],
    "words": [
      {
        "start": 0.000,
        "end": 0.500,
        "word": "Здравствуйте!",
        "probability": 0.98,
        "speaker": "speaker_01"
      }
    ]
  },
  "statistics": { /* ... */ }
}
```


## 4. WebSocket API

**URL:** `ws://localhost:8000/ws/{client_id}`
Типы сообщений: `ping`/`pong`, `create_job`, `job_created`, `subscribe_job`, `subscribed`, `status_update`, `result`, `error`.


## 5. Результат транскрипции

### Поле `transcription`

| Параметр | Описание |
| :-- | :-- |
| full_text | Полный текст с разметкой спикеров |
| confidence | Средняя уверенность транскрипции (0.0–1.0) |
| language | Определённый язык транскрипции |
| segments | Список сегментов с полями: start, end, text, speaker, speaker_confidence, no_speech_prob, avg_logprob |
| words | Детализированные слова с полями: start, end, word, probability, speaker |

```json
"transcription": {
  "full_text": "[speaker_01]: Добрый день!",
  "confidence": 0.95,
  "language": "ru",
  "segments": [ /* ... */ ],
  "words": [ /* ... */ ]
}
```

См. [docs/diarization-example.json](../docs/diarization-example.json).

### OpenAPI-схема

Используйте [docs/diarization-schema.json](../docs/diarization-schema.json) для валидации.

## 6. Результат транскрипции

### Пример результата

```json
{
  "transcription": {
    "full_text": "...",
    "confidence": 0.93,
    "language": "ru",
    "segments": [...],
    "words": [...]
  }
}
```

См. [docs/transcription-example.json](../docs/transcription-example.json).

### OpenAPI-схема

Используйте [docs/transcription-schema.json](../docs/transcription-schema.json).

## 7. Безопасность и надежность

- Максимальный размер: 1 GB
- Rate Limiting: 60 запросов/мин
- Проверка MIME, санитизация путей


## 8. Мониторинг и логирование

Логи в JSON в `volume/logs/system/app.log`, health-запрос возвращает статус компонентов.

## 9. Примеры использования

### REST (cURL)

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -F "file=@recording.wav"
curl http://localhost:8000/api/v1/jobs/{job_id}
curl http://localhost:8000/api/v1/jobs/{job_id}/result
curl -X DELETE http://localhost:8000/api/v1/jobs/{job_id}
```


### WebSocket (JavaScript)

```js
socket.send(JSON.stringify({type:"create_job",filename:"rec.wav"}));
```


### Python SDK

SDK доступны на PyPI и npm, генерируются из OpenAPI.
