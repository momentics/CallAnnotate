# CallAnnotate

Полнофункциональное решение для автоматической аннотации телефонных разговоров в контейнере Docker под python:3.11-slim-bullseye.

CallAnnotate предоставляет разработчикам и системным администраторам гибкий серверный компонент с WebSocket и REST/JSON API, способный асинхронно обрабатывать аудиозаписи любой длительности и сложности, используя ресурсы CPU или при необходимости GPU.

<!-- Badges -->
[![CI Status](https://github.com/momentics/CallAnnotate/actions/workflows/ci.yml/badge.svg)](https://github.com/moment/CallAnnotate/actions/workflows/ci.yml)   [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

[![Latest Release](https://img.shields.io/github/v/release/momentics/CallAnnotate?style=for-the-badge)](https://github.com/momentics/CallAnnotate/releases)   [![Last Commit](https://img.shields.io/github/last-commit/momentics/CallAnnotate?style=for-the-badge)](https://github.com/momentics/CallAnnotate/commits/main)


## Требования

1. Требуемая версия Python: 3.11

## Основные возможности

1. **Docker-образ на базе python:3.11-slim-bullseye**
CallAnnotate создаёт лёгкий и безопасный контейнер, готовый к развёртыванию на любых серверах и облачных платформах, поддерживающих Docker. 
2. **Универсальные интерфейсы API**
    - **WebSocket** позволяет устанавливать постоянное соединение для потоковой передачи аудио и мгновенного получения результатов.
    - **REST/JSON** интерфейс предназначен для пакетной обработки задач: отправьте запрос с файлом, получите полную аннотацию в ответе.
3. **Асинхронная архитектура обработки**
Все задачи помещаются в очередь и обрабатываются асинхронно, что исключает блокировку сервера. CallAnnotate может распределять нагрузку между ядрами CPU и, при наличии поддерживаемого оборудования, GPU-ускорителем.
4. **Этапная архитектура обработки аудио** (preprocess → diarization → transcription → recognition → carddav)  
5. **Предобработка аудио** (SoX, RNNoise, DeepFilterNet)  
4. **Диаризация говорящих**
Алгоритмы выделения сегментов аудио автоматически распознают смены говорящих, разделяют файл на спикер-ориентированные блоки и сохраняют таймкоды каждого участка.
5. **Распознавание известных голосов**
Согласно конфигурации контейнера, CallAnnotate загружает заранее сформированные голосовые эмбеддинги и сопоставляет их с поступающими фрагментами. Это позволяет автоматически отмечать в аннотации имена участников, чьи голоса уже известны системе.
6. **Идентификация новых записей через CardDAV**
При встрече незнакомого голоса CallAnnotate выполняет запросы к CardDAV-серверу, чтобы найти совпадения по контактам пользователя. Если совпадений не обнаружено, система аккуратно помечает фрагмент как «неизвестный спикер».
7. **Транскрипция с многослойной разметкой**
Используя современные движки ASR, CallAnnotate переводит аудио в текст и структурирует результат по уровням (схоже с системой ELAN):
    - отдельное слово и его начало/конец
    - предложения и абзацы
    - метки спикеров
8. **Готовая аннотация в одном JSON**
По окончании обработки API возвращает полный объём метаданных:
    - сегментация по спикерам и таймкоды
    - идентификаторы и имена найденных голосов
    - ссылки на источники эмбеддингов
    - транскрипция с выделением уровней
    - при необходимости — URL для скачивания аудиофрагментов



## Секция preprocess

В `config/default.yaml` в разделе `preprocess`:
```

preprocess:
model: "DeepFilterNet2"
device: "cpu"
chunk_duration: 2.0
overlap: 0.5
target_rms: -20.0

```

## Секция транскрипции

Этап **transcription** осуществляет пакетную транскрипцию аудио с помощью **OpenAI Whisper**.  

Конфигурация этапа в `config/default.yaml`:
```

transcription:
model: "openai/whisper-small"   \# размер модели Whisper
device: "cpu"                   \# вычислительное устройство
language: "ru"                  \# язык транскрипции ("auto" — автоопределение)
batch_size: 16                  \# размер пакета фрагментов
task: "transcribe"              \# "transcribe" или "translate"

```

### Пример использования

```

curl -X POST http://localhost:8000/api/v1/jobs \
-F "file=@recording.wav"

```

При получении результата через
`GET /api/v1/jobs/{job_id}/result` возвращается JSON, содержащий, среди прочего, поле `transcription`:
```

"transcription": {
"full_text": "[speaker_01]: Здравствуйте!\n[speaker_02]: Привет!",
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
}

```

## Основные этапы

1. preprocess  
2. diarization  
3. transcription  
4. recognition  
5. carddav  


## Лицензия

Проект распространяется под лицензией Apache License 2.0. Подробности — в файлах [LICENSE](LICENSE) и [NOTICE](NOTICE).
