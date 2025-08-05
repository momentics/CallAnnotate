# Dockerfile для CallAnnotate с предзагрузкой моделей Whisper и Pyannote
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

FROM python:3.11-slim-bullseye as builder

ARG WHISPER_MODEL=small
ARG DIAR_MODEL=pyannote/speaker-diarization-3.1
ARG RECOG_MODEL=speechbrain/spkrec-ecapa-voxceleb
ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="akoodoy@capilot.ru" \
      version="1.0.0" \
      description="CallAnnotate - автоматическая аннотация аудио (offline)"

RUN apt-get update \
    && apt-get install -y libsndfile1-dev ffmpeg sox curl autoconf libtool build-essential git cargo rustc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Установка RNNoise
RUN git clone https://github.com/xiph/rnnoise.git /tmp/rnnoise \
    && cd /tmp/rnnoise \
    && export CFLAGS="-march=native -O3" \
    && ./autogen.sh \
    && ./configure \
    && make \
    && make install \
    && ldconfig \
    && rm -rf /tmp/rnnoise

# Создаём пользователя
RUN groupadd --gid 1000 callannotate \
    && useradd --uid 1000 --gid callannotate --shell /bin/bash --create-home callannotate

WORKDIR /app

COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY scripts/preload_whisper.py /app/preload_whisper.py
COPY scripts/preload_hf_models.py /app/preload_hf_models.py

# Предзагрузка моделей в volume/models/cache
RUN mkdir -p /volume/models/cache \
    && mkdir -p /volume/models/diarization \
    && mkdir -p /volume/models/recognition \
    && chown -R callannotate:callannotate /volume/models

# Устанавливаем переменные для оффлайн-кеша
ENV HF_HOME=/volume/models/cache \
    TRANSFORMERS_CACHE=/volume/models/cache \
    TORCH_HOME=/volume/models/cache

# Предзагружаем Whisper
RUN echo "Предзагрузка Whisper модели: $WHISPER_MODEL" \
    && python /app/preload_whisper.py --model $WHISPER_MODEL --cache-dir /volume/models/cache

# Предзагрузка Pyannote и SpeechBrain моделей
RUN python /app/preload_hf_models.py \
      --cache-dir /volume/models/cache \
      $DIAR_MODEL $RECOG_MODEL

COPY src/ /app/src/
COPY config/ /app/config/

RUN chown -R callannotate:callannotate /app \
    && chmod +x /app/preload_whisper.py /app/preload_hf_models.sh

USER callannotate

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=/app/config/default.yaml \
    VOLUME_PATH=/volume

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
