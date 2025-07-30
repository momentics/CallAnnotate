# Dockerfile для CallAnnotate с предзагрузкой Whisper моделей
# Автор: akoodoy@capilot.ru
# Ссылка: https://github.com/momentics/CallAnnotate
# Лицензия: Apache-2.0

# Используем официальный образ Python slim для оптимизации размера
FROM python:3.11-slim-bullseye as builder

# Аргументы сборки
ARG WHISPER_MODEL=small
ARG DEBIAN_FRONTEND=noninteractive

# Метаданные образа
LABEL maintainer="akoodoy@capilot.ru" \
      version="1.0.0" \
      description="CallAnnotate - автоматическая аннотация аудиозаписей с предзагруженными Whisper моделями"

# Установка системных зависимостей в один слой
RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    ffmpeg \
    sox \
    curl \
    autoconf \
    libtool \
    build-essential \
    git \
    cargo \
    rustc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Сборка и установка RNNoise из исходников
RUN git clone https://github.com/xiph/rnnoise.git /tmp/rnnoise \
    && cd /tmp/rnnoise \
    && export CFLAGS="-march=native -O3" \
    && ./autogen.sh \
    && ./configure \
    && make \
    && make install \
    && ldconfig \
    && rm -rf /tmp/rnnoise

# Создание пользователя для безопасности
RUN groupadd --gid 1000 callannotate \
    && useradd --uid 1000 --gid callannotate --shell /bin/bash --create-home callannotate

WORKDIR /app

COPY src/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY scripts/preload_whisper.py /app/preload_whisper.py

RUN mkdir -p /home/callannotate/.cache/whisper \
    && chown -R callannotate:callannotate /home/callannotate/.cache \
    && echo "Предзагрузка Whisper модели: $WHISPER_MODEL" \
    && python /app/preload_whisper.py --model $WHISPER_MODEL --cache-dir /home/callannotate/.cache/whisper \
    && ls -la /home/callannotate/.cache/whisper/ \
    && du -sh /home/callannotate/.cache/whisper/

COPY src/ /app/src/
COPY config/ /app/config/

RUN chown -R callannotate:callannotate /app \
    && chmod +x /app/preload_whisper.py

USER callannotate

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=/app/config/default.yaml \
    WHISPER_CACHE_DIR=/home/callannotate/.cache/whisper \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

RUN echo "=== Финальная проверка Whisper кэша ===" \
    && ls -la /home/callannotate/.cache/whisper/ \
    && python -c "import whisper; print('Доступные модели Whisper:', whisper.available_models())" \
    && echo "=== Проверка завершена ==="

CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
