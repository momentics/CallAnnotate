# Dockerfile для CallAnnotate

FROM python:3.11-slim-bullseye

# Устанавливаем необходимые системные пакеты: компиляторы, библиотеки для аудио,
# утилиты для распаковки архивов и сборки RNNoise
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential autoconf libtool git unzip p7zip-full libsndfile1-dev ffmpeg sox curl \
    && \
    # Сборка и установка RNNoise из исходников
    git clone https://github.com/xiph/rnnoise.git /tmp/rnnoise && \
    cd /tmp/rnnoise && \
    export CFLAGS="-march=native -O3" && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    rm -rf /tmp/rnnoise && \
    # Обновляем pip и устанавливаем Python-зависимости
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      fastapi==0.104.1 \
      starlette==0.27.0 \
      uvicorn[standard]==0.24.0 \
      python-multipart==0.0.6 \
      websockets>=11.0.3,<12.0 \
      aiofiles==23.2.1 \
      httpx==0.24.1 \
      anyio==3.7.1 \
      openai-whisper>=20231117 \
      speechbrain>=1.0.0 \
      pyannote.audio>=3.1.1 \
      faiss-cpu>=1.7.4 \
      torch>=2.1.1,<2.8.0 \
      torchaudio>=2.2.0,<2.8.0 \
      librosa==0.10.1 \
      soundfile==0.12.1 \
      numpy>=1.24,<2.0 \
      scipy==1.11.4 \
      deepfilternet>=0.2.0 \
      pydub>=0.25.1 \
      transformers>=4.35.2 \
      pytorch-lightning>=2.5.2 \
      caldav>=1.3.9 \
      vobject>=0.9.6.1 \
      requests>=2.32.4 \
      pyyaml>=6.0.1 \
      python-dotenv>=1.0.0 \
      pydantic>=2.5.0 \
      pydantic-settings>=2.1.0 \
      python-dateutil>=2.8.2 \
      pytz>=2025.2 \
      click>=8.1.7 \
      pytest==7.4.0 \
      pytest-mock>=3.12.0 \
      pytest-aioresponses>=0.3.0 \
      pytest-httpx>=0.18.0 \
      pytest-asyncio==0.21.1 \
    && \
    # Очищаем кеш apt и временные файлы
    #apt-get purge -y --auto-remove build-essential autoconf libtool git unzip p7zip-full && \
    apt-get purge -y --auto-remove build-essential autoconf libtool git && \
    rm -rf /var/lib/apt/lists/*

# Копируем исходный код приложения
COPY src/ /app/src/
COPY config/ /app/config/
COPY docker/preload_models.sh /app/preload_models.sh

# Копируем локальные архивы моделей в контейнер и распаковываем
# Предполагается, что в контексте сборки есть каталог volume/models с zip-архивами
COPY volume/models /volume/models
RUN mkdir -p /volume/models/embeddings /volume/models/pyannote /volume/models/whisper && \
    # Распаковываем все zip-архивы моделей (включая multi-part .zip/.z01/.z02)
    for archive in /volume/models/*.zip; do \
      7z x "$archive" -o/volume/models -y || true; \
    done && \
    # Удаляем архивы после распаковки
    rm -f /volume/models/*.zip /volume/models/*.z0*

# Создаём пользователя и устанавливаем права
RUN groupadd --gid 1000 callannotate && \
    useradd --uid 1000 --gid callannotate --shell /bin/bash --create-home callannotate && \
    chown -R callannotate:callannotate /app /volume && \
    chmod +x /app/preload_models.sh

# Переключаемся на непривилегированного пользователя
USER callannotate

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONFIG_PATH=/app/config/default.yaml \
    VOLUME_PATH=/volume \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/volume/models

WORKDIR /app

# Точка входа запуска сервера
EXPOSE 8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
