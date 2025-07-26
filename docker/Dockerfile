# Используем официальный образ Python slim на базе Debian Bookworm
FROM python:3.11-slim-bullseye

# Метаданные
LABEL maintainer="akoodoy@capilot.ru"
LABEL repository="https://github.com/momentics/CallAnnotate"
LABEL license="Apache-2.0"

# Устанавливаем системные зависимости с обновлением списков пакетов
RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    portaudio19-dev \
    ffmpeg \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Создаём рабочую директорию
WORKDIR /app

# Копируем и устанавливаем Python-зависимости
COPY src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и конфигурацию
COPY src/ ./src/
COPY config/default.yaml ./config/default.yaml

# Копирование исходного кода
COPY src/ /app/src/
COPY config/ /app/config/

# Создание volume директорий
RUN mkdir -p /app/volume/incoming /app/volume/processing /app/volume/completed /app/volume/logs

# Установка переменных окружения
ENV PYTHONPATH=/app/src
ENV CONFIG_PATH=/app/config/default.yaml
ENV VOLUME_PATH=/app/volume

# Открываем порт для HTTP/WebSocket
EXPOSE 8000

# Health check для Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск приложения
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]