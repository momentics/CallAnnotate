# Используем официальный образ Python slim на базе Debian Bookworm
FROM python:3.13-slim-bookworm

# Метаданные
LABEL maintainer="akoodoy@capilot.ru"
LABEL repository="https://github.com/momentics/CallAnnotate"
LABEL license="Apache-2.0"

# Устанавливаем системные зависимости
RUN apt-get install -y --no-install-recommends \
      ffmpeg \
      curl 

# Создаём рабочую директорию
WORKDIR /app

# Копируем и устанавливаем Python-зависимости
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и конфигурацию
COPY src/ ./src/
COPY config/default.yaml ./config/default.yaml

# Создаём папку для хранения эмбеддингов голосов
RUN mkdir -p /app/embeddings

# Открываем порт для HTTP/WebSocket
EXPOSE 8000

# Точка входа
CMD ["python", "src/app.py", "--config", "config/default.yaml"]
