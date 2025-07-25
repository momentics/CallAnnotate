# Используем минимальный образ Alpine Linux для архитектуры amd64
FROM alpine:latest

# Устанавливаем необходимые пакеты: Python 3, pip, ffmpeg иDAV-клиент
RUN apk update && \
    apk add --no-cache \
      python3 \
      py3-pip \
      ffmpeg \
      curl \
      git \
      && \
    # Обновляем pip и устанавливаем Python-зависимости
    pip3 install --no-cache-dir --upgrade pip

WORKDIR /app

# Копируем файл зависимостей и устанавливаем библиотеки
COPY src/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения и конфиг по умолчанию
COPY src/ ./src/
COPY config/default.yaml ./config/default.yaml

# Создаём каталог для хранения эмбеддингов голосов
RUN mkdir -p /app/embeddings

# Открываем порт для HTTP/WebSocket
EXPOSE 8000

# Точка входа
CMD ["python3", "src/app.py", "--config", "config/default.yaml"]
