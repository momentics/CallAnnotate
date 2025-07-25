# Используем минимальный образ Alpine Linux
FROM alpine:latest

# Устанавливаем системные зависимости для сборки Python-расширений и работы с аудио
RUN apk update && \
    apk add --no-cache \
      python3 \
      py3-pip \
      python3-dev \
      build-base \
      libsndfile-dev \
      ffmpeg \
      curl \
      git \
    && pip3 install --no-cache-dir --upgrade pip

WORKDIR /app

# Копируем файл зависимостей и устанавливаем PyTorch и остальные библиотеки
# Устанавливаем CPU-версию PyTorch из официального индекса, затем pyannote.audio
COPY src/requirements.txt .
RUN pip3 install --no-cache-dir \
      torch==2.0.1+cpu \
      torchvision==0.15.2+cpu \
      torchaudio==2.0.2+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir -r requirements.txt

# Копируем исходники и конфиг
COPY src/ ./src/
COPY config/default.yaml ./config/default.yaml

# Создаём папку для эмбеддингов
RUN mkdir -p /app/embeddings

EXPOSE 8000

# Команда запуска
CMD ["python3", "src/app.py", "--config", "config/default.yaml"]
