FROM python:3.11-slim-bullseye

LABEL maintainer="akoodoy@capilot.ru"
LABEL repository="https://github.com/momentics/CallAnnotate"
LABEL license="Apache-2.0"

RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsndfile1-dev \
    portaudio19-dev \
    ffmpeg \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Установка Rust (для deepfilternet)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    rustc --version

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r src/requirements.txt

# Копируем исходники
COPY src/ src/
COPY config/default.yaml config/default.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]



