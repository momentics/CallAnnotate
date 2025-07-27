FROM python:3.11-slim-bullseye

LABEL maintainer="akoodoy@capilot.ru"
LABEL repository="https://github.com/momentics/CallAnnotate"
LABEL license="Apache-2.0"

RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    portaudio19-dev \
    ffmpeg \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/default.yaml ./config/default.yaml

COPY src/ /app/src/
COPY config/ /app/config/

RUN mkdir -p /volume/incoming /volume/processing /volume/completed /volume/logs

ENV PYTHONPATH=/app/src
ENV CONFIG_PATH=/app/config/default.yaml
ENV VOLUME_PATH=/volume

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


