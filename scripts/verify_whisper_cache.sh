#!/usr/bin/env bash
# Проверка наличия закэшированных моделей Whisper в CI
set -e
CACHE_DIR="${WHISPER_CACHE_DIR:-$HOME/.cache/whisper}"
for sz in tiny base small medium large; do
  if [ ! -d "$CACHE_DIR/$sz" ]; then
    echo "Whisper model $sz not found in cache"
    exit 1
  fi
done
echo "All Whisper models are cached."
