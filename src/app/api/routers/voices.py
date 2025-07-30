# src/app/api/routers/voices.py
"""
REST API для управления известными голосами и эмбеддингами в CallAnnotate.

Автор: akoodoy@capilot.ru
Ссылка: https://github.com/momentics/CallAnnotate
Лицензия: Apache-2.0
"""

from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Form,
    status,
    Response,
)

from ...schemas import VoiceInfo
from ...config import load_settings
from ...utils import ensure_directory

router = APIRouter()

CFG = load_settings()
EMBEDDINGS_DIR = Path(
    CFG.recognition.embeddings_path or "./volume/models/embeddings"
).resolve()


def _embedding_file_path(name: str) -> Path:
    return EMBEDDINGS_DIR / f"{name}.vec"


def _voice_exists(name: str) -> bool:
    return _embedding_file_path(name).exists()


@router.get("/", response_model=List[VoiceInfo], tags=["Voices"])
async def list_voices():
    """Получить список известных голосов."""
    ensure_directory(str(EMBEDDINGS_DIR))
    voices: List[VoiceInfo] = [
        VoiceInfo(name=p.stem, embedding=str(p))
        for p in EMBEDDINGS_DIR.glob("*.vec")
    ]
    return voices


@router.post(
    "/", response_model=VoiceInfo, status_code=status.HTTP_201_CREATED, tags=["Voices"]
)
async def create_voice(
    name: str = Form(..., description="Имя голоса (уникальное, без пробелов)"),
    embedding_file: UploadFile = File(..., description="Файл эмбеддинга (.vec)"),
):
    """Добавить новый голос."""
    ensure_directory(str(EMBEDDINGS_DIR))

    if not name.isidentifier():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Имя должно быть валидным идентификатором (без пробелов и спецсимволов)",
        )
    if _voice_exists(name):
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Голос с именем '{name}' уже существует"
        )

    content = await embedding_file.read()
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустой файл эмбеддинга")

    path = _embedding_file_path(name)
    with open(path, "wb") as f:
        f.write(content)

    return VoiceInfo(name=name, embedding=str(path))


@router.get("/{name}", response_model=VoiceInfo, tags=["Voices"])
async def get_voice(name: str):
    """Получить информацию по голосу."""
    path = _embedding_file_path(name)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Голос не найден")
    return VoiceInfo(name=name, embedding=str(path))


@router.put("/{name}", response_model=VoiceInfo, tags=["Voices"])
async def update_voice(
    name: str,
    embedding_file: UploadFile = File(..., description="Новый файл эмбеддинга (.vec)"),
):
    """Обновить эмбеддинг голоса."""
    path = _embedding_file_path(name)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Голос не найден")

    content = await embedding_file.read()
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустой файл эмбеддинга")

    with open(path, "wb") as f:
        f.write(content)

    return VoiceInfo(name=name, embedding=str(path))


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT, tags=["Voices"])
async def delete_voice(name: str):
    """Удалить голос и файл эмбеддинга."""
    path = _embedding_file_path(name)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Голос не найден")

    try:
        path.unlink()
    except Exception as exc:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Ошибка при удалении: {exc}"
        )

    # Возврат 204 No Content без тела
    return Response(status_code=status.HTTP_204_NO_CONTENT)
