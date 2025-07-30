# tests/test_api_voices.py

import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.app import app

client = TestClient(app)

@pytest.fixture
def setup_embeddings_dir(tmp_path):
    emb_dir = tmp_path / "models" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    # Создаём фиктивный embedding файл
    (emb_dir / "testuser.vec").write_text("0.1 0.2 0.3\n0.4 0.5 0.6")
    os.environ["VOLUME_PATH"] = str(tmp_path)
    return emb_dir

def test_list_voices_empty(tmp_path):
    resp = client.get("/api/v1/voices/")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

def test_create_and_get_and_delete_voice(tmp_path, monkeypatch):
    emb_dir = tmp_path / "models" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    
    # POST create voice
    data = {
        "name": "testvoice"
    }
    file_content = b"0.1 0.2 0.3\n0.4 0.5 0.6"
    files = {
        "embedding_file": ("embedding.vec", file_content, "application/octet-stream")
    }
    response = client.post(
        "/api/v1/voices/",
        data={"name": data["name"]},
        files=files,
    )
    assert response.status_code == 201
    created = response.json()
    assert created["name"] == "testvoice"
    path = created["embedding"]
    assert Path(path).exists()

    # GET voice info
    resp_get = client.get(f"/api/v1/voices/{data['name']}")
    assert resp_get.status_code == 200
    assert resp_get.json()["name"] == data["name"]

    # PUT update voice
    new_file_content = b"0.7 0.8 0.9"
    files = {
        "embedding_file": ("embedding_updated.vec", new_file_content, "application/octet-stream")
    }
    resp_put = client.put(f"/api/v1/voices/{data['name']}", files=files)
    assert resp_put.status_code == 200
    assert resp_put.json()["name"] == data["name"]

    # DELETE voice
    resp_del = client.delete(f"/api/v1/voices/{data['name']}")
    assert resp_del.status_code == 204
    # Проверяем что файл удалён
    assert not Path(path).exists()

def test_conflict_on_create_existing_voice(tmp_path):
    emb_dir = tmp_path / "models" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    file_path = emb_dir / "conflict.vec"
    file_path.write_text("0 0 0")

    files = {
        "embedding_file": ("embedding.vec", b"0 0 0", "application/octet-stream")
    }

    # Первый раз создаём
    data = {"name": "conflict"}
    response1 = client.post("/api/v1/voices/", data=data, files=files)
    assert response1.status_code == 409
