import os
import io
import pytest
import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path
from app.app import app
import app.api.routers.voices as voices_router

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_embeddings(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    emb = vol / "models" / "embeddings"
    emb.mkdir(parents=True, exist_ok=True)
    vec = emb / "john.vec"
    np.savetxt(str(vec), np.array([1.0,2.0,3.0]))
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    voices_router.EMBEDDINGS_DIR = emb.resolve()
    app.state.volume_path = str(vol)

def test_list_and_get_voices():
    r = client.get("/api/v1/voices/")
    assert r.status_code == 200
    data = r.json()
    assert any(v["name"]=="john" for v in data)

    g = client.get("/api/v1/voices/john")
    assert g.status_code == 200

    ng = client.get("/api/v1/voices/unknown")
    assert ng.status_code == 404

def test_create_voice_conflict_and_invalid():
    # invalid name
    resp = client.post(
        "/api/v1/voices/",
        data={"name":"bad name"},
        files={"embedding_file":("e.vec",io.BytesIO(b""))}
    )
    assert resp.status_code == 400

    # conflict
    resp2 = client.post(
        "/api/v1/voices/",
        data={"name":"john"},
        files={"embedding_file":("john.vec",io.BytesIO(b"1 2 3"))}
    )
    assert resp2.status_code == 409

def test_delete_voice_not_found_and_success():
    dn = client.delete("/api/v1/voices/unknown")
    assert dn.status_code == 404

    # upload new
    resp = client.post(
        "/api/v1/voices/",
        data={"name":"new"},
        files={"embedding_file":("new.vec",io.BytesIO(b"0.1 0.2 0.3"))}
    )
    assert resp.status_code == 201

    dd = client.delete("/api/v1/voices/new")
    assert dd.status_code == 204
