import os
import io
import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.app import app
import app.api.routers.voices as vr

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_embeddings(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    emb = vol / "models" / "embeddings"
    emb.mkdir(parents=True, exist_ok=True)
    vec = emb / "john.vec"
    np.savetxt(str(vec), np.array([1,2,3]))
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    vr.EMBEDDINGS_DIR = emb.resolve()
    app.state.volume_path = str(vol)

def test_list_and_get_and_crud_voice():
    res = client.get("/api/v1/voices/")
    assert res.status_code == 200
    assert any(v["name"]=="john" for v in res.json())
    get = client.get("/api/v1/voices/john")
    assert get.status_code==200
    # create
    file = io.BytesIO(b"0.1 0.2")
    cr = client.post("/api/v1/voices/", files={"embedding_file":("a.vec",file,"application/octet-stream")}, data={"name":"new"})
    assert cr.status_code==201
    # delete
    d = client.delete("/api/v1/voices/new")
    assert d.status_code==204
