import json
import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_ws(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    (vol/"incoming").mkdir(parents=True)
    f = vol/"incoming"/"d.wav"
    f.write_bytes(b"RIFF")
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    app.state.volume_path = str(vol)

def test_ping_pong_and_error():
    with client.websocket_connect("/ws/client") as ws:
        # valid ping
        ws.send_text(json.dumps({"type":"ping","timestamp":"t"}))
        m1 = json.loads(ws.receive_text())
        assert m1["type"]=="pong"
        # invalid json
        ws.send_text("notjson")
        err = json.loads(ws.receive_text())
        assert err["type"]=="error"

def test_unknown_type():
    with client.websocket_connect("/ws/c") as ws:
        ws.send_text(json.dumps({"type":"foo"}))
        err = json.loads(ws.receive_text())
        assert err["type"]=="error"
