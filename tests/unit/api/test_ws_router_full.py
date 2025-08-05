import os
import json
import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_ws(monkeypatch, tmp_path):
    vol = tmp_path / "volume"
    (vol/"incoming").mkdir(parents=True)
    f = vol/"incoming"/"d.wav"
    f.write_bytes(b"RIFF") 
    monkeypatch.setenv("VOLUME_PATH", str(vol))
    app.state.volume_path = str(vol)

def test_ping_pong_and_job(monkeypatch):
    # patch queue to echo back status updates
    class DummyQ:
        async def start(self): pass
        async def stop(self): pass
        async def subscribe_to_task(self, j,c): pass
        async def add_task(self, j, m):
            # simulate immediate completion send via WS manager
            return True
    monkeypatch.setattr("app.api.routers.ws.get_queue", lambda: DummyQ())
    with client.websocket_connect("/ws/cid") as ws:
        ws.send_text(json.dumps({"type":"ping","timestamp":"t"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"]=="pong"
        # create_job
        ws.send_text(json.dumps({"type":"create_job","filename":"d.wav"}))
        cr = json.loads(ws.receive_text())
        assert cr["type"]=="job_created"
        # subscribe
        ws.send_text(json.dumps({"type":"subscribe_job","job_id":cr["job_id"]}))
        sub = json.loads(ws.receive_text())
        assert sub["type"]=="subscribed"
