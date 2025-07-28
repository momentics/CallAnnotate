# tests/test_websocket_integration.py
# Автор: akoodoy@capilot.ru
# Лицензия: Apache-2.0

import json
from fastapi.testclient import TestClient

from app.app import app

client = TestClient(app)

def test_ws_ping():
    with client.websocket_connect("/ws/test_client") as ws:
        ws.send_text(json.dumps({"type":"ping","timestamp":"now"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "pong"

def test_ws_create_and_subscribe(tmp_path, monkeypatch):
    vol = tmp_path / "volume"
    (vol / "incoming").mkdir(parents=True)
    f = vol / "incoming" / "f.wav"
    f.write_bytes(b"data")
    monkeypatch.setenv("VOLUME_PATH", str(vol))

    with client.websocket_connect("/ws/wsid") as ws:
        # create
        ws.send_text(json.dumps({"type":"create_job","filename":"f.wav","priority":3}))
        create = json.loads(ws.receive_text())
        assert create["type"] == "job_created"
        job_id = create["job_id"]

        # subscribe
        ws.send_text(json.dumps({"type":"subscribe_job","job_id":job_id}))
        sub = json.loads(ws.receive_text())
        assert sub["type"] == "subscribed"
        assert sub["job_id"] == job_id

def test_ws_invalid_json():
    with client.websocket_connect("/ws/cid") as ws:
        ws.send_text("notjson")
        err = json.loads(ws.receive_text())
        assert err["code"] == "INVALID_JSON"

def test_ws_unknown_type():
    with client.websocket_connect("/ws/cid2") as ws:
        ws.send_text(json.dumps({"type":"foo"}))
        err = json.loads(ws.receive_text())
        assert err["code"] == "UNKNOWN_TYPE"
