import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def locked(tmp_data, monkeypatch):
    monkeypatch.setenv("VOICECLONE_PASSWORD", "s3cret")
    from voiceclone.server import create_app

    with TestClient(create_app()) as c:
        yield c


def test_open_without_password(client):
    assert client.get("/api/system").status_code == 200
    assert client.get("/api/auth").json() == {"protected": False}


def test_password_protects_api_ui_and_ws(locked):
    assert locked.get("/api/system").status_code == 401
    r = locked.get("/", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("/login.html")
    assert locked.get("/login.html").status_code == 200
    with pytest.raises(WebSocketDisconnect) as e:
        with locked.websocket_connect("/api/realtime/ws") as ws:
            ws.receive_json()
    assert e.value.code == 4401
    assert locked.post("/api/login", json={"password": "nope"}).status_code == 401
    assert locked.post("/api/login", json={"password": "s3cret"}).status_code == 200
    assert locked.get("/api/system").status_code == 200  # cookie posé
    assert locked.get("/").status_code == 200
    locked.post("/api/logout")
    locked.cookies.clear()
    assert locked.get("/api/system").status_code == 401


def test_bearer_basic_and_key(locked):
    import base64

    assert locked.get("/api/system", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    basic = base64.b64encode(b"x:s3cret").decode()
    assert locked.get("/api/system", headers={"Authorization": f"Basic {basic}"}).status_code == 200
    locked.cookies.clear()
    r = locked.get("/api/system?key=s3cret")
    assert r.status_code == 200 and "voiceclone_auth" in r.headers.get("set-cookie", "")
    assert locked.get("/api/system?key=bad").status_code == 200  # le cookie suffit désormais
    locked.cookies.clear()
    assert locked.get("/api/system?key=bad").status_code == 401
