import time

from integrations.twitch_tts import ChatMessage, Policy, parse_line
from tests.conftest import make_voice_wav


def test_parse_twitch_line():
    line = ("@badge-info=;badges=subscriber/12,premium/1;display-name=Alice;mod=0;subscriber=1 "
            ":alice!alice@alice.tmi.twitch.tv PRIVMSG #chaine :!tts Bonjour le chat")
    m = parse_line(line)
    assert m.user == "Alice" and m.text == "!tts Bonjour le chat" and "subscriber" in m.badges
    assert parse_line(":tmi.twitch.tv 001 justinfan123 :Welcome") is None


def test_policy_filters():
    p = Policy(command="!tts", who="subs", cooldown=10, blocklist=("gros mot",))
    sub = ChatMessage("Alice", "!tts salut https://x.y/z regardez", {"subscriber"})
    assert p.text_for(sub, now=100) == "Alice dit : salut un lien regardez"
    assert p.text_for(ChatMessage("Alice", "!tts encore", {"subscriber"}), now=105) is None  # délai
    assert p.text_for(ChatMessage("Alice", "!tts encore", {"subscriber"}), now=111) == "Alice dit : encore"
    assert p.text_for(ChatMessage("Bob", "!tts salut", set()), now=100) is None  # pas abonné
    assert p.text_for(ChatMessage("Carl", "salut", {"subscriber"}), now=100) is None  # pas de commande
    assert p.text_for(ChatMessage("Dan", "!tts un gros mot", {"subscriber"}), now=100) is None
    assert p.text_for(ChatMessage("Nightbot", "!tts pub", {"moderator"}), now=100) is None
    assert Policy(say_name=False).text_for(ChatMessage("E", "ouiiiiiiii", set())) == "ouiii"


def test_overlay_and_say_route_to_browser_live(client, fake_model):
    assert client.get("/api/overlay").json() == {"active": False, "speaking": False, "lines": []}
    assert client.post("/api/realtime/say", json={"text": "salut"}).status_code == 409
    v = client.post("/api/voices", data={"name": "V", "consent": "true"},
                    files=[("files", ("a.wav", make_voice_wav(4.0), "audio/wav"))]).json()
    with client.websocket_connect("/api/realtime/ws") as ws:
        ws.send_json({"sample_rate": 16000, "mode": "vc", "model_id": "fake", "voice_id": v["id"],
                      "say_model_id": "fake", "warmup": False})
        msg = ws.receive_json()
        while msg["type"] == "loading":
            msg = ws.receive_json()
        assert msg["type"] == "started"
        r = client.post("/api/realtime/say", json={"text": "Bonjour le chat"})
        assert r.status_code == 200, r.text
        deadline = time.time() + 5
        ov = {}
        while time.time() < deadline:
            ov = client.get("/api/overlay").json()
            if ov["lines"]:
                break
            time.sleep(0.05)
        assert ov["active"] and ov["voice"] == "V" and ov["lines"][-1]["text"] == "Bonjour le chat"
        assert ov["lines"][-1]["typed"] is True
        ws.send_text("stop")
    time.sleep(0.3)
    assert client.get("/api/overlay").json()["active"] is False
