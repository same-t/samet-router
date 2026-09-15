"""Samet Router testleri - ag'a cikmaz, requests.post mock'lanir."""
import json
import tempfile
from pathlib import Path

import app as router


def make_client(tmp_path):
    data_file = Path(tempfile.mkdtemp()) / "data.json"
    router.DATA_FILE = data_file
    client_app = router.create_app()
    client_app.config["TESTING"] = True
    c = client_app.test_client()
    # 1 fake key + 1 model ekle
    assert c.post("/admin/api/keys", json={"key": "AIza-FAKE-1", "label": "t1"}).status_code == 201
    return c


def fake_gemini(text="Selam test"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_openai_to_gemini_conversion():
    sys_text, contents = router.openai_messages_to_gemini([
        {"role": "system", "content": "Nazik ol"},
        {"role": "user", "content": "Merhaba"},
        {"role": "assistant", "content": "Selam"},
        {"role": "user", "content": "Naber?"},
    ])
    assert sys_text == "Nazik ol"
    assert contents[0] == {"role": "user", "parts": [{"text": "Merhaba"}]}
    assert contents[1] == {"role": "model", "parts": [{"text": "Selam"}]}


def test_anthropic_to_gemini_conversion():
    sys_text, contents = router.anthropic_messages_to_gemini(
        "Kisa yaz", [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}])
    assert sys_text == "Kisa yaz"
    assert contents[0]["parts"][0]["text"] == "Hi"


def test_gemini_to_openai_shape():
    out = router.gemini_to_openai_response(fake_gemini("abc"), "samet-flash")
    assert out["choices"][0]["message"]["content"] == "abc"
    assert out["model"] == "samet-flash"


def test_gemini_to_anthropic_shape():
    out = router.gemini_to_anthropic_response(fake_gemini("xyz"), "samet-flash")
    assert out["content"][0]["text"] == "xyz"


def test_health_and_models(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/health").get_json()["ok"] is True
    models = c.get("/v1/models").get_json()["data"]
    assert any(m["id"] == "samet-flash" for m in models)


def test_admin_crud(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/admin/api/models",
                   json={"name": "m1", "gemini_model": "gemini-2.0-flash"}).status_code == 201
    state = c.get("/admin/api/state").get_json()
    assert any(m["name"] == "m1" for m in state["models"])
    assert c.delete("/admin/api/models/m1").get_json()["ok"] is True
    # key maskeli donmeli, ham key sizmamali
    assert "AIza-FAKE-1" not in json.dumps(c.get("/admin/api/state").get_json())


def test_chat_completions_mocked(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    class R:
        status_code = 200
        def json(self): return fake_gemini("mock-cevap")
        text = "ok"

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1/chat/completions",
               json={"model": "samet-flash",
                     "messages": [{"role": "user", "content": "Merhaba"}]})
    assert r.status_code == 200, r.text
    assert "mock-cevap" in r.get_json()["choices"][0]["message"]["content"]


def test_anthropic_mocked(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    class R:
        status_code = 200
        def json(self): return fake_gemini("a-cevap")
        text = "ok"

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1/messages",
               json={"model": "samet-flash", "max_tokens": 50,
                     "messages": [{"role": "user", "content": "Hi"}]})
    assert r.status_code == 200, r.text
    assert r.get_json()["content"][0]["text"] == "a-cevap"


def test_gemini_passthrough_mocked(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    class R:
        status_code = 200
        def json(self): return fake_gemini("g-cevap")
        text = "ok"

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1beta/models/samet-flash:generateContent",
               json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert r.status_code == 200
    assert "g-cevap" in json.dumps(r.get_json())


def test_no_key_gives_502(tmp_path):
    data_file = Path(tempfile.mkdtemp()) / "data.json"
    router.DATA_FILE = data_file
    a = router.create_app()
    a.config["TESTING"] = True
    c = a.test_client()
    # henuz key yok ama default modeller var -> chat 502 donmeli (key yok hatasi)
    r = c.post("/v1/chat/completions",
               json={"model": "samet-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert "key" in r.get_json()["error"].lower()
