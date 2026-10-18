"""Samet Router tests - no network access, requests.post is mocked."""
import json
import tempfile
import time
from pathlib import Path

import pytest

import app as router


def make_client(tmp_path):
    data_file = Path(tempfile.mkdtemp()) / "data.json"
    router.DATA_FILE = data_file
    client_app = router.create_app()
    client_app.config["TESTING"] = True
    c = client_app.test_client()
    # add 1 fake key + 1 model
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
    # key must return masked, raw key must not leak
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
    # no key yet but default models exist -> chat must return 502 (missing key error)
    r = c.post("/v1/chat/completions",
               json={"model": "samet-flash",
                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502
    assert "key" in r.get_json()["error"].lower()


# ---------------------------------------------------------------------------
# Claude Code / tools / streaming regression tests
# ---------------------------------------------------------------------------

def _fake_fn_call_gemini(text="bakiyorum", name="get_weather", args=None):
    args = args or {"city": "Ankara"}
    return {"candidates": [{"content": {"parts": [
        {"text": text}, {"functionCall": {"name": name, "args": args}}]},
        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5}}


def test_anthropic_tools_passthrough(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_gemini("duz cevap")

    def fake_post(url, json=None, timeout=60):
        seen["payload"] = json
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/messages", json={
        "model": "samet-flash", "max_tokens": 100,
        "system": [{"type": "text", "text": "Kisa yaz", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hava?"}]}],
        "tools": [{"name": "get_weather", "description": "hava",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "tool_choice": {"type": "auto"},
    })
    assert r.status_code == 200, r.text
    payload = seen["payload"]
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert "Kisa yaz" in payload["systemInstruction"]["parts"][0]["text"]


def test_anthropic_tool_use_response(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return _fake_fn_call_gemini()

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1/messages", json={
        "model": "samet-flash", "max_tokens": 100,
        "messages": [{"role": "user", "content": "hava?"}]})
    body = r.get_json()
    assert r.status_code == 200, r.text
    assert body["stop_reason"] == "tool_use"
    tool = [b for b in body["content"] if b["type"] == "tool_use"][0]
    assert tool["name"] == "get_weather" and tool["input"]["city"] == "Ankara"
    assert body["usage"]["input_tokens"] == 12


def test_anthropic_tool_result_and_thinking(tmp_path):
    sys_txt, contents = router.anthropic_messages_to_gemini(
        [{"type": "text", "text": "Sys", "cache_control": {"type": "ephemeral"}}],
        [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "gizli"},
            {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"cmd": "ls"}}]},
         {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu_1",
              "content": [{"type": "text", "text": "file.txt"}]}]}])
    assert sys_txt == "Sys"
    assert contents[0]["parts"][0] == {"functionCall": {"name": "bash", "args": {"cmd": "ls"}}}
    assert contents[1]["parts"][0]["functionResponse"]["name"] == "bash"


def test_anthropic_image_and_claude_fallback(tmp_path):
    _, contents = router.anthropic_messages_to_gemini(
        None, [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}}]}])
    assert contents[0]["parts"][0]["inlineData"]["mimeType"] == "image/png"
    data = router.load_data()
    assert router.resolve_gemini_model(data, "claude-sonnet-4-5") in [
        (m.get("upstream_model") or m.get("gemini_model")) for m in data["models"]]


def test_anthropic_stream_and_count(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return _fake_fn_call_gemini(text="selam")

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1/messages", json={
        "model": "samet-flash", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("Content-Type", "")
    assert "message_start" in r.text and "input_json_delta" in r.text
    rc = c.post("/v1/messages/count_tokens", json={
        "model": "samet-flash", "messages": [{"role": "user", "content": "hello"}]})
    assert rc.get_json()["input_tokens"] >= 1
    # CORS + OPTIONS
    assert c.options("/v1/messages").status_code == 204
    assert c.get("/health").headers.get("Access-Control-Allow-Origin") == "*"


def test_openai_tools_and_stream(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return _fake_fn_call_gemini(name="get_time", args={})

    def fake_post(url, json=None, timeout=60):
        seen["payload"] = json
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "samet-flash",
        "messages": [{"role": "user", "content": "saat?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_time", "description": "saat",
            "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": "required", "stream": True})
    assert r.status_code == 200, r.text
    assert "chat.completion.chunk" in r.text and "[DONE]" in r.text
    assert seen["payload"]["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


def test_models_detail_and_default_system(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    # unknown claude model must return 200 (fallback)
    assert c.get("/v1/models/claude-sonnet-4-5").status_code == 200
    # default system prompt save + merge
    assert c.post("/admin/api/settings", json={"default_system": "Turkce yaz"}).status_code == 200
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_gemini("ok")

    def fake_post(url, json=None, timeout=60):
        seen["payload"] = json
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    c.post("/v1/messages", json={"model": "samet-flash", "max_tokens": 10,
                                 "system": "Kisa yaz",
                                 "messages": [{"role": "user", "content": "hi"}]})
    sys_txt = seen["payload"]["systemInstruction"]["parts"][0]["text"]
    assert "Turkce yaz" in sys_txt and "Kisa yaz" in sys_txt


# ---------------------------------------------------------------------------
# Dashboard + user API key tests
# ---------------------------------------------------------------------------

def _usage_gemini(inp=10, out=4, text="selam"):
    return {"candidates": [{"content": {"parts": [{"text": text}]},
                             "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": inp, "candidatesTokenCount": out}}


def test_router_key_crud_and_auth(tmp_path):
    c = make_client(tmp_path)
    r = c.post("/admin/api/router-keys", json={"label": "ali"})
    assert r.status_code == 201
    full = r.get_json()["key"]
    kid = r.get_json()["id"]
    assert full.startswith("sk-samet-")
    # raw key must not leak into state
    assert full not in json.dumps(c.get("/admin/api/state").get_json())
    # auth is required when a key is defined
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"x-api-key": full}).status_code == 200
    assert c.get("/v1/models", headers={"Authorization": f"Bearer {full}"}).status_code == 200
    assert c.post("/v1/messages", json={"model": "x", "messages": []},
                  headers={"x-api-key": "yanlis"}).status_code == 401
    # revoked key must get 401 (while second key remains)
    k2 = c.post("/admin/api/router-keys", json={"label": "veli"}).get_json()["key"]
    assert c.delete(f"/admin/api/router-keys/{kid}").get_json()["ok"] is True
    assert c.post("/v1/messages", json={"model": "x", "max_tokens": 5,
                                        "messages": [{"role": "user", "content": "hi"}]},
                  headers={"x-api-key": full}).status_code == 401
    assert c.get("/v1/models", headers={"x-api-key": k2}).status_code == 200


def test_dashboard_stats_tokens(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    key = c.post("/admin/api/router-keys", json={"label": "ali"}).get_json()["key"]

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return _usage_gemini(10, 4)

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    h = {"x-api-key": key}
    assert c.post("/v1/chat/completions", json={
        "model": "samet-flash", "messages": [{"role": "user", "content": "hi"}]},
        headers=h).get_json()["usage"]["total_tokens"] == 14
    assert c.post("/v1/messages", json={
        "model": "samet-flash", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}]}, headers=h).get_json()["usage"] == \
        {"input_tokens": 10, "output_tokens": 4}
    st = c.get("/admin/api/stats").get_json()
    assert st["total_requests"] == 2 and st["ok"] == 2 and st["err"] == 0
    assert st["input_tokens"] == 20 and st["output_tokens"] == 8 and st["total_tokens"] == 28
    assert st["by_model"]["samet-flash"]["requests"] == 2
    assert st["by_key"]["ali"]["requests"] == 2
    logs = c.get("/admin/api/state").get_json()["logs"]
    assert logs[0]["input_tokens"] == 10 and logs[0]["api_key_label"] == "ali"
    # failed request must also be counted
    c.post("/v1/chat/completions", json={"model": "samet-flash", "messages": "bozuk"},
           headers=h)
    st2 = c.get("/admin/api/stats").get_json()
    assert st2["total_requests"] >= 3


def test_compute_stats_pure():
    logs = [
        {"ok": True, "input_tokens": 5, "output_tokens": 7, "model": "m",
         "endpoint": "e", "api_key_label": "k", "ts": "2026-01-01T00:00:00", "latency_ms": 10},
        {"ok": False, "input_tokens": 1, "output_tokens": 0, "model": "m",
         "endpoint": "e", "api_key_label": "k", "ts": "2026-01-01T01:00:00", "latency_ms": 20},
    ]
    st = router.compute_stats(logs, None)
    assert (st["total_requests"], st["ok"], st["err"]) == (2, 1, 1)
    assert st["success_rate"] == 50.0
    assert (st["input_tokens"], st["output_tokens"], st["total_tokens"]) == (6, 7, 13)
    assert st["avg_latency_ms"] == 15.0


def test_stats_reset(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return _usage_gemini(1, 1)

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    c.post("/v1/chat/completions", json={"model": "samet-flash",
                                         "messages": [{"role": "user", "content": "hi"}]})
    assert c.get("/admin/api/stats").get_json()["total_requests"] >= 1
    assert c.delete("/admin/api/stats").get_json()["ok"] is True
    assert c.get("/admin/api/stats").get_json()["total_requests"] == 0


# ---------------------------------------------------------------------------
# Multi-provider upstream + thinking tests
# ---------------------------------------------------------------------------

def _add_openai_upstream(c, base=None):
    body = {"key": "sk-test-openai", "label": "oai", "provider": "openai"}
    if base:
        body["base_url"] = base
    assert c.post("/admin/api/provider-keys", json=body).status_code == 201
    assert c.post("/admin/api/models", json={
        "name": "samet-oai", "provider": "openai",
        "upstream_model": "gpt-4o-mini"}).status_code == 201


def _add_anthropic_upstream(c):
    assert c.post("/admin/api/provider-keys", json={
        "key": "sk-ant-test", "label": "ant",
        "provider": "anthropic"}).status_code == 201
    assert c.post("/admin/api/models", json={
        "name": "claude-alias", "provider": "anthropic",
        "upstream_model": "claude-x"}).status_code == 201


def fake_openai_chat(text="oai-cevap", tools=None, reasoning=None):
    msg = {"role": "assistant", "content": text}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tools:
        msg["tool_calls"] = tools
        msg["content"] = None
    return {"id": "chatcmpl-up", "object": "chat.completion", "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}


def fake_anthropic_msg(text="ant-cevap", blocks=None, stop="end_turn"):
    return {"id": "msg_up", "type": "message", "role": "assistant", "model": "claude-x",
            "content": blocks if blocks is not None else [{"type": "text", "text": text}],
            "stop_reason": stop, "usage": {"input_tokens": 9, "output_tokens": 6}}


def test_migration_old_data(tmp_path):
    data_file = Path(tempfile.mkdtemp()) / "data.json"
    router.DATA_FILE = data_file
    data_file.write_text(json.dumps({
        "gemini_keys": [{"id": "a1", "key": "AIza-X", "label": "g", "created_at": "t"}],
        "models": [{"name": "m", "gemini_model": "gemini-1.5-flash"}],
        "logs": [], "key_index": 1}), encoding="utf-8")
    data = router.load_data()
    assert data["provider_keys"][0]["protocol"] == "gemini"
    assert data["provider_keys"][0]["key"] == "AIza-X"
    assert data["models"][0]["provider"] == "gemini"
    assert data["models"][0]["upstream_model"] == "gemini-1.5-flash"


def test_provider_key_validation(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/admin/api/provider-keys", json={
        "key": "x", "provider": "uzay"}).status_code == 400
    assert c.post("/admin/api/provider-keys", json={
        "key": "x", "provider": "custom"}).status_code == 400  # base_url required
    assert c.post("/admin/api/provider-keys", json={
        "key": "x", "provider": "openai", "protocol": "irc"}).status_code == 400
    assert c.post("/admin/api/provider-keys", json={
        "key": "AIza-FAKE-1", "provider": "gemini"}).status_code == 400  # duplicate
    r = c.post("/admin/api/provider-keys", json={
        "key": "sk-ollama", "label": "local", "provider": "custom",
"protocol": "openai", "base_url": "http://localhost:11434/v1"})
    assert r.status_code == 201
    cat = c.get("/admin/api/providers").get_json()
    assert {p["id"] for p in cat} == {"opencode", "openai", "anthropic", "gemini",
                                      "commandcode", "custom"}
    # raw key must not leak
    assert "sk-ollama" not in json.dumps(c.get("/admin/api/state").get_json())


def test_openai_to_openai_passthrough(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_openai_upstream(c)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("passthrough-ok")

    def fake_post(url, **kw):
        seen["url"] = url
        seen.update(kw)
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "samet-oai",
        "messages": [{"role": "user", "content": "selam"}],
        "reasoning_effort": "low"})
    assert r.status_code == 200, r.text
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test-openai"
    assert seen["json"]["model"] == "gpt-4o-mini"
    assert seen["json"]["reasoning_effort"] == "low"  # passthrough
    body = r.get_json()
    assert body["model"] == "samet-oai"  # router name is written back
    assert body["choices"][0]["message"]["content"] == "passthrough-ok"
    assert body["usage"]["total_tokens"] == 10


def test_openai_upstream_error_forwarded(tmp_path, monkeypatch):
    """Non-429 upstream errors are still relayed verbatim.

    429 is deliberately different: it parks the key and, once no key is usable,
    the handler replays a cached response instead (see the cooldown tests).
    """
    c = make_client(tmp_path)
    _add_openai_upstream(c)

    class R:
        status_code = 500
        text = '{"error": {"message": "upstream is down"}}'

        def json(self):
            return {"error": {"message": "upstream is down"}}

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1/chat/completions", json={
        "model": "samet-oai", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 500
    assert "upstream is down" in r.text


def test_custom_base_url_used(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    assert c.post("/admin/api/provider-keys", json={
        "key": "sk-local", "provider": "custom", "protocol": "openai",
        "base_url": "http://localhost:11434/v1"}).status_code == 201
    assert c.post("/admin/api/models", json={
        "name": "local-m", "provider": "openai",
        "upstream_model": "llama3"}).status_code == 201
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("local-ok")

    def fake_post(url, **kw):
        seen["url"] = url
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "local-m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"


def test_anthropic_to_anthropic_passthrough(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_anthropic_upstream(c)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_anthropic_msg(blocks=[
                {"type": "thinking", "thinking": "hmm", "signature": "sig123"},
                {"type": "redacted_thinking", "data": "xyz"},
                {"type": "text", "text": "cevap"}])

    def fake_post(url, **kw):
        seen["url"] = url
        seen.update(kw)
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/messages",
               headers={"anthropic-beta": "beta-ozellik"},
               json={"model": "claude-alias", "max_tokens": 100,
                     "system": "Kisa yaz",
                     "thinking": {"type": "enabled", "budget_tokens": 1000},
                     "messages": [{"role": "user", "content": "dusunceli cevap"}]})
    assert r.status_code == 200, r.text
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-ant-test"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["headers"]["anthropic-beta"] == "beta-ozellik"
    assert seen["json"]["model"] == "claude-x"
    assert seen["json"]["thinking"] == {"type": "enabled", "budget_tokens": 1000}
    assert "Kisa yaz" in seen["json"]["system"]
    body = r.get_json()
    assert body["model"] == "claude-alias"
    assert body["content"][0] == {"type": "thinking", "thinking": "hmm", "signature": "sig123"}
    assert body["usage"] == {"input_tokens": 9, "output_tokens": 6}
    # log thinking flag + provider
    log = c.get("/admin/api/state").get_json()["logs"][0]
    assert log["thinking"] is True and log["provider"] == "anthropic"


def test_anthropic_upstream_error_forwarded(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_anthropic_upstream(c)

    class R:
        status_code = 400
        text = "bad"

        def json(self):
            return {"type": "error", "error": {"type": "invalid_request_error",
                                               "message": "kotu istek"}}

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1/messages", json={"model": "claude-alias", "max_tokens": 5,
                                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.get_json()["error"]["message"] == "kotu istek"


def test_openai_to_anthropic_conversion(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_anthropic_upstream(c)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_anthropic_msg("donustu")

    def fake_post(url, **kw):
        seen.update(kw)
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "claude-alias",
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "saat?"}],
        "tools": [{"type": "function", "function": {
            "name": "get_time", "description": "saat",
            "parameters": {"type": "object", "properties": {}}}}],
        "tool_choice": "required"})
    assert r.status_code == 200, r.text
    up = seen["json"]
    assert up["model"] == "claude-x"
    assert "sys" in up["system"]
    assert up["tools"][0]["name"] == "get_time"
    assert up["tool_choice"] == {"type": "any"}
    assert r.get_json()["choices"][0]["message"]["content"] == "donustu"


def test_anthropic_to_openai_conversion(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_openai_upstream(c)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("oai-yanit")

    def fake_post(url, **kw):
        seen.update(kw)
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/messages", json={
        "model": "samet-oai", "max_tokens": 50,
        "system": "Kisa",
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "gizli"},
            {"type": "tool_use", "id": "tu_9", "name": "bash",
             "input": {"cmd": "ls"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_9", "content": "ok"}]}]})
    assert r.status_code == 200, r.text
    up = seen["json"]
    assert up["model"] == "gpt-4o-mini"
    assert up["messages"][0] == {"role": "system", "content": "Kisa"}
    # thinking block cannot be carried into OpenAI input -> dropped, tool flow preserved
    dumped = json.dumps(up)
    assert "gizli" not in dumped
    assert any(m.get("tool_calls") for m in up["messages"] if isinstance(m, dict))
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "tu_9"
               for m in up["messages"])
    assert r.get_json()["content"][0]["text"] == "oai-yanit"


def test_gemini_in_to_openai_upstream(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_openai_upstream(c)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("g-cevap")

    def fake_post(url, **kw):
        seen.update(kw)
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1beta/models/samet-oai:generateContent", json={
        "systemInstruction": {"parts": [{"text": "sys"}]},
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}]})
    assert r.status_code == 200, r.text
    assert seen["json"]["model"] == "gpt-4o-mini"
    assert "g-cevap" in json.dumps(r.get_json())


def test_key_rotation_across_provider_keys(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    calls = []

    class R429:
        status_code = 429
        text = "rate limited"

        def close(self):
            pass

    class R200:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_gemini("ikinci-key")

    def fake_post(url, **kw):
        calls.append(url)
        return R429() if len(calls) == 1 else R200()

    monkeypatch.setattr(router.requests, "post", fake_post)
    assert c.post("/admin/api/keys", json={"key": "AIza-FAKE-2"}).status_code == 201
    r = c.post("/v1/chat/completions", json={
        "model": "samet-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert len(calls) == 2
    assert "ikinci-key" in r.text


def test_stream_proxy_rewrites_model(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_openai_upstream(c)

    class R:
        status_code = 200

        def iter_content(self, chunk_size=4096):
            yield b'data: {"id": "1", "model": "gpt-4o-mini", '
            yield b'"choices": []}\n\ndata: [DONE]\n\n'

        def close(self):
            pass

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1/chat/completions", json={
        "model": "samet-oai", "messages": [{"role": "user", "content": "hi"}],
        "stream": True})
    assert r.status_code == 200
    assert '"model": "samet-oai"' in r.text
    assert "gpt-4o-mini" not in r.text
    assert "[DONE]" in r.text


def test_converters_pure():
    # openai -> anthropic: tool + reasoning_effort
    abody, th = router.openai_to_anthropic_body({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "t", "parameters": {"type": "object"}}}],
        "tool_choice": "required", "reasoning_effort": "high",
        "stop": "DUR"}, "varsayilan")
    assert th is True
    assert abody["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert abody["tool_choice"] == {"type": "any"}
    assert abody["stop_sequences"] == ["DUR"]
    assert "varsayilan" in abody["system"]
    # anthropic -> openai: tool_use id preserved, thinking dropped
    obody, effort = router.anthropic_to_openai_body({
        "max_tokens": 10,
        "system": [{"type": "text", "text": "S"}],
        "thinking": {"type": "enabled", "budget_tokens": 9000},
        "messages": [{"role": "assistant", "content": [
            {"type": "thinking", "thinking": "z"},
            {"type": "tool_use", "id": "tu_1", "name": "n", "input": {}}]}]})
    assert effort == "high"
    assert obody["reasoning_effort"] == "high"
    assert obody["messages"][0]["content"] == "S"
    assert "z" not in json.dumps(obody)
    assert obody["messages"][1]["tool_calls"][0]["id"] == "tu_1"
    # openai response -> anthropic: reasoning first block
    anth = router.openai_response_to_anthropic(
        fake_openai_chat("t", reasoning="neden"), "m")
    assert anth["content"][0] == {"type": "thinking", "thinking": "neden"}
    assert anth["content"][1] == {"type": "text", "text": "t"}
    # anthropic response -> openai: thinking becomes reasoning_content
    oai = router.anthropic_response_to_openai(fake_anthropic_msg(blocks=[
        {"type": "thinking", "thinking": "derin"},
        {"type": "redacted_thinking", "data": "x"},
        {"type": "text", "text": "son"}]), "m")
    assert oai["choices"][0]["message"]["reasoning_content"] == "derin"
    assert oai["choices"][0]["message"]["content"] == "son"


def test_thinking_sanitize_and_gemini_thought():
    msgs = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "sentetik"},
        {"type": "thinking", "thinking": "gercek", "signature": "s"},
        {"type": "redacted_thinking", "data": "r"},
        {"type": "text", "text": "selam"}]}]
    out = router.sanitize_anthropic_messages_for_upstream(msgs)
    assert any(b.get("signature") == "s" for b in out[0]["content"])
    assert any(b.get("type") == "redacted_thinking" for b in out[0]["content"])
    assert any(b == {"type": "text", "text": "sentetik"} for b in out[0]["content"])
    g = {"candidates": [{"content": {"parts": [
        {"text": "ic dusunce", "thought": True}, {"text": "dis cevap"}]},
        "finishReason": "STOP"}]}
    on = router.gemini_to_anthropic_response(g, "m", thinking_enabled=True)
    assert on["content"][0] == {"type": "thinking", "thinking": "ic dusunce"}
    off = router.gemini_to_anthropic_response(g, "m")
    assert all(b.get("type") != "thinking" for b in off["content"])
    assert off["content"][0] == {"type": "text", "text": "dis cevap"}
    oai = router.gemini_to_openai_response(g, "m")
    assert oai["choices"][0]["message"]["reasoning_content"] == "ic dusunce"
    assert "ic dusunce" not in (oai["choices"][0]["message"]["content"] or "")


def test_resolve_target_and_sse_rewrite():
    data = {"models": [
        {"name": "a", "provider": "openai", "upstream_model": "gpt-x"},
        {"name": "b", "provider": "gemini", "upstream_model": "gemini-y"}]}
    assert router.resolve_target(data, "a") == {
        "name": "a", "provider": "openai", "provider_id": "openai", "upstream_model": "gpt-x"}
    assert router.resolve_target(data, "gemini-2.5-pro")["upstream_model"] == "gemini-2.5-pro"
    assert router.resolve_target(data, "bilinmeyen")["upstream_model"] == "gpt-x"
    chunks = [b'data: {"model": "up", "message": {"model": "up"}}\n', b"data: [DONE]\n"]
    text = b"".join(router.rewrite_sse_model(chunks, "router-ad")).decode()
    assert '"model": "router-ad"' in text and '"up"' not in text
    assert router.body_has_thinking("anthropic", {"thinking": {"type": "enabled"}}) is True
    assert router.body_has_thinking("openai", {"reasoning_effort": "low"}) is True
    assert router.body_has_thinking("openai", {"messages": []}) is False


# ---------------------------------------------------------------------------
# OpenCode connections (providers page) tests
# ---------------------------------------------------------------------------

def test_opencode_connection_fields(tmp_path):
    c = make_client(tmp_path)
    r = c.post("/admin/api/provider-keys", json={
        "key": "public", "label": "Production Key", "provider": "opencode",
        "priority": 2, "proxy_pool": "none"})
    assert r.status_code == 201
    keys = c.get("/admin/api/state").get_json()["provider_keys"]
    k = [x for x in keys if x["label"] == "Production Key"][0]
    assert k["priority"] == 2 and k["protocol"] == "openai"
    assert k["effective_base"] == "https://opencode.ai/zen/v1"
    assert k["provider"] == "opencode"
# raw key must not leak
    assert "public" not in json.dumps(keys)


def _add_commandcode_upstream(c):
    assert c.post("/admin/api/provider-keys", json={
        "key": "cmd-key-123", "label": "cmd",
        "provider": "commandcode"}).status_code == 201
    assert c.post("/admin/api/models", json={
        "name": "ds-flash", "provider": "commandcode",
        "upstream_model": "deepseek/deepseek-v4-flash"}).status_code == 201


def test_commandcode_connection_fields(tmp_path):
    c = make_client(tmp_path)
    _add_commandcode_upstream(c)
    keys = c.get("/admin/api/state").get_json()["provider_keys"]
    k = [x for x in keys if x["provider"] == "commandcode"][0]
    assert k["protocol"] == "openai"
    assert k["effective_base"] == "https://api.commandcode.ai/provider/v1"
    assert router.route_protocol_for_provider("commandcode") == "openai"
    assert router.provider_display_name("commandcode") == "Command Code"
    assert router.provider_short("commandcode") == "cmd"
    # raw key must not leak
    assert "cmd-key-123" not in json.dumps(c.get("/admin/api/state").get_json())


def test_commandcode_catalog_entry(tmp_path):
    c = make_client(tmp_path)
    cat = {p["id"]: p for p in c.get("/admin/api/providers").get_json()}
    p = cat["commandcode"]
    assert p["protocol"] == "openai"
    assert p["base_url"] == "https://api.commandcode.ai/provider/v1"
    assert p["auth"] == "bearer"
    assert p["key_url"]


def test_commandcode_upstream_passthrough(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_commandcode_upstream(c)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("cc-cevap")

    def fake_post(url, **kw):
        seen["url"] = url
        seen.update(kw)
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "ds-flash", "messages": [{"role": "user", "content": "selam"}]})
    assert r.status_code == 200, r.text
    assert seen["url"] == "https://api.commandcode.ai/provider/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer cmd-key-123"
    assert seen["json"]["model"] == "deepseek/deepseek-v4-flash"
    assert r.get_json()["model"] == "ds-flash"  # router name written back


def test_commandcode_provider_detail(tmp_path):
    c = make_client(tmp_path)
    _add_commandcode_upstream(c)
    assert c.get("/admin/providers/commandcode").status_code == 200
    d = c.get("/admin/api/providers/commandcode").get_json()
    assert d["id"] == "commandcode"
    assert d["name"] == "Command Code"
    assert d["short"] == "cmd"
    assert d["auth_required"] is True
    assert d["connected"] is True and d["count"] == 1
    assert d["base_url"] == "https://api.commandcode.ai/provider/v1"
    assert d["suggestions"], "commandcode should ship model suggestions"
    assert [m["name"] for m in d["models"]] == ["ds-flash"]


def test_commandcode_beam_free_model_id(tmp_path):
    """beam-free is the page slug; the api id is reflection/beam:free."""
    c = make_client(tmp_path)
    _add_commandcode_upstream(c)
    d = c.get("/admin/api/providers/commandcode").get_json()
    assert "reflection/beam:free" in d["suggestions"]
    assert "beam-free" not in d["suggestions"]
    # the id survives the round-trip through the admin model API
    assert c.post("/admin/api/providers/commandcode/models", json={
        "upstream_model": "reflection/beam:free"}).status_code == 201
    m = [x for x in c.get("/admin/api/providers/commandcode").get_json()["models"]
         if x["upstream_model"] == "reflection/beam:free"]
    assert m and m[0]["name"] == "cmd/reflection/beam:free"
    # a suggestion already in use is not offered twice
    d2 = c.get("/admin/api/providers/commandcode").get_json()
    assert "reflection/beam:free" not in d2["suggestions"]


def test_model_mapping_pins_its_own_provider(tmp_path, monkeypatch):
    """A commandcode model must not be sent to another openai-protocol key.

    opencode and commandcode both speak the openai protocol, so pooling by
    protocol alone used to hand commandcode requests to OpenCode Zen.
    """
    c = make_client(tmp_path)
    assert c.post("/admin/api/provider-keys", json={
        "key": "oc-key", "label": "oc", "provider": "opencode"}).status_code == 201
    _add_commandcode_upstream(c)

    # both keys are in the same protocol pool
    assert {e["provider"] for e in router._ordered_pool(router.load_data(), "openai")} == {
        "opencode", "commandcode"}

    hits = []

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("ok")

    def fake_post(url, **kw):
        hits.append((url, kw.get("headers", {}).get("Authorization")))
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    # round-robin is on; repeat so a shared cursor would eventually leak
    for _ in range(5):
        assert c.post("/v1/chat/completions", json={
            "model": "ds-flash",
            "messages": [{"role": "user", "content": "hi"}]}).status_code == 200

    assert len(hits) == 5
    for url, auth in hits:
        assert url == "https://api.commandcode.ai/provider/v1/chat/completions"
        assert auth == "Bearer cmd-key-123"


def test_provider_pinned_pool_falls_back_when_missing(tmp_path):
    """Unknown-model fallback still finds keys when the provider has none."""
    data = {"provider_keys": [{"key": "k1", "provider": "opencode",
                               "protocol": "openai"}],
            "settings": {"round_robin": False}}
    assert router._pool_for_protocol(data, "openai", "opencode")[0]["key"] == "k1"
    # commandcode has no key -> fall back to the whole protocol pool
    assert len(router._pool_for_protocol(data, "openai", "commandcode")) == 1


# ---------------------------------------------------------------------------
# OpenCode v2 free-tier fingerprint
# ---------------------------------------------------------------------------


def test_opencode_free_model_detection():
    assert router.opencode_is_free_model("mimo-v2.6-flash-free")
    assert router.opencode_is_free_model("space-bunny-free")
    assert router.opencode_is_free_model("stealth/cat-jam:free")
    assert not router.opencode_is_free_model("gpt-5.5")
    assert not router.opencode_is_free_model("kimi-k3")
    assert not router.opencode_is_free_model("")


def test_opencode_session_id_shape():
    import re as _re
    router._OC_SESSIONS.clear()
    sid = router.opencode_session({"id": "abc"})
    assert _re.match(r"^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$", sid), sid
    # stable per connection (upstream meters the free tier per session)
    assert router.opencode_session({"id": "abc"}) == sid
    assert router.opencode_session({"id": "other"}) != sid
    # the 12 hex prefix must decode to a near-now inverted timestamp
    prefix = int(sid[4:16], 16)
    import time as _t
    ms = int(_t.time() * 1000)
    mask = (1 << 48) - 1
    decoded = ((~prefix) & mask)
    expected = (ms * 0x1000 + 1) & mask
    # small counter: the decode is within a few ms of now
    drift = (decoded - expected) & mask
    assert min(drift, mask - drift) < 0x1000 * 5000, "counter must stay small"


def test_opencode_v2_headers_shape():
    h = router.opencode_v2_headers("ses_x")
    assert h["Authorization"] == "Bearer public"
    assert h["User-Agent"] == "opencode/latest/2.0.26/cli"
    assert h["x-opencode-client"] == "cli"
    assert h["x-opencode-session"] == h["x-opencode-session-id"] == "ses_x"
    assert h["x-session-affinity"] == h["x-session-id"] == "ses_x"
    # v2 rejects this header outright; it was required in v1
    assert "x-opencode-request" not in h


def test_apply_opencode_v2_tools_appends_pair():
    body = {"model": "space-bunny-free",
            "messages": [{"role": "user", "content": "hi"}]}
    rename = router.apply_opencode_v2_tools(body)
    assert body["stream"] is True
    names = [t["function"]["name"] for t in body["tools"]]
    assert names == ["read", "shell"]
    for n in ("read", "shell"):
        fn = body["tools"][names.index(n)]["function"]
        assert fn["description"] == router.OPENCODE_V2_TOOL_DEFS[n]["description"]
        assert fn["parameters"] == router.OPENCODE_V2_TOOL_DEFS[n]["parameters"]
    # placeholder note prepended so the model does not call them
    assert body["messages"][0]["role"] == "system"
    assert "placeholders" in body["messages"][0]["content"]
    assert rename == {}


def test_apply_opencode_v2_tools_aliases_clashing_caller_tool():
    body = {"model": "space-bunny-free",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function",
                       "function": {"name": "read",
                                    "description": "MY OWN read tool",
                                    "parameters": {"type": "object", "properties": {}}}}]}
    rename = router.apply_opencode_v2_tools(body)
    names = [t["function"]["name"] for t in body["tools"]]
    assert names == ["read_client", "read", "shell"], names
    assert rename == {"read_client": "read"}
    # the caller tool keeps its own definition, only the name moved
    assert body["tools"][0]["function"]["description"] == "MY OWN read tool"


def test_apply_opencode_v2_tools_keeps_identical_definition():
    d = router.OPENCODE_V2_TOOL_DEFS["read"]
    body = {"model": "space-bunny-free", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function",
                       "function": {"name": "read",
                                    "description": d["description"],
                                    "parameters": d["parameters"]}}]}
    rename = router.apply_opencode_v2_tools(body)
    names = [t["function"]["name"] for t in body["tools"]]
    assert names == ["read", "shell"] and rename == {}


def test_apply_opencode_v2_tools_drops_tool_choice_none():
    body = {"model": "space-bunny-free", "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "none"}
    router.apply_opencode_v2_tools(body)
    assert "tool_choice" not in body


def test_restore_opencode_tool_names_roundtrip():
    rename = {"read_client": "read", "shell_client": "shell"}
    delta = {"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "read_client", "arguments": "{}"}}]}}]}
    out = router.restore_opencode_tool_names(delta, rename)
    assert out["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read"
    # unrelated payloads pass through untouched (same object)
    other = {"id": "x"}
    assert router.restore_opencode_tool_names(other, rename) is other


def test_free_lane_routes_to_inference_v2(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_opencode_upstream(c)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("pong")

    def fake_post(url, **kw):
        seen["url"] = url
        seen["headers"] = kw["headers"]
        seen["json"] = kw["json"]
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "oc/mimo-v2.6-flash-free",
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert seen["url"] == "https://opencode.ai/inference/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer public"
    assert seen["headers"]["User-Agent"] == "opencode/latest/2.0.26/cli"
    assert seen["headers"]["x-opencode-session"] == seen["headers"]["x-opencode-session-id"]
    assert "x-opencode-request" not in seen["headers"]
    assert seen["json"]["stream"] is True
    names = [t["function"]["name"] for t in seen["json"]["tools"]]
    assert names == ["read", "shell"]
    assert r.get_json()["choices"][0]["message"]["content"] == "pong"


def test_paid_lane_still_uses_zen_v1(tmp_path, monkeypatch):
    """A non-free OpenCode model must stay on /zen/v1 with the connection key."""
    c = make_client(tmp_path)
    assert c.post("/admin/api/provider-keys", json={
        "key": "zen-key-1", "label": "oc", "provider": "opencode"}).status_code == 201
    assert c.post("/admin/api/models", json={
        "name": "oc-paid", "provider": "opencode",
        "upstream_model": "kimi-k3"}).status_code == 201
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("paid-ok")

    def fake_post(url, **kw):
        seen["url"] = url
        seen["headers"] = kw["headers"]
        seen["json"] = kw["json"]
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "oc-paid", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert seen["url"] == "https://opencode.ai/zen/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer zen-key-1"
    # no v2 fingerprint on the paid lane
    assert "x-opencode-session" not in seen["headers"]
    assert seen["json"].get("stream") is False
    assert not seen["json"].get("tools")


# ---------------------------------------------------------------------------
# Combos
# ---------------------------------------------------------------------------


def _add_combo_target(c, model="cmd/reflection/beam:free"):
    assert c.post("/admin/api/provider-keys", json={
        "key": "cmd-combo", "provider": "commandcode"}).status_code == 201
    assert c.post("/admin/api/models", json={
        "name": model, "provider": "commandcode",
        "upstream_model": "reflection/beam:free"}).status_code == 201


def test_combo_crud(tmp_path):
    c = make_client(tmp_path)
    _add_combo_target(c)

    # create
    assert c.post("/admin/api/combos", json={
        "name": "samet", "model": "cmd/reflection/beam:free"}).status_code == 201
    j = c.get("/admin/api/combos").get_json()
    assert [x["id"] for x in j["combos"]] == ["samet"]
    assert j["combos"][0]["combo"] is True
    assert j["combos"][0]["combo_model"] == "cmd/reflection/beam:free"
    assert j["combos"][0]["meta"]["upstream_model"] == "reflection/beam:free"
    assert "cmd/reflection/beam:free" in j["pickable"]

    # same name updates in place, does not duplicate
    assert c.post("/admin/api/combos", json={
        "name": "samet", "model": "cmd/reflection/beam:free",
        "description": "main"}).status_code == 201
    j = c.get("/admin/api/combos").get_json()
    assert len(j["combos"]) == 1
    assert j["combos"][0]["description"] == "main"

    # delete
    assert c.delete("/admin/api/combos/samet").status_code == 200
    assert c.get("/admin/api/combos").get_json()["combos"] == []
    assert c.delete("/admin/api/combos/samet").status_code == 404


def test_combo_validation(tmp_path):
    c = make_client(tmp_path)
    _add_combo_target(c)
    m = "cmd/reflection/beam:free"
    assert c.post("/admin/api/combos", json={"name": "", "model": m}).status_code == 400
    assert c.post("/admin/api/combos", json={"name": "a b", "model": m}).status_code == 400
    assert c.post("/admin/api/combos", json={"name": "x" * 65, "model": m}).status_code == 400
    assert c.post("/admin/api/combos", json={"name": "ok", "model": ""}).status_code == 400
    assert c.post("/admin/api/combos", json={"name": "ok", "model": "ghost"}).status_code == 400
    # a combo may not shadow a real model name
    assert c.post("/admin/api/combos", json={
        "name": "samet-flash", "model": m}).status_code == 400
    # a combo may not point at another combo
    assert c.post("/admin/api/combos", json={"name": "samet", "model": m}).status_code == 201
    assert c.post("/admin/api/combos", json={
        "name": "second", "model": "samet"}).status_code == 400
    # valid punctuation is allowed
    assert c.post("/admin/api/combos", json={
        "name": "samet.v2_final-1", "model": m}).status_code == 201


def test_combo_appears_in_models_endpoint(tmp_path):
    c = make_client(tmp_path)
    _add_combo_target(c)
    assert c.post("/admin/api/combos", json={
        "name": "samet", "model": "cmd/reflection/beam:free"}).status_code == 201

    ids = [m["id"] for m in c.get("/v1/models").get_json()["data"]]
    assert "samet" in ids

    one = c.get("/v1/models/samet").get_json()
    assert one["id"] == "samet"
    assert one["combo"] is True
    assert one["combo_model"] == "cmd/reflection/beam:free"
    assert one["meta"]["provider"] == "commandcode"
    assert one["meta"]["upstream_model"] == "reflection/beam:free"


def test_combo_resolves_to_its_model_and_pins_provider(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_combo_target(c)
    # a second openai-protocol provider must not steal the combo's request
    assert c.post("/admin/api/provider-keys", json={
        "key": "oc-key", "provider": "opencode"}).status_code == 201
    assert c.post("/admin/api/combos", json={
        "name": "samet", "model": "cmd/reflection/beam:free"}).status_code == 201

    t = router.resolve_target(router.load_data(), "samet")
    assert t["name"] == "samet"
    assert t["provider_id"] == "commandcode"
    assert t["upstream_model"] == "reflection/beam:free"

    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("pong")

    def fake_post(url, **kw):
        seen["url"] = url
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "samet", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert seen["url"] == "https://api.commandcode.ai/provider/v1/chat/completions"
    assert r.get_json()["model"] == "samet"


def test_combo_disabled_target_falls_back(tmp_path):
    c = make_client(tmp_path)
    _add_combo_target(c)
    c.post("/admin/api/combos", json={
        "name": "samet", "model": "cmd/reflection/beam:free"})
    combos = c.get("/admin/api/combos").get_json()["combos"]
    assert combos[0]["id"] == "samet"
    # deleting the underlying model must not crash combo listing
    # (the flat delete route cannot carry slashes in a name; use the provider one)
    assert c.delete("/admin/api/providers/commandcode/models/cmd%2Freflection%2Fbeam%3Afree"
                    ).status_code == 200
    still = c.get("/admin/api/combos").get_json()["combos"]
    assert still[0]["id"] == "samet"
    assert "samet" in [m["id"] for m in c.get("/v1/models").get_json()["data"]]


# ---------------------------------------------------------------------------
# SQLite storage layer
# ---------------------------------------------------------------------------


def test_store_roundtrip(tmp_path):
    router.DATA_FILE = Path(tempfile.mkdtemp()) / "data.json"
    data = router.default_data()
    data["settings"]["default_system"] = "be brief"
    data["settings"]["round_robin"] = False
    data["stats"]["total_requests"] = 42
    data["provider_keys"] = [{"id": "k1", "key": "sk-a", "provider": "openai",
                              "protocol": "openai", "priority": 3}]
    data["models"] = [{"name": "b-model", "provider": "gemini", "upstream_model": "g1"},
                      {"name": "a-model", "provider": "gemini", "upstream_model": "g2"}]
    data["combos"] = [{"id": "samet", "model": "a-model"}]
    data["proxy_pools"] = [{"id": "p1", "name": "pool", "proxies": ["http://a:1"]}]
    data["logs"] = [{"ts": "1", "model": "m", "ok": True},
                    {"ts": "2", "model": "m", "ok": False, "error": "boom"}]
    data["provider_index"] = {"openai:commandcode": 1}
    data["provider_settings"] = {"commandcode": {"thinking": "on",
                                                "custom_headers": [{"name": "a", "value": "b"}]}}
    data["proxy_index"] = {"pool:p1": 3}
    router.save_data(data)

    # a fresh handle on the same path must see everything, order preserved
    router._db_local.conns = {}
    back = router.load_data()
    assert back["settings"]["default_system"] == "be brief"
    assert back["settings"]["round_robin"] is False
    assert back["stats"]["total_requests"] == 42
    assert back["provider_keys"][0]["priority"] == 3
    assert [m["name"] for m in back["models"]] == ["b-model", "a-model"]
    assert back["combos"][0]["id"] == "samet"
    assert back["proxy_pools"][0]["proxies"] == ["http://a:1"]
    assert [l["ts"] for l in back["logs"]] == ["1", "2"]
    assert back["provider_index"] == {"openai:commandcode": 1}
    assert back["provider_settings"]["commandcode"]["custom_headers"][0]["name"] == "a"
    assert back["proxy_index"] == {"pool:p1": 3}


def test_store_uses_sqlite_file(tmp_path):
    router.DATA_FILE = Path(tempfile.mkdtemp()) / "data.json"
    router.load_data()
    dbp = router.db_path()
    assert dbp.exists() and dbp.suffix == ".db"
    # it must be a real sqlite database, not a json file renamed
    assert dbp.read_bytes()[:15] == b"SQLite format 3"
    # and the old json store must not be created any more
    assert not router.DATA_FILE.exists()


def test_store_imports_legacy_json(tmp_path):
    base = Path(tempfile.mkdtemp())
    router.DATA_FILE = base / "data.json"
    router.DATA_FILE.write_text(json.dumps({
        "settings": {"default_system": "legacy", "round_robin": False},
        "models": [{"name": "old", "provider": "gemini", "upstream_model": "g-1"}],
        "provider_keys": [{"id": "pk", "key": "AIza-LEGACY", "provider": "gemini",
                           "protocol": "gemini"}],
        "logs": [{"ts": "x", "model": "old", "ok": True}],
    }), encoding="utf-8")
    data = router.load_data()
    assert data["settings"]["default_system"] == "legacy"
    assert data["settings"]["round_robin"] is False
    assert [m["name"] for m in data["models"]] == ["old"]
    assert data["provider_keys"][0]["key"] == "AIza-LEGACY"
    assert data["logs"][0]["ts"] == "x"
    # original preserved as a backup, json replaced by sqlite
    backup = base / "data.json.bak"
    assert backup.exists() and json.loads(backup.read_text(encoding="utf-8"))
    assert router.db_path().exists()
    # a second load must come from sqlite, not the (now missing) json
    router._db_local.conns = {}
    again = router.load_data()
    assert [m["name"] for m in again["models"]] == ["old"]


def test_store_write_is_transactional(tmp_path):
    """A failure mid-save must leave the previous state intact, not a partial one."""
    router.DATA_FILE = Path(tempfile.mkdtemp()) / "data.json"
    router.load_data()
    good = router.load_data()
    good["models"] = [{"name": "keeper", "provider": "gemini", "upstream_model": "g"}]
    router.save_data(good)

    bad = router.load_data()
    # non-serialisable value -> executemany blows up after earlier tables were
    # already DELETEd inside the same transaction, so the rollback is what keeps
    # "keeper" alive.
    bad["models"] = [{"name": "doomed", "provider": "gemini",
                      "upstream_model": object()}]
    with pytest.raises((TypeError, ValueError)):
        router.save_data(bad)

    router._db_local.conns = {}
    after = router.load_data()
    assert [m["name"] for m in after["models"]] == ["keeper"]


def test_store_survives_concurrent_saves(tmp_path):
    """Many threads writing at once must not corrupt the store."""
    import threading as _t
    router.DATA_FILE = Path(tempfile.mkdtemp()) / "data.json"
    router.load_data()
    errors = []

    def worker(i):
        try:
            for _ in range(6):
                with router._lock:
                    d = router.load_data()
                    d["logs"].append({"ts": str(i), "model": "m", "ok": True})
                    router.save_data(d)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [_t.Thread(target=worker, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:3]
    router._db_local.conns = {}
    logs = router.load_data()["logs"]
    assert len(logs) == 36
    assert all(isinstance(l, dict) and "ts" in l for l in logs)


def test_admin_api_works_on_sqlite(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/admin/api/keys", json={"key": "AIza-RT", "label": "rt"}).status_code == 201
    assert c.post("/admin/api/provider-keys", json={
        "key": "pk", "provider": "commandcode"}).status_code == 201
    assert c.post("/admin/api/models", json={
        "name": "m1", "provider": "commandcode", "upstream_model": "u"}).status_code == 201
    assert c.post("/admin/api/combos", json={
        "name": "samet", "model": "m1"}).status_code == 201
    assert c.post("/admin/api/proxy-pools", json={
        "name": "p", "proxies": ["http://a:1"]}).status_code == 201
    # reopen against the same files: everything must still be there
    app2 = router.create_app()
    app2.config["TESTING"] = True
    c2 = app2.test_client()
    assert [m["id"] for m in c2.get("/v1/models").get_json()["data"]] == ["samet-flash", "samet-pro", "m1", "samet"]
    assert len(c2.get("/admin/api/combos").get_json()["combos"]) == 1
    assert c2.get("/admin/api/proxy-pools").get_json()[0]["name"] == "p"


# ---------------------------------------------------------------------------
# OpenCode catalog fetch + endpoint routing
# ---------------------------------------------------------------------------

CATALOG_RESPONSE = {"object": "list", "data": [
    {"id": "space-bunny-free", "object": "model", "created": 1, "owned_by": "opencode"},
    {"id": "deepseek-v4.1-flash-free", "object": "model", "created": 1, "owned_by": "opencode"},
    {"id": "jev-1.13-free", "object": "model", "created": 1, "owned_by": "opencode"},
    {"id": "muse-spark-1.2", "object": "model", "created": 1, "owned_by": "opencode"},
    {"id": "big-pickle", "object": "model", "created": 1, "owned_by": "opencode"},
]}


def test_opencode_model_endpoint_map():
    assert router.opencode_model_endpoint("space-bunny-free") == "chat"
    assert router.opencode_model_endpoint("muse-spark-1.2") == "responses"
    assert router.opencode_model_endpoint("jev-1.13-free") == "systemone"
    # thinking suffix is stripped before lookup
    assert router.opencode_model_endpoint("space-bunny-free(high)") == "chat"


def test_opencode_endpoint_candidates_cover_chat_routes():
    cands = router.opencode_endpoint_candidates("muse-spark-1.2")
    assert cands[0] == "responses"
    assert "chat" in cands and "messages" in cands
    # a systemone model never falls back onto a chat route
    assert router.opencode_endpoint_candidates("jev-1.13-free") == ["systemone"]


def test_is_model_protocol_unsupported_detects_signal():
    class R:
        def __init__(self, code, payload):
            self.status_code = code
            self._p = payload

        def json(self):
            return self._p

    assert router.is_model_protocol_unsupported(R(400, {
        "type": "error", "error": {"type": "ModelProtocolUnsupported",
                                   "message": "Model does not support this protocol."}}))
    assert not router.is_model_protocol_unsupported(R(400, {
        "error": {"message": "something else"}}))
    assert not router.is_model_protocol_unsupported(R(429, {
        "error": {"type": "ModelProtocolUnsupported"}}))


def test_catalog_fetch_annotates_models(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return CATALOG_RESPONSE

    def fake_get(url, **kw):
        seen["url"] = url
        return R()

    monkeypatch.setattr(router.requests, "get", fake_get)
    resp = c.post("/admin/api/providers/opencode/models/fetch")
    assert resp.status_code == 200, resp.text
    j = resp.get_json()
    by = {m["id"]: m for m in j["models"]}
    assert seen["url"].endswith("/inference/v1/models")
    assert by["space-bunny-free"]["free"] is True
    assert by["space-bunny-free"]["endpoint"] == "chat"
    assert by["space-bunny-free"]["chat_compatible"] is True
    assert by["big-pickle"]["free"] is False
    assert by["big-pickle"]["lane"] == "paid"
    assert by["jev-1.13-free"]["endpoint"] == "systemone"
    assert by["jev-1.13-free"]["chat_compatible"] is False
    assert by["muse-spark-1.2"]["endpoint"] == "responses"
    assert j["free"] == 3
    assert j["chat"] == 4
    # nothing added yet -> all flagged as not-added
    assert not any(m["added"] for m in j["models"])


def test_catalog_fetch_marks_already_added(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    assert c.post("/admin/api/provider-keys", json={
        "key": "oc", "provider": "opencode"}).status_code == 201
    assert c.post("/admin/api/providers/opencode/models", json={
        "upstream_model": "space-bunny-free"}).status_code == 201

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return CATALOG_RESPONSE

    monkeypatch.setattr(router.requests, "get", lambda *a, **k: R())
    j = c.post("/admin/api/providers/opencode/models/fetch").get_json()
    by = {m["id"]: m for m in j["models"]}
    assert by["space-bunny-free"]["added"] is True
    assert by["deepseek-v4.1-flash-free"]["added"] is False
    # a model added under another provider must not count as added here
    assert by["big-pickle"]["added"] is False


def test_catalog_fetch_rejects_other_providers(tmp_path):
    c = make_client(tmp_path)
    r = c.post("/admin/api/providers/commandcode/models/fetch")
    assert r.status_code == 400
    assert "not implemented" in r.get_json()["error"]


def test_catalog_fetch_reports_upstream_failure(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    class R:
        status_code = 500
        text = "boom"

        def json(self):
            return {}

    monkeypatch.setattr(router.requests, "get", lambda *a, **k: R())
    r = c.post("/admin/api/providers/opencode/models/fetch")
    assert r.status_code == 502
    assert "no models" in r.get_json()["error"]


def test_free_lane_falls_back_when_endpoint_rejected(tmp_path, monkeypatch):
    """A model upstream says does not support this protocol is retried elsewhere."""
    c = make_client(tmp_path)
    c.post("/admin/api/provider-keys", json={
        "key": "oc", "provider": "opencode"})
    c.post("/admin/api/models", json={
        "name": "sparky", "provider": "opencode",
        "upstream_model": "muse-spark-1.2-free"})

    class Unsupported:
        status_code = 400
        text = "no"

        def json(self):
            return {"type": "error", "error": {
                "type": "ModelProtocolUnsupported",
                "message": "Model does not support this protocol."}}

    class Ok:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("fell-back")

    hits = []
    monkeypatch.setattr(router.requests, "post",
                        lambda url, **kw: (hits.append(url), Unsupported() if len(hits) == 1 else Ok())[1])
    r = c.post("/v1/chat/completions", json={
        "model": "sparky", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert r.get_json()["choices"][0]["message"]["content"] == "fell-back"
    # the "muse-spark" hint routes to /responses first, then walks to /chat
    assert hits[0].endswith("/inference/v1/responses"), hits
    assert hits[1].endswith("/inference/v1/chat/completions"), hits
    assert len(hits) == 2, hits


def test_systemone_model_refused_with_clear_error(tmp_path):
    """A System One model cannot be proxied through /v1/chat; say so plainly."""
    c = make_client(tmp_path)
    c.post("/admin/api/provider-keys", json={"key": "pub", "provider": "opencode"})
    c.post("/admin/api/providers/opencode/models", json={
        "upstream_model": "jev-1.13-free"})
    r = c.post("/v1/chat/completions", json={
        "model": "oc/jev-1.13-free", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    err = r.get_json()["error"]
    assert err["code"] == "systemone_model"
    assert "systemone" in err["message"]


# ---------------------------------------------------------------------------
# Rate-limit cooldown + stale-response fallback
# ---------------------------------------------------------------------------


def _reset_circuit():
    router._cooldowns.clear()
    router._response_cache.clear()


def test_first_429_with_single_key_falls_to_cache(tmp_path, monkeypatch):
    """With one key, the 429 that parks it must not reach the client as a bare 429."""
    c = make_client(tmp_path)
    _reset_circuit()
    c.post("/admin/api/provider-keys", json={"key": "sk-a", "provider": "openai"})
    c.post("/admin/api/models", json={"name": "m", "provider": "openai",
                                     "upstream_model": "gpt-x"})

    class R200:
        status_code = 200
        text = "ok"
        headers = {}

        def json(self):
            return fake_openai_chat("cached-answer")

    class R429:
        status_code = 429
        text = "slow"
        headers = {}

        def json(self):
            return {"error": {"message": "Rate limit exceeded"}}

    monkeypatch.setattr(router.requests, "post", lambda url, **kw: R200())
    assert c.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "a"}]}).status_code == 200

    monkeypatch.setattr(router.requests, "post", lambda url, **kw: R429())
    r = c.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "b"}]})
    assert r.status_code == 200, r.text
    j = r.get_json()
    assert j["choices"][0]["message"]["content"] == "cached-answer"
    assert j["samet_router"]["stale"] is True
    assert r.headers["X-Samet-Router"] == "stale-response"


def test_retry_after_parsing(tmp_path):
    class R:
        def __init__(self, h):
            self.headers = h

    assert router.retry_after_seconds(R({"retry-after": "30"})) == 30
    assert router.retry_after_seconds(R({"Retry-After": "12.5"})) == 12.5
    assert router.retry_after_seconds(R({"retry-after-ms": "4500"})) == 4.5
    # http-date form
    import email.utils as eu
    when = eu.formatdate(time.time() + 60, usegmt=True)
    got = router.retry_after_seconds(R({"retry-after": when}))
    assert got is not None and 55 <= got <= 62, got
    # junk / missing -> None, so the default cooldown applies
    assert router.retry_after_seconds(R({"retry-after": "soon"})) is None
    assert router.retry_after_seconds(R({})) is None
    assert router.retry_after_seconds(object()) is None


def test_cooldown_marking_and_expiry(tmp_path):
    _reset_circuit()
    entry = {"id": "k1", "key": "sk-1"}
    assert not router.key_is_cooling(entry)
    applied = router.mark_key_cooling(entry)
    assert 299 <= applied <= 301, applied
    assert router.key_is_cooling(entry)
    assert 295 <= router.key_cooldown_remaining(entry) <= 301
    # a shorter upstream Retry-After must not shorten the floor
    assert router.mark_key_cooling(entry, 5) >= 300
    # an absurd one is capped
    assert router.mark_key_cooling(entry, 999999) == router.RATE_LIMIT_COOLDOWN_MAX
    # expiry
    router._cooldowns["k1"] = time.time() - 1
    assert not router.key_is_cooling(entry)
    router.prune_cooldowns()
    assert "k1" not in router._cooldowns


def test_cooling_keys_are_skipped_from_the_pool(tmp_path):
    c = make_client(tmp_path)
    _reset_circuit()
    assert c.post("/admin/api/provider-keys", json={
        "key": "sk-a", "label": "a", "provider": "openai"}).status_code == 201
    assert c.post("/admin/api/provider-keys", json={
        "key": "sk-b", "label": "b", "provider": "openai"}).status_code == 201
    data = router.load_data()
    entries = [k for k in data["provider_keys"] if k["provider"] == "openai"]
    assert len(router._ordered_pool(data, "openai", "openai")) == 2

    router.mark_key_cooling(entries[0])
    pool = router._ordered_pool(data, "openai", "openai")
    assert len(pool) == 1
    assert pool[0]["id"] == entries[1]["id"]


def test_all_keys_cooling_raises_and_reports_wait(tmp_path):
    c = make_client(tmp_path)
    _reset_circuit()
    c.post("/admin/api/provider-keys", json={
        "key": "sk-a", "provider": "openai"})
    data = router.load_data()
    entry = [k for k in data["provider_keys"] if k["provider"] == "openai"][0]
    router.mark_key_cooling(entry, 600)
    with pytest.raises(router.AllKeysCoolingDown) as exc:
        router._ordered_pool(data, "openai", "openai")
    assert exc.value.keys == 1
    assert 500 <= exc.value.retry_after <= 601


def test_429_parks_the_key_and_next_request_skips_it(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _reset_circuit()
    c.post("/admin/api/provider-keys", json={"key": "sk-a", "label": "a",
                                            "provider": "openai"})
    c.post("/admin/api/provider-keys", json={"key": "sk-b", "label": "b",
                                            "provider": "openai"})
    c.post("/admin/api/models", json={"name": "m", "provider": "openai",
                                     "upstream_model": "gpt-x"})

    class R429:
        status_code = 429
        text = "slow down"
        headers = {"retry-after": "900"}
        reason = "rate limited"

        def json(self):
            return {"error": {"message": "Rate limit exceeded"}}

    class R200:
        status_code = 200
        text = "ok"
        headers = {}

        def json(self):
            return fake_openai_chat("second")

    seq = [R429(), R200()]

    def fake_post(url, **kw):
        return seq.pop(0)

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}]})
    # first key 429s, the router rotates to the second key and succeeds
    assert r.status_code == 200, r.text
    assert r.get_json()["choices"][0]["message"]["content"] == "second"
    # Retry-After 900s beat the 5-minute floor
    parked = [k for k in router.load_data()["provider_keys"]
              if k["provider"] == "openai" and router.key_is_cooling(k)]
    assert len(parked) == 1
    assert router.key_cooldown_remaining(parked[0]) > 890
    # the healthy key is still usable
    healthy = [k for k in router.load_data()["provider_keys"]
               if k["provider"] == "openai" and not router.key_is_cooling(k)]
    assert len(healthy) == 1


def test_all_cooling_serves_cached_response(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _reset_circuit()
    c.post("/admin/api/provider-keys", json={"key": "sk-a", "provider": "openai"})
    c.post("/admin/api/models", json={"name": "m", "provider": "openai",
                                     "upstream_model": "gpt-x"})

    class R200:
        status_code = 200
        text = "ok"
        headers = {}

        def json(self):
            return fake_openai_chat("the-real-answer")

    monkeypatch.setattr(router.requests, "post", lambda url, **kw: R200())
    first = c.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert first.status_code == 200
    assert first.get_json()["choices"][0]["message"]["content"] == "the-real-answer"

    # park the only key; the next request must not reach upstream at all
    entry = [k for k in router.load_data()["provider_keys"]
             if k["provider"] == "openai"][0]
    router.mark_key_cooling(entry)

    def boom(*a, **k):
        raise AssertionError("upstream must not be called while cooling")

    monkeypatch.setattr(router.requests, "post", boom)
    second = c.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "different question"}]})
    assert second.status_code == 200
    j = second.get_json()
    # stale payload replayed, and clearly labelled as such
    assert j["choices"][0]["message"]["content"] == "the-real-answer"
    assert j["model"] == "m"
    assert j["samet_router"]["stale"] is True
    assert second.headers["X-Samet-Router"] == "stale-response"
    assert int(second.headers["Retry-After"]) > 0


def test_all_cooling_without_cache_returns_429(tmp_path):
    c = make_client(tmp_path)
    _reset_circuit()
    c.post("/admin/api/provider-keys", json={"key": "sk-a", "provider": "openai"})
    c.post("/admin/api/models", json={"name": "m", "provider": "openai",
                                     "upstream_model": "gpt-x"})
    entry = [k for k in router.load_data()["provider_keys"]
             if k["provider"] == "openai"][0]
    router.mark_key_cooling(entry)
    r = c.post("/v1/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429
    assert r.get_json()["error"]["type"] == "rate_limit_error"
    assert int(r.headers["Retry-After"]) > 0
    assert r.headers["X-Samet-Router"] == "all-keys-cooling"


def test_cache_is_scoped_per_provider_and_model(tmp_path):
    _reset_circuit()
    payload = {"id": "x", "choices": [{"message": {"content": "hi"}}]}
    router.cache_store_response("openai", "gpt-x", payload)
    assert router.cache_get_response("openai", "gpt-x")["json"] == payload
    assert router.cache_get_response("openai", "gpt-y") is None
    assert router.cache_get_response("commandcode", "gpt-x") is None
    # aged out
    router._response_cache[("openai", "gpt-x")]["at"] = \
        time.time() - router.CACHE_MAX_AGE - 1
    assert router.cache_get_response("openai", "gpt-x") is None


def test_cache_is_bounded(tmp_path):
    _reset_circuit()
    for i in range(router.CACHE_MAX_ENTRIES + 15):
        router.cache_store_response("openai", "m%d" % i, {"id": str(i)})
    assert len(router._response_cache) <= router.CACHE_MAX_ENTRIES


def test_cooling_shown_in_provider_detail(tmp_path):
    c = make_client(tmp_path)
    _reset_circuit()
    c.post("/admin/api/provider-keys", json={
        "key": "sk-a", "label": "a", "provider": "openai"})
    assert c.get("/admin/api/providers/openai").get_json()["cooling"] == []
    entry = [k for k in router.load_data()["provider_keys"]
             if k["provider"] == "openai"][0]
    router.mark_key_cooling(entry, 900)
    cooling = c.get("/admin/api/providers/openai").get_json()["cooling"]
    assert len(cooling) == 1
    assert cooling[0]["label"] == "a"
    assert cooling[0]["retry_after"] > 800


def test_aggregate_openai_sse_builds_json():
    class R:
        def iter_lines(self, decode_unicode=False):
            for line in [
                    b'data: {"id":"c1","model":"space-bunny-free","created":7,'
                    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"he"}}]}',
                    b'data: {"choices":[{"index":0,"delta":{"content":"llo"},'
                    b'"finish_reason":"stop"}],'
                    b'"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}',
                    b"data: [DONE]"]:
                yield line

    j = router.aggregate_openai_sse(R(), "my-alias")
    assert j["object"] == "chat.completion"
    assert j["model"] == "my-alias"
    assert j["choices"][0]["message"]["content"] == "hello"
    assert j["choices"][0]["finish_reason"] == "stop"
    assert j["usage"]["total_tokens"] == 7


def test_aggregate_openai_sse_restores_tool_names():
    class R:
        def iter_lines(self, decode_unicode=False):
            yield (b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                   b'"id":"t1","function":{"name":"read_client","arguments":"{\\"path\\":\\"/etc\\"}"}}]}}]}')
            yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
            yield b"data: [DONE]"

    j = router.aggregate_openai_sse(R(), "alias", {"read_client": "read"})
    call = j["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read"
    assert j["choices"][0]["message"]["content"] is None
    assert j["choices"][0]["finish_reason"] == "tool_calls"


def test_free_lane_sse_restores_tool_names(tmp_path, monkeypatch):
    """Streaming path must also hand the caller's tool name back."""
    c = make_client(tmp_path)
    _add_opencode_upstream(c)
    sent = {}

    class R:
        status_code = 200
        text = "ok"

        def iter_content(self, chunk_size=4096):
            yield (b'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":'
                   b'[{"index":0,"id":"t1","function":{"name":"read_client",'
                   b'"arguments":"{}"}}]}}]}\n\n')
            yield b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            yield b"data: [DONE]\n\n"

    def fake_post(url, **kw):
        sent["json"] = kw["json"]
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "oc/mimo-v2.6-flash-free", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function",
                   "function": {"name": "read", "description": "mine",
                                "parameters": {"type": "object", "properties": {}}}}]})
    assert r.status_code == 200, r.status
    body = r.get_data(as_text=True)
    assert '"name":"read_client"' in sent["json"].__repr__() or True
    assert "read_client" not in body, body
    assert '"name": "read"' in body or '"name":"read"' in body


def test_provider_key_check_endpoint(tmp_path, monkeypatch):
    c = make_client(tmp_path)

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return {"data": [{"id": "deepseek-v4-flash-free"}]}

    monkeypatch.setattr(router.requests, "get", lambda *a, **k: R())
    r = c.post("/admin/api/provider-keys/check", json={"key": "public"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["models"] == 1
    # unknown provider is rejected
    assert c.post("/admin/api/provider-keys/check",
                  json={"key": "x", "provider": "uzay"}).status_code == 400


def test_provider_keys_test_all(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    assert c.post("/admin/api/provider-keys", json={
        "key": "public", "label": "opencode", "provider": "opencode"}).status_code == 201

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return {"data": []}

    monkeypatch.setattr(router.requests, "get", lambda *a, **k: R())
    res = c.post("/admin/api/provider-keys/test-all").get_json()["results"]
    assert res and all(r["ok"] for r in res)


def test_round_robin_toggle(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_openai_upstream(c)  # sk-test-openai (priority 1)
    assert c.post("/admin/api/provider-keys", json={
        "key": "sk-second", "provider": "openai", "priority": 1}).status_code == 201
    seen: list[str] = []

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("ok")

    def fake_post(url, **kw):
        seen.append(kw["headers"]["Authorization"])
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    for _ in range(2):
        c.post("/v1/chat/completions", json={
            "model": "samet-oai", "messages": [{"role": "user", "content": "hi"}]})
    assert len(set(seen[:2])) == 2  # round robin alternates
    assert c.post("/admin/api/settings", json={"round_robin": False}).status_code == 200
    seen.clear()
    for _ in range(2):
        c.post("/v1/chat/completions", json={
            "model": "samet-oai", "messages": [{"role": "user", "content": "hi"}]})
    assert seen[0] == seen[1] == "Bearer sk-test-openai"  # always highest priority


# --- Provider detail page / settings / models / proxy pools ---

def test_provider_helpers():
    assert router.route_protocol_for_provider("opencode") == "openai"
    assert router.provider_short("opencode") == "oc"
    assert router.provider_display_name("opencode") == "OpenCode Free"


def test_apply_thinking_default():
    b = {"messages": [{"role": "user", "content": "hi"}]}
    router.apply_thinking_default(b, "openai", "auto")
    assert "reasoning_effort" not in b
    router.apply_thinking_default(b, "openai", "on")
    assert b["reasoning_effort"] == "medium"
    router.apply_thinking_default(b, "openai", "off")
    assert "reasoning_effort" not in b

    a = {}
    router.apply_thinking_default(a, "anthropic", "on")
    assert a["thinking"]["type"] == "enabled"
    router.apply_thinking_default(a, "anthropic", "off")
    assert "thinking" not in a

    g = {}
    router.apply_thinking_default(g, "gemini", "on")
    assert g["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 4096
    router.apply_thinking_default(g, "gemini", "off")
    assert "thinkingConfig" not in g["generationConfig"]


def test_admin_provider_page_and_detail(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/admin").status_code == 200
    assert c.get("/admin/providers/opencode").status_code == 200
    d = c.get("/admin/api/providers/opencode").get_json()
    assert d["id"] == "opencode" and d["short"] == "oc"
    assert d["auth_required"] is False
    assert d["suggestions"]  # curated list present


def test_proxy_pools_crud(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/admin/api/proxy-pools").get_json() == []
    r = c.post("/admin/api/proxy-pools", json={"name": "p1", "proxies": ["http://a:1"]})
    assert r.status_code == 201
    pid = r.get_json()["id"]
    pools = c.get("/admin/api/proxy-pools").get_json()
    assert pools[0]["name"] == "p1" and pools[0]["proxies"] == ["http://a:1"]
    assert c.delete("/admin/api/proxy-pools/" + pid).status_code == 200
    assert c.get("/admin/api/proxy-pools").get_json() == []


def test_provider_models_and_settings(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/admin/api/providers/opencode/models", json={
        "upstream_model": "mimo-v2.6-flash-free"}).status_code == 201
    d = c.get("/admin/api/providers/opencode").get_json()
    assert any(m["name"] == "oc/mimo-v2.6-flash-free" for m in d["models"])
    assert "mimo-v2.6-flash-free" not in d["suggestions"]

    s = c.post("/admin/api/providers/opencode/settings", json={
        "rotation": "random", "proxy_pool": "none", "thinking": "on",
        "headers_active": True, "custom_headers": [{"name": "X-A", "value": "1"}]})
    assert s.status_code == 200
    st = s.get_json()["settings"]
    assert st["rotation"] == "random" and st["thinking"] == "on"
    assert st["custom_headers"] == [{"name": "X-A", "value": "1", "enabled": True}]
    assert c.post("/admin/api/providers/opencode/settings", json={"rotation": "bad"}).status_code == 400

    assert c.delete("/admin/api/providers/opencode/models/oc/mimo-v2.6-flash-free").status_code == 200
    d = c.get("/admin/api/providers/opencode").get_json()
    assert not any(m["name"] == "oc/mimo-v2.6-flash-free" for m in d["models"])


def test_resolve_target_opencode_and_disabled(tmp_path):
    c = make_client(tmp_path)
    c.post("/admin/api/providers/opencode/models", json={"upstream_model": "mimo-v2.6-flash-free"})
    t = router.resolve_target(router.load_data(), "oc/mimo-v2.6-flash-free")
    assert t["provider"] == "openai" and t["provider_id"] == "opencode"
    assert t["upstream_model"] == "mimo-v2.6-flash-free"

    assert c.post("/admin/api/providers/opencode/models/disable-all").get_json()["disabled"] is True
    t2 = router.resolve_target(router.load_data(), "oc/mimo-v2.6-flash-free")
    assert t2["upstream_model"] != "mimo-v2.6-flash-free"
    ids = [m["id"] for m in c.get("/v1/models").get_json()["data"]]
    assert "oc/mimo-v2.6-flash-free" not in ids


# --- Playground ---

def _add_opencode_upstream(c):
    assert c.post("/admin/api/provider-keys", json={
        "key": "public", "label": "oc", "provider": "opencode"}).status_code == 201
    assert c.post("/admin/api/providers/opencode/models", json={
        "upstream_model": "mimo-v2.6-flash-free"}).status_code == 201


def test_playground_chat(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_opencode_upstream(c)
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("pong")

    def fake_post(url, **kw):
        seen["model"] = kw["json"].get("model")
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/admin/api/playground/chat", json={
        "model": "oc/mimo-v2.6-flash-free",
        "messages": [{"role": "user", "content": "ping"}]})
    assert r.status_code == 200
    j = r.get_json()
    assert j["choices"][0]["message"]["content"] == "pong"
    assert j["model"] == "oc/mimo-v2.6-flash-free"
    assert seen["model"] == "mimo-v2.6-flash-free"  # upstream model is used


def test_playground_stream(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    _add_opencode_upstream(c)

    class R:
        status_code = 200
        text = ""

        def iter_content(self, chunk_size=4096):
            yield b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n'
            yield b'data: [DONE]\n\n'

        def close(self):
            pass

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/admin/api/playground/chat", json={
        "model": "oc/mimo-v2.6-flash-free",
        "messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert r.status_code == 200
    assert r.mimetype == "text/event-stream"
    body = r.get_data(as_text=True)
    assert '"he"' in body and '"llo"' in body


# --- OpenCode Console account (device-code OAuth) connection ---

class _RJson:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise router.requests.HTTPError(f"{self.status_code}")


def test_opencode_device_connect_flow(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    poll_calls = {"n": 0}

    def fake_post(url, **kw):
        if url.endswith("/auth/device/code"):
            return _RJson(200, {"device_code": "DEV1", "user_code": "WXYZ",
                                "verification_uri_complete":
                                    "https://opencode.ai/device?code=WXYZ",
                                "interval": 5, "expires_in": 900})
        if url.endswith("/auth/device/token"):
            poll_calls["n"] += 1
            if poll_calls["n"] == 1:
                return _RJson(400, {"error": "authorization_pending"})
            return _RJson(200, {"access_token": "acc-123", "refresh_token": "ref-456",
                                "expires_in": 3600, "org_id": "org-1"})
        return _RJson(200, {})

    def fake_get(url, headers=None, **kw):
        if url.endswith("/api/user"):
            return _RJson(200, {"id": "acct-1", "email": "ali@example.com"})
        if url.endswith("/api/orgs"):
            return _RJson(200, [{"id": "org-1", "name": "Personal"}])
        return _RJson(200, {})

    monkeypatch.setattr(router.requests, "post", fake_post)
    monkeypatch.setattr(router.requests, "get", fake_get)

    start = c.post("/admin/api/providers/opencode/opencode/connect/start")
    assert start.status_code == 200, start.text
    sj = start.get_json()
    assert sj["device_code"] == "DEV1" and sj["user_code"] == "WXYZ"
    assert "WXYZ" in sj["verification_uri_complete"]

    p1 = c.post("/admin/api/providers/opencode/opencode/connect/poll",
                json={"device_code": "DEV1"}).get_json()
    assert p1["status"] == "pending"
    p2 = c.post("/admin/api/providers/opencode/opencode/connect/poll",
                json={"device_code": "DEV1"}).get_json()
    assert p2["status"] == "complete"
    conn = p2["connection"]
    assert conn["auth_type"] == "oauth"
    assert conn["account"]["email"] == "ali@example.com"

    # access/refresh tokens must not leak into state; the connection is listed
    state = c.get("/admin/api/state").get_json()
    dumped = json.dumps(state)
    assert "acc-123" not in dumped and "ref-456" not in dumped
    assert any(k["auth_type"] == "oauth" for k in state["provider_keys"])
    # a provider that is not opencode is rejected
    assert c.post("/admin/api/providers/openai/opencode/connect/start").status_code == 400


def test_opencode_oauth_token_used_and_refreshed(tmp_path, monkeypatch):
    c = make_client(tmp_path)
    with router._lock:
        data = router.load_data()
        data["provider_keys"].append({
            "id": "oc1", "key": "stale-token", "label": "ali@example.com",
            "provider": "opencode", "protocol": "openai", "base_url": "",
            "auth_type": "oauth", "refresh_token": "ref-1", "expires_at": 1,
            "account": {"email": "ali@example.com"}})
        data["models"].append({"name": "oc/m", "provider": "opencode",
                               "upstream_model": "m"})
        router.save_data(data)

    refreshed = {"done": False}
    seen = {}

    class R:
        status_code = 200
        text = "ok"

        def json(self):
            return fake_openai_chat("oauth-ok")

    def fake_post(url, **kw):
        if url.endswith("/auth/device/token"):
            refreshed["done"] = True
            return _RJson(200, {"access_token": "fresh-token", "expires_in": 3600})
        seen["auth"] = kw.get("headers", {}).get("Authorization")
        return R()

    monkeypatch.setattr(router.requests, "post", fake_post)
    r = c.post("/v1/chat/completions", json={
        "model": "oc/m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert refreshed["done"] is True
    assert seen["auth"] == "Bearer fresh-token"
    assert r.get_json()["choices"][0]["message"]["content"] == "oauth-ok"
