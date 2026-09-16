"""Samet Router tests - no network access, requests.post is mocked."""
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
    assert {p["id"] for p in cat} == {"openai", "anthropic", "gemini", "custom"}
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
    c = make_client(tmp_path)
    _add_openai_upstream(c)

    class R:
        status_code = 429
        text = '{"error": {"message": "hiz limiti"}}'

        def json(self):
            return {"error": {"message": "hiz limiti"}}

    monkeypatch.setattr(router.requests, "post", lambda *a, **k: R())
    r = c.post("/v1/chat/completions", json={
        "model": "samet-oai", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429
    assert "hiz limiti" in r.text


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
        "name": "a", "provider": "openai", "upstream_model": "gpt-x"}
    assert router.resolve_target(data, "gemini-2.5-pro")["upstream_model"] == "gemini-2.5-pro"
    assert router.resolve_target(data, "bilinmeyen")["upstream_model"] == "gpt-x"
    chunks = [b'data: {"model": "up", "message": {"model": "up"}}\n', b"data: [DONE]\n"]
    text = b"".join(router.rewrite_sse_model(chunks, "router-ad")).decode()
    assert '"model": "router-ad"' in text and '"up"' not in text
    assert router.body_has_thinking("anthropic", {"thinking": {"type": "enabled"}}) is True
    assert router.body_has_thinking("openai", {"reasoning_effort": "low"}) is True
    assert router.body_has_thinking("openai", {"messages": []}) is False
