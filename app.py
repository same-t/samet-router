"""Samet Router - Gemini backend'li, OpenAI/Anthropic/Gemini uyumlu mini LLM router.

Mimari:
  Client (OpenAI SDK / Anthropic SDK / Gemini SDK)
      -> Samet Router (bu dosya: Flask)
      -> Google Gemini API (gercek LLM cagrisi, arkada)

Admin panel (/admin):
  Sidebar'li tek sayfa arayuz. Gemini API key ve model eslemesi eklenir/silinir.
  Tum veri JSON dosyada saklanir (default: ./data.json).

Router endpoint'leri:
  GET  /health
  GET  /v1/models                                  -> OpenAI uyumlu model listesi
  POST /v1/chat/completions                        -> OpenAI uyumlu chat
  POST /v1/messages                                -> Anthropic uyumlu messages
  POST /v1beta/models/<model>:generateContent      -> Gemini uyumlu passthrough
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, redirect, render_template, request

DATA_FILE = Path(os.environ.get("SAMET_ROUTER_DATA", str(Path(__file__).parent / "data.json")))
ROUTER_API_KEY = os.environ.get("ROUTER_API_KEY", "").strip()  # bossa istemciler Bearer ile gelir
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()  # doluysa admin API'leri Bearer ister
GEMINI_BASE = "https://generativelanguage.googleapis.com"

_lock = threading.Lock()
_key_cycle = None  # round-robin icin
_key_ids: list[str] = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_data() -> dict:
    return {
        "gemini_keys": [],  # [{id, key, label, created_at}]
        "models": [  # router model adi -> gercek gemini modeli
            {"name": "samet-flash", "gemini_model": "gemini-2.0-flash", "description": "Varsayilan hizli model"},
            {"name": "samet-pro", "gemini_model": "gemini-2.0-flash", "description": "Pro takma adi (ayni backend)"},
        ],
        "logs": [],  # [{ts, endpoint, model, ok, latency_ms, error}]
        "key_index": 0,
    }


def load_data() -> dict:
    if not DATA_FILE.exists():
        data = default_data()
        save_data(data)
        return data
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = default_data()
    for k in default_data():
        data.setdefault(k, default_data()[k])
    return data


def save_data(data: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DATA_FILE)


def reset_key_cycle(data: dict) -> None:
    global _key_cycle, _key_ids
    _key_ids = [k["id"] for k in data.get("gemini_keys", [])]
    _key_cycle = itertools.cycle(_key_ids) if _key_ids else None
    # kayitli index'e ilerle (basit round-robin)
    idx = int(data.get("key_index", 0) or 0)
    if _key_cycle is not None and _key_ids:
        for _ in range(idx % max(len(_key_ids), 1)):
            next(_key_cycle, None)


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]


# ---------------------------------------------------------------------------
# Donusum fonksiyonlari (saf fonksiyonlar -> testi kolay)
# ---------------------------------------------------------------------------

def resolve_gemini_model(data: dict, router_model: str) -> str:
    """Router'daki takma adi gercek Gemini modeline cevir, eslesme yoksa aynen dondur."""
    for m in data.get("models", []):
        if m.get("name") == router_model:
            return m.get("gemini_model", router_model)
    return router_model or "gemini-2.0-flash"


def openai_messages_to_gemini(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI messages -> (system_instruction, gemini contents)."""
    system_text = None
    contents: list[dict] = []
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):  # [{type:text, text:...}] formunu duzlestir
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if role == "system":
            system_text = (system_text + "\n" if system_text else "") + str(content)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": str(content)}]})
    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
    return system_text, contents


def anthropic_messages_to_gemini(system, messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Anthropic (system + messages) -> (system_instruction, gemini contents)."""
    if isinstance(system, list):
        system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
    gemini_contents: list[dict] = []
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        gemini_role = "model" if role == "assistant" else "user"
        gemini_contents.append({"role": gemini_role, "parts": [{"text": str(content)}]})
    if not gemini_contents:
        gemini_contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
    return (str(system) if system else None), gemini_contents


def build_gemini_payload(
    system_text: str | None,
    contents: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict:
    payload: dict = {"contents": contents}
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    gen_cfg: dict = {}
    if max_tokens:
        try:
            gen_cfg["maxOutputTokens"] = int(max_tokens)
        except (TypeError, ValueError):
            pass
    if temperature is not None:
        try:
            gen_cfg["temperature"] = float(temperature)
        except (TypeError, ValueError):
            pass
    if gen_cfg:
        payload["generationConfig"] = gen_cfg
    return payload


def gemini_text_of_response(gemini_json: dict) -> str:
    try:
        cands = gemini_json.get("candidates", [])
        parts = cands[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    except (IndexError, AttributeError, TypeError):
        return ""


def gemini_to_openai_response(gemini_json: dict, router_model: str) -> dict:
    text = gemini_text_of_response(gemini_json)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": router_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def gemini_to_anthropic_response(gemini_json: dict, router_model: str) -> dict:
    text = gemini_text_of_response(gemini_json)
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": router_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Gemini cagrisi (key rotation'li)
# ---------------------------------------------------------------------------

def call_gemini(gemini_model: str, payload: dict, timeout: int = 60) -> dict:
    """Siradaki key ile Gemini'ye POST at, 401/403/429'da sonraki key'e gec."""
    with _lock:
        data = load_data()
        keys = [k["key"] for k in data.get("gemini_keys", []) if k.get("key")]
        if not keys:
            raise RuntimeError("Gemini API key eklenmemis. Once /admin panelinden key ekleyin.")
        start_idx = int(data.get("key_index", 0) or 0) % len(keys)

    last_err: Exception | None = None
    for attempt in range(len(keys)):
        key = keys[(start_idx + attempt) % len(keys)]
        url = f"{GEMINI_BASE}/v1beta/models/{gemini_model}:generateContent?key={key}"
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
            continue
        if resp.status_code == 200:
            with _lock:
                data = load_data()
                data["key_index"] = (start_idx + attempt + 1) % len(keys)
                save_data(data)
            return resp.json()
        last_err = RuntimeError(f"Gemini {resp.status_code}: {resp.text[:500]}")
        if resp.status_code not in (401, 403, 429):
            break
    raise last_err or RuntimeError("Gemini cagrisi basarisiz.")


def add_log(endpoint: str, model: str, ok: bool, latency_ms: int, error: str = "") -> None:
    with _lock:
        data = load_data()
        logs = data.get("logs", [])
        logs.append(
            {"ts": _now_iso(), "endpoint": endpoint, "model": model,
             "ok": ok, "latency_ms": latency_ms, "error": error[:300]}
        )
        data["logs"] = logs[-100:]
        save_data(data)


# ---------------------------------------------------------------------------
# Flask uygulamasi
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    with _lock:
        reset_key_cycle(load_data())

    def need_router_auth() -> tuple[bool, dict | None]:
        if not ROUTER_API_KEY:
            return True, None
        auth = request.headers.get("Authorization", "")
        if auth == f"Bearer {ROUTER_API_KEY}":
            return True, None
        return False, (jsonify({"error": "unauthorized, Bearer token gerekli"}), 401)

    def need_admin_auth():
        if not ADMIN_TOKEN:
            return None
        if request.headers.get("Authorization", "") == f"Bearer {ADMIN_TOKEN}":
            return None
        return jsonify({"error": "unauthorized"}), 401

    @app.get("/")
    def index():
        return redirect("/admin")

    @app.get("/health")
    def health():
        data = load_data()
        return jsonify({
            "ok": True, "service": "samet-router",
            "keys": len(data.get("gemini_keys", [])),
            "models": len(data.get("models", [])),
            "time": _now_iso(),
        })

    @app.get("/admin")
    def admin():
        data = load_data()
        return render_template(
            "admin.html",
            keys=[{**k, "key": mask_key(k.get("key", ""))} for k in data.get("gemini_keys", [])],
            models=data.get("models", []),
            logs=list(reversed(data.get("logs", [])[-20:])),
        )

    # --- Admin JSON API ---
    @app.get("/admin/api/state")
    def admin_state():
        if (err := need_admin_auth()) is not None:
            return err
        data = load_data()
        return jsonify({
            "keys": [{**k, "key": mask_key(k.get("key", ""))} for k in data.get("gemini_keys", [])],
            "models": data.get("models", []),
            "logs": list(reversed(data.get("logs", [])[-50:])),
            "router_auth_required": bool(ROUTER_API_KEY),
        })

    @app.post("/admin/api/keys")
    def admin_add_key():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        key = str(body.get("key", "")).strip()
        label = str(body.get("label", "")).strip() or "gemini-key"
        if not key:
            return jsonify({"error": "key bos olamaz"}), 400
        with _lock:
            data = load_data()
            if any(k.get("key") == key for k in data["gemini_keys"]):
                return jsonify({"error": "bu key zaten ekli"}), 400
            data["gemini_keys"].append({
                "id": uuid.uuid4().hex[:8], "key": key, "label": label,
                "created_at": _now_iso(),
            })
            save_data(data)
            reset_key_cycle(data)
        return jsonify({"ok": True}), 201

    @app.delete("/admin/api/keys/<key_id>")
    def admin_del_key(key_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        with _lock:
            data = load_data()
            data["gemini_keys"] = [k for k in data["gemini_keys"] if k.get("id") != key_id]
            save_data(data)
            reset_key_cycle(data)
        return jsonify({"ok": True})

    @app.post("/admin/api/models")
    def admin_add_model():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        name = str(body.get("name", "")).strip()
        gemini_model = str(body.get("gemini_model", "")).strip() or "gemini-2.0-flash"
        desc = str(body.get("description", "")).strip()
        if not name:
            return jsonify({"error": "model adi (name) bos olamaz"}), 400
        with _lock:
            data = load_data()
            data["models"] = [m for m in data["models"] if m.get("name") != name]
            data["models"].append({"name": name, "gemini_model": gemini_model, "description": desc})
            save_data(data)
        return jsonify({"ok": True}), 201

    @app.delete("/admin/api/models/<name>")
    def admin_del_model(name: str):
        if (err := need_admin_auth()) is not None:
            return err
        with _lock:
            data = load_data()
            data["models"] = [m for m in data["models"] if m.get("name") != name]
            save_data(data)
        return jsonify({"ok": True})

    # --- Router: OpenAI uyumlu ---
    @app.get("/v1/models")
    def list_models():
        ok, err = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        data = load_data()
        return jsonify({
            "object": "list",
            "data": [
                {"id": m["name"], "object": "model", "owned_by": "samet-router",
                 "meta": {"gemini_model": m.get("gemini_model")}}
                for m in data.get("models", [])
            ],
        })

    @app.post("/v1/chat/completions")
    def chat_completions():
        ok, err = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        body = request.get_json(force=True, silent=True) or {}
        router_model = str(body.get("model", "")).strip() or "samet-flash"
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        if stream:
            return jsonify({"error": "stream henuz desteklenmiyor, stream:false gonderin"}), 400
        t0 = time.time()
        try:
            with _lock:
                data = load_data()
            gemini_model = resolve_gemini_model(data, router_model)
            system_text, contents = openai_messages_to_gemini(messages)
            payload = build_gemini_payload(
                system_text, contents,
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
            )
            g = call_gemini(gemini_model, payload)
            add_log("openai:/v1/chat/completions", router_model, True,
                    int((time.time() - t0) * 1000))
            return jsonify(gemini_to_openai_response(g, router_model))
        except Exception as exc:  # noqa: BLE001 - hata istemciye sade doner
            add_log("openai:/v1/chat/completions", router_model, False,
                    int((time.time() - t0) * 1000), str(exc))
            return jsonify({"error": str(exc)[:500]}), 502

    # --- Router: Anthropic uyumlu ---
    @app.post("/v1/messages")
    def anthropic_messages():
        ok, err = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        body = request.get_json(force=True, silent=True) or {}
        router_model = str(body.get("model", "")).strip() or "samet-flash"
        t0 = time.time()
        try:
            with _lock:
                data = load_data()
            gemini_model = resolve_gemini_model(data, router_model)
            system_text, contents = anthropic_messages_to_gemini(
                body.get("system"), body.get("messages", [])
            )
            payload = build_gemini_payload(
                system_text, contents,
                max_tokens=body.get("max_tokens"),
                temperature=body.get("temperature"),
            )
            g = call_gemini(gemini_model, payload)
            add_log("anthropic:/v1/messages", router_model, True,
                    int((time.time() - t0) * 1000))
            return jsonify(gemini_to_anthropic_response(g, router_model))
        except Exception as exc:  # noqa: BLE001
            add_log("anthropic:/v1/messages", router_model, False,
                    int((time.time() - t0) * 1000), str(exc))
            return jsonify({"type": "error", "error": {"type": "api_error",
                            "message": str(exc)[:500]}}), 502

    # --- Router: Gemini uyumlu passthrough ---
    @app.post("/v1beta/models/<path:model_name>:generateContent")
    def gemini_passthrough(model_name: str):
        ok, err = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        body = request.get_json(force=True, silent=True) or {}
        t0 = time.time()
        try:
            with _lock:
                data = load_data()
            gemini_model = resolve_gemini_model(data, model_name)
            g = call_gemini(gemini_model, body or {"contents": []})
            add_log(f"gemini:/v1beta/models/{model_name}:generateContent",
                    model_name, True, int((time.time() - t0) * 1000))
            return jsonify(g)
        except Exception as exc:  # noqa: BLE001
            add_log(f"gemini:/v1beta/models/{model_name}:generateContent",
                    model_name, False, int((time.time() - t0) * 1000), str(exc))
            return jsonify({"error": {"message": str(exc)[:500]}}), 502

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
