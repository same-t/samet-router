"""Samet Router - multi-provider mini LLM router compatible with OpenAI/Anthropic/Gemini.

Architecture:
  Client (OpenAI SDK / Anthropic SDK / Claude Code / Gemini SDK)
      -> Samet Router (this file: Flask)
      -> Upstream provider (OpenAI / Anthropic / Gemini / OpenAI-compatible custom endpoint)

Each model alias is bound to a provider; for each provider key the
protocol (openai | anthropic | gemini) and base_url can be configured.
For example an OpenAI key + openai protocol, an Anthropic key + anthropic
protocol, or an OpenAI-compatible server with a custom base_url.

Admin panel (/admin):
  Single-page UI with a sidebar. Provider keys, model mappings and a
  default system prompt can be added/removed.
  All data is stored in a JSON file (default: ./data.json).

Router endpoints:
  GET  /health
  GET  /v1/models                                  -> OpenAI + Anthropic compatible model list
  GET  /v1/models/<model_id>                       -> single model detail
  POST /v1/chat/completions                        -> OpenAI compatible chat (incl. tools + stream SSE)
  POST /v1/messages                                -> Anthropic compatible messages (incl. tools + stream/thinking)
  POST /v1/messages/count_tokens                   -> Anthropic token counting (estimated)
  POST /v1beta/models/<model>:generateContent      -> Gemini compatible (passthrough to gemini provider,
                                                      converting for other providers)
  POST /v1beta/models/<model>:streamGenerateContent -> Gemini stream fallback (returns JSON)

Admin API:
  GET    /admin/api/state                          -> provider_keys + router_keys + stats + logs
  GET    /admin/api/stats                          -> dashboard summary
  DELETE /admin/api/stats                          -> reset stats + logs
  POST   /admin/api/router-keys {label}            -> generate user API key (raw key returned once)
  DELETE /admin/api/router-keys/<id>               -> revoke key
  GET    /admin/api/providers                      -> provider catalog (protocol/base_url defaults)
  POST   /admin/api/provider-keys {key,label,provider,protocol?,base_url?}
  DELETE /admin/api/provider-keys/<id>
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, redirect, render_template, request, stream_with_context

DATA_FILE = Path(os.environ.get("SAMET_ROUTER_DATA", str(Path(__file__).parent / "data.json")))
ROUTER_API_KEY = os.environ.get("ROUTER_API_KEY", "").strip()  # if empty, clients come in with Bearer
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()  # if set, admin APIs require Bearer
GEMINI_BASE = "https://generativelanguage.googleapis.com"
OPENAI_BASE = "https://api.openai.com/v1"
ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDERS = ("openai", "anthropic", "gemini", "custom")
PROTOCOLS = ("openai", "anthropic", "gemini")

# provider -> defaults (protocol, base_url, auth scheme)
PROVIDER_DEFAULTS: dict[str, dict] = {
    "openai": {"protocol": "openai", "base_url": OPENAI_BASE, "auth": "bearer",
               "key_hint": "sk-...", "key_url": "https://platform.openai.com/api-keys"},
    "anthropic": {"protocol": "anthropic", "base_url": ANTHROPIC_BASE, "auth": "x-api-key",
                  "key_hint": "sk-ant-...", "key_url": "https://console.anthropic.com/"},
    "gemini": {"protocol": "gemini", "base_url": GEMINI_BASE, "auth": "query",
               "key_hint": "AIza...", "key_url": "https://aistudio.google.com/apikey"},
    "custom": {"protocol": "openai", "base_url": "", "auth": "bearer",
               "key_hint": "server key", "key_url": ""},
}

STREAM_TIMEOUT = (10, 180)

_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_data() -> dict:
    return {
        # upstream provider keys:
        # [{id, key, label, provider, protocol, base_url, created_at}]
        # provider: openai | anthropic | gemini | custom
        # protocol: openai | anthropic | gemini (which format is used to talk to upstream)
        # base_url: if empty, the provider default is used
        "provider_keys": [],
        "router_keys": [],  # user API keys [{id, key, prefix, label, created_at, last_used_at, request_count}]
        "models": [  # router model name -> {provider, upstream_model}
            {"name": "samet-flash", "provider": "gemini", "upstream_model": "gemini-2.0-flash",
             "description": "Default fast model"},
            {"name": "samet-pro", "provider": "gemini", "upstream_model": "gemini-2.0-flash",
             "description": "Pro alias (same backend)"},
        ],
        "logs": [],  # [{ts, endpoint, model, provider, thinking, ok, latency_ms, error, ...}]
        "provider_index": {},  # provider -> round-robin order
        "settings": {"default_system": ""},
        # cumulative counters (unaffected by log rotation)
        "stats": {"total_requests": 0, "ok": 0, "err": 0,
                  "input_tokens": 0, "output_tokens": 0},
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
    base = default_data()
    for k, v in base.items():
        data.setdefault(k, v)
    # also migrate the settings content
    if not isinstance(data.get("settings"), dict):
        data["settings"] = {"default_system": ""}
    data["settings"].setdefault("default_system", "")
    if not isinstance(data.get("router_keys"), list):
        data["router_keys"] = []
    if not isinstance(data.get("provider_keys"), list):
        data["provider_keys"] = []
    if not isinstance(data.get("provider_index"), dict):
        data["provider_index"] = {}
    # --- legacy migration: gemini_keys -> provider_keys ---
    old_gemini = data.pop("gemini_keys", None)
    if old_gemini:
        have = {k.get("key") for k in data["provider_keys"]}
        for k in old_gemini:
            if isinstance(k, dict) and k.get("key") and k["key"] not in have:
                data["provider_keys"].append({
                    "id": k.get("id") or uuid.uuid4().hex[:8],
                    "key": k["key"],
                    "label": k.get("label", "gemini-key"),
                    "provider": "gemini",
                    "protocol": "gemini",
                    "base_url": "",
                    "created_at": k.get("created_at", _now_iso()),
                })
                have.add(k["key"])
    # --- legacy migration: models[].gemini_model -> provider/upstream_model ---
    for m in data.get("models", []):
        if not isinstance(m, dict):
            continue
        if "gemini_model" in m and "upstream_model" not in m:
            m["upstream_model"] = m.pop("gemini_model")
        m.setdefault("provider", "gemini")
        if m.get("provider") not in PROVIDERS:
            m["provider"] = "gemini"
        m.setdefault("description", "")
    if not isinstance(data.get("stats"), dict):
        data["stats"] = {"total_requests": 0, "ok": 0, "err": 0,
                         "input_tokens": 0, "output_tokens": 0}
    for sk in ("total_requests", "ok", "err", "input_tokens", "output_tokens"):
        data["stats"].setdefault(sk, 0)
    return data


def save_data(data: dict) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DATA_FILE)


def provider_catalog() -> list[dict]:
    """Provider catalog for the admin UI (pure function)."""
    out = []
    for pid in PROVIDERS:
        d = PROVIDER_DEFAULTS[pid]
        out.append({"id": pid, "protocol": d["protocol"], "base_url": d["base_url"],
                    "auth": d["auth"], "key_hint": d["key_hint"], "key_url": d["key_url"]})
    return out


def normalize_provider_key(provider: str, protocol: str | None, base_url: str | None) -> tuple[str, str]:
    """Validate the (provider, protocol, base_url) triple + fill in defaults.

    Returns: (protocol, base_url). Raises ValueError on error.
    """
    provider = (provider or "gemini").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider} (valid: {', '.join(PROVIDERS)})")
    defaults = PROVIDER_DEFAULTS[provider]
    protocol = (protocol or defaults["protocol"]).strip().lower()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol} (valid: {', '.join(PROTOCOLS)})")
    base_url = (base_url or "").strip().rstrip("/")
    if provider == "custom" and not base_url:
        raise ValueError("base_url is required for custom provider (e.g. http://localhost:11434/v1)")
    return protocol, base_url


def effective_base(entry: dict) -> str:
    """Effective base URL for a provider key (provider default if no custom one)."""
    base = (entry.get("base_url") or "").strip().rstrip("/")
    if base:
        return base
    provider = entry.get("provider") or entry.get("protocol") or "gemini"
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["gemini"])["base_url"]


def effective_protocol(entry: dict) -> str:
    proto = (entry.get("protocol") or "").strip().lower()
    if proto in PROTOCOLS:
        return proto
    provider = (entry.get("provider") or "gemini").strip().lower()
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["gemini"])["protocol"]


def pick_provider_keys(data: dict, provider: str) -> list[dict]:
    """Return the keys in the provider pool in round-robin order ([] if empty)."""
    pool = [k for k in data.get("provider_keys", [])
            if isinstance(k, dict) and k.get("key")
            and (k.get("provider") or "gemini") == provider]
    if not pool:
        return []
    idx = int((data.get("provider_index", {}) or {}).get(provider, 0) or 0) % len(pool)
    return pool[idx:] + pool[:idx]


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return key[:4] + "..." + key[-4:]


def generate_router_key() -> tuple[str, str]:
    """Generate a new user API key. Returns: (full_key, prefix)."""
    rand = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    full = f"sk-samet-{rand[:24]}"
    return full, full[:11] + "..." + full[-4:]


def mask_router_key(entry: dict) -> dict:
    """Safe copy returned to the admin list (raw key is never leaked)."""
    out = {k: v for k, v in entry.items() if k != "key"}
    out["masked"] = entry.get("prefix") or mask_key(entry.get("key", ""))
    return out


def extract_bearer_token(headers) -> str:
    """Extract the token from the Authorization Bearer / x-api-key / api-key headers."""
    auth = headers.get("Authorization", "") or ""
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    for h in ("x-api-key", "api-key"):
        v = headers.get(h, "") or ""
        if v.strip():
            return v.strip()
    return ""


def compute_stats(logs: list[dict], stats: dict | None = None) -> dict:
    """Dashboard summary: success/failure, input/output/total tokens, breakdowns.

    `logs` holds recent requests (detail breakdowns come from here), `stats` holds
    cumulative counters (totals resilient to rotation). Pure function -> easy to test.
    """
    stats = stats or {}
    total = int(stats.get("total_requests", 0) or 0)
    ok = int(stats.get("ok", 0) or 0)
    err = int(stats.get("err", 0) or 0)
    inp = int(stats.get("input_tokens", 0) or 0)
    out = int(stats.get("output_tokens", 0) or 0)
    # in case of unprocessed stats + non-empty logs, also aggregate from logs
    # (only when stats are empty, to avoid double counting)
    if total == 0 and logs:
        for l in logs:
            total += 1
            if l.get("ok"):
                ok += 1
            else:
                err += 1
            try:
                inp += int(l.get("input_tokens", 0) or 0)
                out += int(l.get("output_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
    by_model: dict[str, dict] = {}
    by_endpoint: dict[str, dict] = {}
    by_key: dict[str, dict] = {}
    by_day: dict[str, dict] = {}
    by_provider: dict[str, dict] = {}
    thinking_requests = 0
    latencies: list[int] = []
    for l in logs or []:
        m = str(l.get("model", "?") or "?")
        e = str(l.get("endpoint", "?") or "?")
        k = str(l.get("api_key_label", "") or "anonim")
        p = str(l.get("provider", "") or "?")
        day = str(l.get("ts", "") or "")[:10] or "?"
        if l.get("thinking"):
            thinking_requests += 1
        try:
            li = int(l.get("input_tokens", 0) or 0)
            lo = int(l.get("output_tokens", 0) or 0)
        except (TypeError, ValueError):
            li, lo = 0, 0
        for bucket, name in ((by_model, m), (by_endpoint, e), (by_key, k),
                             (by_day, day), (by_provider, p)):
            b = bucket.setdefault(name, {"requests": 0, "ok": 0, "err": 0,
                                         "input_tokens": 0, "output_tokens": 0})
            b["requests"] += 1
            b["ok" if l.get("ok") else "err"] += 1
            b["input_tokens"] += li
            b["output_tokens"] += lo
        try:
            latencies.append(int(l.get("latency_ms", 0) or 0))
        except (TypeError, ValueError):
            pass
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0
    return {
        "total_requests": total,
        "ok": ok,
        "err": err,
        "success_rate": round(100.0 * ok / total, 1) if total else 0,
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": inp + out,
        "avg_latency_ms": avg_latency,
        "thinking_requests": thinking_requests,
        "by_model": by_model,
        "by_endpoint": by_endpoint,
        "by_key": by_key,
        "by_day": by_day,
        "by_provider": by_provider,
    }


def _safe_str(v, default: str = "") -> str:
    if v is None:
        return default
    if isinstance(v, str):
        return v
    return str(v)


def _merge_system(default_system: str, client_system: str | None) -> str | None:
    """Merge the default system prompt from admin with the client-provided system."""
    default_system = (default_system or "").strip()
    client_system = (client_system or "").strip()
    if default_system and client_system:
        return default_system + "\n\n" + client_system
    return default_system or client_system or None


def _extract_text_from_anthropic_content(content) -> str:
    """content inside tool_result (str | list[block]) -> plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") in ("text", "input_text") and b.get("text"):
                    parts.append(str(b["text"]))
                elif isinstance(b.get("text"), str):
                    parts.append(str(b["text"]))
                elif b.get("type") == "image":
                    parts.append("[image]")
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text", content))
    return str(content)


# ---------------------------------------------------------------------------
# Conversion functions (pure functions -> easy to test)
# ---------------------------------------------------------------------------

def resolve_gemini_model(data: dict, router_model: str) -> str:
    """Legacy signature kept: router alias -> upstream model name."""
    return resolve_target(data, router_model)["upstream_model"]


def _openai_part_text(part: dict) -> str:
    t = part.get("type", "text")
    if t in ("text", "input_text"):
        return str(part.get("text", ""))
    return ""


def _parse_image_url_to_gemini(url: str):
    """data: URL -> inlineData, http(s) -> None (placeholder left)."""
    if not url:
        return None
    if url.startswith("data:"):
        try:
            header, _, b64 = url.partition(",")
            mime = "image/png"
            if ";" in header:
                mime = header[5:].split(";")[0] or mime
            # rough check for valid base64
            base64.b64decode(b64[:64] + "==")
            return {"inlineData": {"mimeType": mime, "data": b64}}
        except Exception:
            return None
    return None


def openai_messages_to_gemini(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """OpenAI messages -> (system_instruction, gemini contents).

    Supported: system/developer roles, text + image_url contents,
    assistant tool_calls, tool role (converted to functionResponse).
    Unknown fields are ignored (no error raised).
    """
    system_text = None
    contents: list[dict] = []
    # tool_call_id -> function name mapping (to resolve tool messages)
    tool_id_to_name: dict[str, str] = {}
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls", []) or []:
            if isinstance(tc, dict):
                fn = tc.get("function", {}) or {}
                if tc.get("id") and fn.get("name"):
                    tool_id_to_name[str(tc["id"])] = str(fn["name"])

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role in ("system", "developer"):
            txt = ""
            if isinstance(content, list):
                txt = " ".join(
                    _openai_part_text(p) for p in content if isinstance(p, dict)
                )
            else:
                txt = str(content or "")
            if txt.strip():
                system_text = (system_text + "\n" if system_text else "") + txt
            continue

        if role == "tool":
            # OpenAI tool result -> Gemini functionResponse
            tool_call_id = str(msg.get("tool_call_id", ""))
            name = tool_id_to_name.get(tool_call_id, "tool")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                )
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": {
                    "name": name,
                    "response": {"result": str(content or "")},
                }}],
            })
            continue

        # assistant tool_calls -> functionCall parts
        tool_calls = msg.get("tool_calls") or []
        fn_parts: list[dict] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            name = fn.get("name")
            if not name:
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except (json.JSONDecodeError, ValueError):
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {"_value": args}
            fn_parts.append({"functionCall": {"name": str(name), "args": args}})

        gemini_role = "model" if role == "assistant" else "user"
        parts: list[dict] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    parts.append({"text": str(part)})
                    continue
                ptype = part.get("type", "text")
                if ptype in ("text", "input_text", "output_text"):
                    parts.append({"text": str(part.get("text", ""))})
                elif ptype == "image_url":
                    img = part.get("image_url", {}) or {}
                    url = img.get("url", "") if isinstance(img, dict) else str(img)
                    inline = _parse_image_url_to_gemini(str(url))
                    if inline is not None:
                        parts.append(inline)
                    else:
                        parts.append({"text": "[image]"})
                elif ptype == "input_text":
                    parts.append({"text": str(part.get("text", ""))})
                # audio/file/refusal etc.: reduced to text or skipped
                elif isinstance(part.get("text"), str):
                    parts.append({"text": str(part["text"])})
        elif isinstance(content, str):
            if content != "" or not fn_parts:
                parts.append({"text": content})
        elif content is None:
            pass
        else:
            parts.append({"text": str(content)})

        parts.extend(fn_parts)
        # some clients send empty content + tool_calls; parts must not stay empty
        parts = [p for p in parts if p != {"text": ""}]
        if not parts:
            continue
        contents.append({"role": gemini_role, "parts": parts})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
    return system_text, contents


def _anthropic_tool_map(messages: list[dict]) -> dict[str, str]:
    """tool_use id -> name mapping (to name tool_results)."""
    mapping: dict[str, str] = {}
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    if b.get("id") and b.get("name"):
                        mapping[str(b["id"])] = str(b["name"])
    return mapping


def anthropic_messages_to_gemini(system, messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Anthropic (system + messages) -> (system_instruction, gemini contents).

    Supported blocks: text, image (base64), tool_use, tool_result,
    thinking/redacted_thinking (skipped), document (reduced to text).
    Fields like `cache_control`, `citations`, `metadata` are ignored.
    """
    if isinstance(system, list):
        texts = []
        for b in system:
            if isinstance(b, dict) and b.get("text"):
                texts.append(str(b["text"]))
            elif isinstance(b, str):
                texts.append(b)
        system = "\n\n".join(t for t in texts if t.strip())
    tool_map = _anthropic_tool_map(messages or [])
    gemini_contents: list[dict] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        gemini_role = "model" if role == "assistant" else "user"
        parts: list[dict] = []
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            if block is None:
                continue
            if isinstance(block, str):
                if block != "":
                    parts.append({"text": block})
                continue
            if not isinstance(block, dict):
                parts.append({"text": str(block)})
                continue
            btype = block.get("type", "text")
            if btype in ("text", "input_text", "output_text"):
                t = block.get("text", "")
                if t is None:
                    t = ""
                parts.append({"text": str(t)})
            elif btype == "image":
                src = block.get("source", {}) or {}
                if src.get("type") == "base64" and src.get("data"):
                    parts.append({"inlineData": {
                        "mimeType": str(src.get("media_type", "image/png")),
                        "data": str(src["data"]),
                    }})
                elif src.get("type") == "url" and src.get("url"):
                    parts.append({"text": "[image]"})
                else:
                    parts.append({"text": "[image]"})
            elif btype == "tool_use":
                name = block.get("name", "tool")
                inp = block.get("input", {})
                if not isinstance(inp, dict):
                    inp = {"_value": inp}
                parts.append({"functionCall": {"name": str(name), "args": inp}})
            elif btype == "tool_result":
                tool_use_id = str(block.get("tool_use_id", ""))
                name = tool_map.get(tool_use_id, "tool")
                text = _extract_text_from_anthropic_content(block.get("content"))
                if block.get("is_error"):
                    resp: dict = {"error": text or "tool error"}
                else:
                    resp = {"result": text}
                parts.append({"functionResponse": {"name": name, "response": resp}})
                # A tool result must have the user role in Gemini; if it arrived
                # inside assistant, mark it as separate content (fixed below)
            elif btype in ("thinking", "redacted_thinking"):
                continue  # extended thinking is not sent to Gemini
            elif btype == "document":
                # extract text if possible, otherwise placeholder
                src = block.get("source", {}) or {}
                if isinstance(src.get("text"), str):
                    parts.append({"text": src["text"]})
                elif isinstance(block.get("text"), str):
                    parts.append({"text": str(block["text"])})
                else:
                    parts.append({"text": "[document]"})
            elif "text" in block and isinstance(block["text"], str):
                parts.append({"text": block["text"]})
            # unknown block types are silently skipped
        # Gemini rejects tool_result inside assistant -> convert to user
        has_fn_response = any("functionResponse" in p for p in parts)
        eff_role = "user" if (gemini_role == "model" and has_fn_response
                              and not any("functionCall" in p or "text" in p for p in parts)) else gemini_role
        parts = [p for p in parts if p != {"text": ""}]
        if not parts:
            continue
        gemini_contents.append({"role": eff_role, "parts": parts})
    if not gemini_contents:
        gemini_contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
    return (str(system) if system and str(system).strip() else None), gemini_contents


def anthropic_tools_to_gemini(tools) -> list[dict]:
    """Anthropic tools -> Gemini tools (functionDeclarations)."""
    decls: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not name:
            continue
        schema = t.get("input_schema", {}) or {}
        if not isinstance(schema, dict):
            schema = {"type": "object"}
        schema.setdefault("type", "object")
        decls.append({
            "name": str(name),
            "description": str(t.get("description", "")),
            "parameters": schema,
        })
    if not decls:
        return []
    return [{"functionDeclarations": decls}]


def anthropic_tool_choice_to_gemini(tool_choice) -> dict | None:
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        return None
    t = tool_choice.get("type", "auto")
    if t == "auto":
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if t == "any":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if t == "tool":
        name = tool_choice.get("name")
        cfg: dict = {"mode": "ANY"}
        if name:
            cfg["allowedFunctionNames"] = [str(name)]
        return {"functionCallingConfig": cfg}
    if t == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    return None


def openai_tools_to_gemini(tools) -> list[dict]:
    decls: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t) if t.get("type", "function") == "function" else t
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters", {}) or {}
        if not isinstance(params, dict):
            params = {"type": "object"}
        params.setdefault("type", "object")
        decls.append({
            "name": str(name),
            "description": str(fn.get("description", "")),
            "parameters": params,
        })
    if not decls:
        return []
    return [{"functionDeclarations": decls}]


def openai_tool_choice_to_gemini(tool_choice) -> dict | None:
    if tool_choice is None or tool_choice == "auto":
        return None
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function", {}) or {}
            name = fn.get("name")
            cfg: dict = {"mode": "ANY"}
            if name:
                cfg["allowedFunctionNames"] = [str(name)]
            return {"functionCallingConfig": cfg}
        if tool_choice.get("type") == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
    return None


# ---------------------------------------------------------------------------
# Thinking (reasoning) types
# ---------------------------------------------------------------------------
# Support matrix:
#   Anthropic extended thinking: `thinking` parameter + `thinking` /
#     `redacted_thinking` blocks. Carried one-to-one to an Anthropic provider
#     (signed thinking and redacted as-is; unsigned thinking converted to
#     plain text because upstream requires a valid signature).
#   OpenAI: `reasoning_effort` parameter passthrough; DeepSeek-style
#     `reasoning_content` field converted to an Anthropic `thinking` block.
#   Gemini: `thinkingConfig.thinkingBudget` + `includeThoughts`; parts marked
#     `thought` converted to a thinking block if thinking was requested,
#     otherwise dropped (to avoid chain-of-thought leakage).

EFFORT_BUDGET = {"minimal": 512, "low": 1024, "medium": 4096, "high": 16384,
                 "xhigh": 32768, "max": 32768}
GEMINI_THINKING_BUDGET_CAP = 24576


def budget_for_effort(effort, default: int = 4096) -> int:
    if isinstance(effort, str) and effort.strip().lower() in EFFORT_BUDGET:
        return EFFORT_BUDGET[effort.strip().lower()]
    return default


def effort_for_budget(n) -> str:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "medium"
    if n <= 2048:
        return "low"
    if n <= 8192:
        return "medium"
    return "high"


def anthropic_thinking_of_body(body) -> tuple[bool, int | None]:
    """thinking parameter in the Anthropic body -> (enabled, budget)."""
    t = (body or {}).get("thinking", {}) or {}
    if isinstance(t, dict) and t.get("type") == "enabled":
        try:
            budget = int(t.get("budget_tokens")) if t.get("budget_tokens") else None
        except (TypeError, ValueError):
            budget = None
        return True, budget
    return False, None


def sanitize_anthropic_messages_for_upstream(messages: list[dict]) -> list[dict]:
    """Clean up history to be sent to the Anthropic upstream.

    `redacted_thinking` and signed `thinking` pass through as-is (upstream
    validates them). Unsigned `thinking` (like the ones we generate) would
    return 400 upstream -> converted to plain text, preserving context.
    """
    out: list[dict] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if not isinstance(content, list):
            out.append(msg)
            continue
        blocks: list[dict] = []
        for b in content:
            if not isinstance(b, dict):
                blocks.append(b)
                continue
            if b.get("type") == "thinking" and not b.get("signature"):
                txt = str(b.get("thinking", "") or "")
                if txt.strip():
                    blocks.append({"type": "text", "text": txt})
                continue
            blocks.append(b)
        if not blocks:
            continue
        out.append({**msg, "content": blocks})
    return out


def body_has_thinking(protocol: str, body) -> bool:
    """For logs: does the request contain thinking? (pure function)."""
    body = body or {}
    if protocol == "anthropic":
        enabled, _ = anthropic_thinking_of_body(body)
        if enabled:
            return True
        for msg in body.get("messages", []) or []:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
                    for b in content):
                return True
        return False
    if protocol == "openai":
        if body.get("reasoning_effort"):
            return True
        for msg in body.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("reasoning_content"):
                return True
            content = msg.get("content", "")
            if isinstance(content, list) and any(
                    isinstance(p, dict) and p.get("type") in ("reasoning", "reasoning_content")
                    for p in content):
                return True
        return False
    gc = (body.get("generationConfig", {}) or {})
    if gc.get("thinkingConfig"):
        return True
    for content in body.get("contents", []) or []:
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if any(isinstance(p, dict) and p.get("thought") is True for p in parts):
            return True
    return False


def _anthropic_system_text(system) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        texts = []
        for b in system:
            if isinstance(b, dict) and b.get("text"):
                texts.append(str(b["text"]))
            elif isinstance(b, str):
                texts.append(b)
        return "\n\n".join(t for t in texts if t.strip())
    return str(system)


def _merge_anthropic_system(default_system: str, client_system) -> str | None:
    parts = [s for s in [(default_system or "").strip(),
                         _anthropic_system_text(client_system).strip()] if s]
    merged = "\n\n".join(parts)
    return merged or None


def _openai_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text", "")) for p in content
                        if isinstance(p, dict) and isinstance(p.get("text"), str))
    return str(content)


def _data_url_to_anthropic_image(url: str):
    if not url:
        return None
    if url.startswith("data:"):
        _header, _, b64 = url.partition(",")
        if not b64:
            return None
        mime = "image/png"
        if ";" in _header:
            mime = _header[5:].split(";")[0] or mime
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
    if url.startswith("http://") or url.startswith("https://"):
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


# ---------------------------------------------------------------------------
# Cross-format converters (pure functions -> easy to test)
# ---------------------------------------------------------------------------

def openai_to_anthropic_body(oai_body, default_system: str = "") -> tuple[dict, bool]:
    """OpenAI chat body -> (Anthropic messages body, thinking_enabled).

    tool_calls -> tool_use (id preserved), tool message -> tool_result,
    reasoning_effort -> thinking parameter, stop -> stop_sequences.
    reasoning_content input cannot be carried over -> dropped.
    """
    oai_body = oai_body or {}
    system_parts = []
    if (default_system or "").strip():
        system_parts.append(default_system.strip())
    messages_in = oai_body.get("messages", []) or []
    anth_messages: list[dict] = []
    for msg in messages_in:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("system", "developer"):
            t = _openai_content_to_text(content)
            if t.strip():
                system_parts.append(t)
            continue
        if role == "tool":
            anth_messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": str(msg.get("tool_call_id", "")),
                "content": _openai_content_to_text(content),
            }]})
            continue
        blocks: list[dict] = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    blocks.append({"type": "text", "text": str(part)})
                    continue
                ptype = part.get("type", "text")
                if ptype in ("text", "input_text", "output_text"):
                    blocks.append({"type": "text", "text": str(part.get("text", ""))})
                elif ptype == "image_url":
                    img = part.get("image_url", {}) or {}
                    url = img.get("url", "") if isinstance(img, dict) else str(img)
                    blk = _data_url_to_anthropic_image(str(url))
                    blocks.append(blk if blk is not None else {"type": "text", "text": "[image]"})
                elif ptype == "refusal":
                    blocks.append({"type": "text", "text": str(part.get("refusal", ""))})
                # reasoning_content / reasoning input: specific to OpenAI output,
                # no equivalent in Anthropic input -> dropped
        elif isinstance(content, str):
            if content != "":
                blocks.append({"type": "text", "text": content})
        elif content is None:
            pass
        else:
            blocks.append({"type": "text", "text": str(content)})
        for tc in msg.get("tool_calls", []) or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {}) or {}
            if not fn.get("name"):
                continue
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except (ValueError, json.JSONDecodeError):
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {"_value": args}
            blocks.append({"type": "tool_use",
                           "id": str(tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                           "name": str(fn["name"]), "input": args})
        blocks = [b for b in blocks if not (b.get("type") == "text" and b.get("text", "") == "")]
        if not blocks:
            continue
        anth_messages.append({"role": "assistant" if role == "assistant" else "user",
                              "content": blocks})
    if not anth_messages:
        anth_messages = [{"role": "user", "content": "Hello"}]
    try:
        max_tokens = int(oai_body.get("max_tokens", oai_body.get("max_completion_tokens")) or 1024)
    except (TypeError, ValueError):
        max_tokens = 1024
    abody: dict = {"max_tokens": max_tokens, "messages": anth_messages}
    if system_parts:
        abody["system"] = "\n\n".join(system_parts)
    tools = []
    for t in oai_body.get("tools", []) or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t) if t.get("type", "function") == "function" else t
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        schema = fn.get("parameters", {}) or {}
        if not isinstance(schema, dict):
            schema = {"type": "object"}
        schema.setdefault("type", "object")
        tools.append({"name": str(fn["name"]), "description": str(fn.get("description", "")),
                      "input_schema": schema})
    if tools:
        abody["tools"] = tools
    tc = oai_body.get("tool_choice")
    if tc == "required":
        abody["tool_choice"] = {"type": "any"}
    elif tc == "none":
        abody["tool_choice"] = {"type": "none"}
    elif isinstance(tc, dict) and tc.get("type") == "function":
        name = (tc.get("function", {}) or {}).get("name")
        abody["tool_choice"] = {"type": "tool", "name": str(name)} if name else {"type": "auto"}
    for k in ("temperature", "top_p", "top_k"):
        if oai_body.get(k) is not None:
            abody[k] = oai_body[k]
    stop = oai_body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if isinstance(stop, list) and stop:
        abody["stop_sequences"] = [str(s) for s in stop if str(s)][:4]
    if isinstance(oai_body.get("user"), str) and oai_body["user"].strip():
        abody["metadata"] = {"user_id": oai_body["user"].strip()[:128]}
    thinking_enabled = False
    if isinstance(oai_body.get("reasoning_effort"), str):
        abody["thinking"] = {"type": "enabled",
                             "budget_tokens": budget_for_effort(oai_body["reasoning_effort"])}
        thinking_enabled = True
    return abody, thinking_enabled


def anthropic_to_openai_body(anth_body, default_system: str = "") -> tuple[dict, str | None]:
    """Anthropic messages body -> (OpenAI chat body, reasoning_effort|None).

    tool_use -> tool_calls (id preserved), tool_result -> tool message
    (empty tool_use_id falls back to user text, empty result gets a placeholder),
    thinking block cannot be carried to upstream OpenAI -> dropped (documented),
    thinking parameter -> reasoning_effort heuristic.
    When thinking is on, temperature/top_p are dropped: OpenAI reasoning
    models (o1/o3/gpt-5 etc.) and strict gateways reject non-default values
    with 400 invalid_request_error.
    """
    anth_body = anth_body or {}
    system_text = _merge_anthropic_system(default_system, anth_body.get("system"))
    tool_map: dict[str, str] = {}
    for msg in anth_body.get("messages", []) or []:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id"):
                    tool_map[str(b["id"])] = str(b.get("name", "tool"))
    messages: list[dict] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for msg in anth_body.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        blocks = content if isinstance(content, list) else [content]
        texts: list[str] = []
        images: list[dict] = []
        tcalls: list[dict] = []
        tresults: list[dict] = []
        for b in blocks:
            if b is None:
                continue
            if isinstance(b, str):
                if b != "":
                    texts.append(b)
                continue
            if not isinstance(b, dict):
                texts.append(str(b))
                continue
            btype = b.get("type", "text")
            if btype in ("text", "input_text", "output_text"):
                if str(b.get("text", "")) != "":
                    texts.append(str(b["text"]))
            elif btype == "image":
                src = b.get("source", {}) or {}
                if src.get("type") == "base64" and src.get("data"):
                    mime = str(src.get("media_type", "image/png"))
                    images.append({"type": "image_url",
                                   "image_url": {"url": f"data:{mime};base64,{src['data']}"}})
                elif src.get("type") == "url" and src.get("url"):
                    images.append({"type": "image_url", "image_url": {"url": str(src["url"])}})
                else:
                    texts.append("[image]")
            elif btype == "tool_use":
                args = b.get("input", {})
                if not isinstance(args, dict):
                    args = {"_value": args}
                tcalls.append({"id": str(b.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                               "type": "function",
                               "function": {"name": str(b.get("name", "tool")),
                                            "arguments": json.dumps(args, ensure_ascii=False)}})
            elif btype == "tool_result":
                tid = str(b.get("tool_use_id", "") or "").strip()
                txt = _extract_text_from_anthropic_content(b.get("content"))
                if b.get("is_error") and txt:
                    txt = f"Error: {txt}"
                if not txt:
                    txt = "(empty tool result)"
                if not tid:
                    # OpenAI rejects tool messages with empty tool_call_id (400).
                    # Fall back to plain user text so the request stays valid.
                    texts.append(f"[tool_result]: {txt}")
                    continue
                tresults.append({"role": "tool", "tool_call_id": tid, "content": txt})
            elif btype in ("thinking", "redacted_thinking"):
                continue  # no equivalent in OpenAI chat input
            elif btype == "document":
                src = b.get("source", {}) or {}
                if isinstance(src.get("text"), str):
                    texts.append(src["text"])
                else:
                    texts.append("[document]")
            elif isinstance(b.get("text"), str):
                texts.append(str(b["text"]))
        if tcalls or texts or images:
            if images:
                parts: list[dict] = []
                if texts:
                    parts.append({"type": "text", "text": "\n".join(texts)})
                parts.extend(images)
                m: dict = {"role": "assistant" if role == "assistant" else "user",
                           "content": parts}
            else:
                m = {"role": "assistant" if role == "assistant" else "user",
                     "content": "\n".join(texts)}
            if tcalls and role == "assistant":
                m["tool_calls"] = tcalls
                if not texts and not images:
                    # OpenAI-compatible gateways (vLLM, Ollama, opencode zen)
                    # reject content:"" with tool_calls; null is the valid form.
                    m["content"] = None
            elif tcalls:
                # tool_use must not appear inside user; reduce to plain text (name preserved)
                for tc in tcalls:
                    texts.append(f"[tool:{tc['function']['name']}]")
                m["content"] = "\n".join(texts) if not images else m.get("content")
            messages.append(m)
        messages.extend(tresults)
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    obody: dict = {"messages": messages}
    try:
        obody["max_tokens"] = int(anth_body.get("max_tokens") or 1024)
    except (TypeError, ValueError):
        obody["max_tokens"] = 1024
    # Clamp absurd max_tokens (Claude Code can send very large values with
    # thinking budgets); strict upstreams return 400 when it exceeds the
    # model's max output window.
    try:
        if int(obody["max_tokens"]) > 16384:
            obody["max_tokens"] = 16384
    except (TypeError, ValueError):
        pass
    reasoning_effort = None
    thinking_on, _budget = anthropic_thinking_of_body(anth_body)
    if not thinking_on:
        for k in ("temperature", "top_p"):
            if anth_body.get(k) is not None:
                obody[k] = anth_body[k]
    # NOTE: when thinking is on, temperature/top_p are intentionally dropped
    # (see docstring).
    stop = anth_body.get("stop_sequences")
    if isinstance(stop, str):
        stop = [stop]
    if isinstance(stop, list) and stop:
        obody["stop"] = [str(s) for s in stop if str(s)][:4]
    tools = []
    for t in anth_body.get("tools", []) or []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        schema = t.get("input_schema", {}) or {}
        if not isinstance(schema, dict):
            schema = {"type": "object"}
        tools.append({"type": "function",
                      "function": {"name": str(t["name"]),
                                   "description": str(t.get("description", "")),
                                   "parameters": schema}})
    if tools:
        obody["tools"] = tools
    choice = anth_body.get("tool_choice")
    if isinstance(choice, dict):
        ctype = choice.get("type", "auto")
        if ctype == "any":
            obody["tool_choice"] = "required"
        elif ctype == "none":
            obody["tool_choice"] = "none"
        elif ctype == "tool" and choice.get("name"):
            obody["tool_choice"] = {"type": "function",
                                    "function": {"name": str(choice["name"])}}
    if thinking_on:
        _b = _budget if _budget else 4096
        reasoning_effort = effort_for_budget(_b)
        obody["reasoning_effort"] = reasoning_effort
    return obody, reasoning_effort


def sanitize_openai_body_for_retry(obody: dict) -> tuple[dict, list[str]]:
    """Strip params that strict OpenAI-compatible gateways reject (pure).

    Returns (sanitized_body, removed_keys). Removes reasoning_effort,
    temperature/top_p (rejected together with reasoning), converts
    max_tokens -> max_completion_tokens (o1/o3/zen style gateways), drops
    empty tool_choice/stop. Used for a one-time retry after a 400.
    """
    body = json.loads(json.dumps(obody or {})) if isinstance(obody, dict) else {}
    removed: list[str] = []
    for k in ("reasoning_effort", "reasoning", "temperature", "top_p"):
        if k in body:
            body.pop(k, None)
            removed.append(k)
    if "max_tokens" in body and "max_completion_tokens" not in body:
        try:
            body["max_completion_tokens"] = int(body.pop("max_tokens"))
            removed.append("max_tokens->max_completion_tokens")
        except (TypeError, ValueError):
            body.pop("max_tokens", None)
            removed.append("max_tokens")
    # fix assistant content "" + tool_calls (some gateways require null)
    try:
        for m in body.get("messages", []) or []:
            if isinstance(m, dict) and m.get("tool_calls") and m.get("content") == "":
                m["content"] = None
                if "content_fixed" not in removed:
                    removed.append("content_fixed")
            if isinstance(m, dict) and m.get("role") == "tool":
                if not str(m.get("tool_call_id", "") or "").strip():
                    m["role"] = "user"
                    m.pop("tool_call_id", None)
                    m["content"] = f"[tool_result]: {m.get('content') or '(empty)'}"
                    if "tool_call_id_fixed" not in removed:
                        removed.append("tool_call_id_fixed")
                elif not str(m.get("content", "") or "").strip():
                    m["content"] = "(empty tool result)"
    except (AttributeError, TypeError):
        pass
    return body, removed


def describe_openai_body(obody: dict) -> str:
    """One-line payload summary for error logs (no message content)."""
    try:
        msgs = obody.get("messages", []) if isinstance(obody, dict) else []
        n = len(msgs) if isinstance(msgs, list) else 0
        roles = ""
        try:
            roles = ",".join(str(m.get("role", "?"))[:5] for m in msgs[:8]) if isinstance(msgs, list) else ""
        except (AttributeError, TypeError):
            pass
        keys = sorted(k for k in (obody or {}) if k != "messages")
        return f"msgs={n} [{roles}] keys={keys} max={obody.get('max_tokens', obody.get('max_completion_tokens', '?'))}"
    except Exception:
        return "?"


def openai_response_to_anthropic(oai: dict, router_model: str) -> dict:
    """OpenAI chat response -> Anthropic message.

    reasoning_content -> thinking block (unsigned), tool_calls -> tool_use.
    """
    oai = oai or {}
    choice = (oai.get("choices") or [{}])[0] or {}
    msg = choice.get("message", {}) or {}
    finish = choice.get("finish_reason", "stop")
    content: list[dict] = []
    rc = msg.get("reasoning_content")
    if isinstance(rc, list):
        rc = "\n".join(str(x) for x in rc if str(x))
    if isinstance(rc, str) and rc.strip():
        content.append({"type": "thinking", "thinking": rc})
    text = msg.get("content") or ""
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls", []) or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (ValueError, json.JSONDecodeError):
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = {"_value": args}
        content.append({"type": "tool_use",
                        "id": str(tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": str(fn.get("name", "tool")), "input": args})
    if not content:
        content = [{"type": "text", "text": ""}]
    if any(b.get("type") == "tool_use" for b in content):
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    usage = oai.get("usage", {}) or {}
    try:
        inp = int(usage.get("prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        inp = 0
    try:
        out = int(usage.get("completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        out = 0
    return {"id": f"msg_{uuid.uuid4().hex[:12]}", "type": "message", "role": "assistant",
            "model": router_model, "content": content, "stop_reason": stop_reason,
            "usage": {"input_tokens": inp, "output_tokens": out}}


def anthropic_response_to_openai(anth: dict, router_model: str) -> dict:
    """Anthropic message -> OpenAI chat response.

    thinking -> reasoning_content (DeepSeek convention), redacted dropped,
    tool_use -> tool_calls (id preserved).
    """
    anth = anth or {}
    blocks = anth.get("content", []) or []
    texts: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            texts.append(b["text"])
        elif b.get("type") == "thinking" and isinstance(b.get("thinking"), str):
            reasoning.append(b["thinking"])
        elif b.get("type") == "tool_use":
            args = b.get("input", {})
            if not isinstance(args, dict):
                args = {"_value": args}
            tool_calls.append({"id": str(b.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                               "type": "function",
                               "function": {"name": str(b.get("name", "tool")),
                                            "arguments": json.dumps(args, ensure_ascii=False)}})
    text = "".join(texts)
    msg: dict = {"role": "assistant",
                 "content": text if not tool_calls else (text or None)}
    if reasoning:
        msg["reasoning_content"] = "\n".join(reasoning)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    stop = (anth.get("stop_reason") or "end_turn")
    finish = {"tool_use": "tool_calls", "max_tokens": "length"}.get(stop, "stop")
    usage = anth.get("usage", {}) or {}
    try:
        inp = int(usage.get("input_tokens", 0) or 0)
    except (TypeError, ValueError):
        inp = 0
    try:
        out = int(usage.get("output_tokens", 0) or 0)
    except (TypeError, ValueError):
        out = 0
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
            "created": int(time.time()), "model": router_model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": inp, "completion_tokens": out,
                      "total_tokens": inp + out}}


def gemini_request_to_openai(gbody) -> dict:
    """Gemini generateContent body -> OpenAI chat body (excluding model)."""
    gbody = gbody or {}
    si = gbody.get("systemInstruction", {}) or {}
    systext = "\n".join(str(p.get("text", "")) for p in si.get("parts", [])
                        if isinstance(p, dict) and str(p.get("text", "")).strip())
    messages: list[dict] = []
    if systext.strip():
        messages.append({"role": "system", "content": systext})
    pending: list[list] = []  # [name, call_id]
    for content in gbody.get("contents", []) or []:
        if not isinstance(content, dict):
            continue
        role = "assistant" if content.get("role") == "model" else "user"
        texts: list[str] = []
        images: list[dict] = []
        tcalls: list[dict] = []
        tresps: list[dict] = []
        for part in content.get("parts", []) or []:
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("text"), str) and part["text"] != "":
                texts.append(part["text"])
            elif isinstance(part.get("inlineData"), dict):
                mime = str(part["inlineData"].get("mimeType", "image/png"))
                d = str(part["inlineData"].get("data", ""))
                images.append({"type": "image_url",
                               "image_url": {"url": f"data:{mime};base64,{d}"}})
            elif isinstance(part.get("functionCall"), dict):
                fc = part["functionCall"]
                name = str(fc.get("name", "tool"))
                args = fc.get("args", {})
                cid = f"call_{uuid.uuid4().hex[:12]}"
                pending.append([name, cid])
                tcalls.append({"id": cid, "type": "function",
                               "function": {"name": name,
                                            "arguments": json.dumps(args, ensure_ascii=False)
                                            if isinstance(args, dict) else str(args)}})
            elif isinstance(part.get("functionResponse"), dict):
                fr = part["functionResponse"]
                name = str(fr.get("name", "tool"))
                cid = None
                for p in pending:
                    if p[0] == name:
                        cid = p[1]
                        pending.remove(p)
                        break
                if cid is None:
                    cid = f"call_{uuid.uuid4().hex[:12]}"
                resp = fr.get("response", "")
                tresps.append({"role": "tool", "tool_call_id": cid,
                               "content": json.dumps(resp, ensure_ascii=False)
                               if isinstance(resp, dict) else str(resp)})
        if tcalls or texts or images:
            if images:
                parts: list[dict] = []
                if texts:
                    parts.append({"type": "text", "text": "\n".join(texts)})
                parts.extend(images)
                messages.append({"role": role, "content": parts})
            else:
                m: dict = {"role": role, "content": "\n".join(texts)}
                if tcalls:
                    m["tool_calls"] = tcalls
                messages.append(m)
        messages.extend(tresps)
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    obody: dict = {"messages": messages}
    tools = []
    for t in gbody.get("tools", []) or []:
        if not isinstance(t, dict):
            continue
        for decl in t.get("functionDeclarations", []) or []:
            if isinstance(decl, dict) and decl.get("name"):
                params = decl.get("parameters", {}) or {}
                tools.append({"type": "function",
                              "function": {"name": str(decl["name"]),
                                           "description": str(decl.get("description", "")),
                                           "parameters": params if isinstance(params, dict)
                                           else {"type": "object"}}})
    if tools:
        obody["tools"] = tools
    tcfg = ((gbody.get("toolConfig", {}) or {}).get("functionCallingConfig", {})) or {}
    if tcfg.get("mode") == "NONE":
        obody["tool_choice"] = "none"
    elif tcfg.get("mode") == "ANY":
        allowed = tcfg.get("allowedFunctionNames", []) or []
        obody["tool_choice"] = {"type": "function",
                                "function": {"name": str(allowed[0])}} if allowed else "required"
    gc = gbody.get("generationConfig", {}) or {}
    if gc.get("maxOutputTokens"):
        obody["max_tokens"] = gc["maxOutputTokens"]
    if gc.get("temperature") is not None:
        obody["temperature"] = gc["temperature"]
    if gc.get("topP") is not None:
        obody["top_p"] = gc["topP"]
    if gc.get("stopSequences"):
        obody["stop"] = gc["stopSequences"]
    return obody


def openai_response_to_gemini(oai: dict) -> dict:
    """OpenAI chat response -> Gemini generateContent response."""
    oai = oai or {}
    choice = (oai.get("choices") or [{}])[0] or {}
    msg = choice.get("message", {}) or {}
    finish = choice.get("finish_reason", "stop")
    parts: list[dict] = []
    if isinstance(msg.get("content"), str) and msg["content"]:
        parts.append({"text": msg["content"]})
    for tc in msg.get("tool_calls", []) or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {}) or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (ValueError, json.JSONDecodeError):
                args = {"_raw": args}
        parts.append({"functionCall": {"name": str(fn.get("name", "tool")),
                                       "args": args if isinstance(args, dict)
                                       else {"_value": args}}})
    if not parts:
        parts = [{"text": ""}]
    usage = oai.get("usage", {}) or {}
    try:
        inp = int(usage.get("prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        inp = 0
    try:
        out = int(usage.get("completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        out = 0
    return {"candidates": [{"content": {"role": "model", "parts": parts},
                            "finishReason": "MAX_TOKENS" if finish == "length" else "STOP"}],
            "usageMetadata": {"promptTokenCount": inp, "candidatesTokenCount": out,
                              "totalTokenCount": inp + out}}


def anthropic_response_to_gemini(anth: dict) -> dict:
    """Anthropic message -> Gemini generateContent response.

    thinking -> converted to a thought-flagged part, redacted dropped
    (Anthropic-specific encrypted data, no equivalent in other formats).
    """
    anth = anth or {}
    parts: list[dict] = []
    for b in anth.get("content", []) or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str) and b["text"]:
            parts.append({"text": b["text"]})
        elif b.get("type") == "thinking" and isinstance(b.get("thinking"), str) and b["thinking"]:
            parts.append({"text": b["thinking"], "thought": True})
        elif b.get("type") == "tool_use":
            args = b.get("input", {})
            if not isinstance(args, dict):
                args = {"_value": args}
            parts.append({"functionCall": {"name": str(b.get("name", "tool")), "args": args}})
    if not parts:
        parts = [{"text": ""}]
    stop = anth.get("stop_reason") or "end_turn"
    usage = anth.get("usage", {}) or {}
    try:
        inp = int(usage.get("input_tokens", 0) or 0)
    except (TypeError, ValueError):
        inp = 0
    try:
        out = int(usage.get("output_tokens", 0) or 0)
    except (TypeError, ValueError):
        out = 0
    return {"candidates": [{"content": {"role": "model", "parts": parts},
                            "finishReason": "MAX_TOKENS" if stop == "max_tokens" else "STOP"}],
            "usageMetadata": {"promptTokenCount": inp, "candidatesTokenCount": out,
                              "totalTokenCount": inp + out}}


def resolve_target(data: dict, router_model: str) -> dict:
    """Resolve a router alias to its target -> {name, provider, upstream_model}.

    An unknown name starting with gemini-* passes through as-is (passthrough); in
    other cases it falls back to the first model (legacy claude-* fallback behavior).
    """
    router_model = (router_model or "").strip() or "samet-flash"
    models = [m for m in data.get("models", []) if isinstance(m, dict)]
    for m in models:
        if m.get("name") == router_model:
            provider = m.get("provider") or "gemini"
            if provider not in ("openai", "anthropic", "gemini"):
                provider = "gemini"
            upstream = m.get("upstream_model") or m.get("gemini_model") or router_model
            return {"name": m["name"], "provider": provider,
                    "upstream_model": str(upstream)}
    low = router_model.lower()
    if low.startswith("gemini-") or low.startswith("gemma-") or low.startswith("models/"):
        return {"name": router_model, "provider": "gemini",
                "upstream_model": router_model.replace("models/", "")}
    if models:
        m0 = models[0]
        provider = m0.get("provider") or "gemini"
        if provider not in ("openai", "anthropic", "gemini"):
            provider = "gemini"
        return {"name": router_model, "provider": provider,
                "upstream_model": str(m0.get("upstream_model") or m0.get("gemini_model")
                                      or "gemini-2.0-flash")}
    return {"name": router_model, "provider": "gemini",
            "upstream_model": "gemini-2.0-flash"}


def build_gemini_payload(
    system_text: str | None,
    contents: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    tools: list[dict] | None = None,
    tool_config: dict | None = None,
    thinking_budget: int | None = None,
    include_thoughts: bool = False,
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
    if top_p is not None:
        try:
            gen_cfg["topP"] = float(top_p)
        except (TypeError, ValueError):
            pass
    if top_k is not None:
        try:
            gen_cfg["topK"] = int(top_k)
        except (TypeError, ValueError):
            pass
    if stop_sequences:
        seqs = [str(s) for s in stop_sequences if str(s)]
        if seqs:
            gen_cfg["stopSequences"] = seqs[:5]
    if thinking_budget:
        try:
            budget = max(0, min(int(thinking_budget), GEMINI_THINKING_BUDGET_CAP))
        except (TypeError, ValueError):
            budget = 0
        if budget > 0:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": budget,
                                         "includeThoughts": bool(include_thoughts)}
    elif include_thoughts:
        gen_cfg["thinkingConfig"] = {"includeThoughts": True}
    if gen_cfg:
        payload["generationConfig"] = gen_cfg
    if tools:
        payload["tools"] = tools
    if tool_config:
        payload["toolConfig"] = tool_config
    return payload


def gemini_text_of_response(gemini_json: dict) -> str:
    try:
        cands = gemini_json.get("candidates", [])
        parts = cands[0].get("content", {}).get("parts", [])
        # parts flagged as thought are chain-of-thought, excluded from visible text
        return "".join(p.get("text", "") for p in parts
                       if isinstance(p, dict) and isinstance(p.get("text"), str)
                       and p.get("thought") is not True)
    except (IndexError, AttributeError, TypeError):
        return ""


def _gemini_usage_of_response(gemini_json: dict) -> tuple[int, int]:
    meta = gemini_json.get("usageMetadata", {}) or {}
    try:
        inp = int(meta.get("promptTokenCount", 0) or 0)
    except (TypeError, ValueError):
        inp = 0
    try:
        out = int(meta.get("candidatesTokenCount", 0) or 0)
    except (TypeError, ValueError):
        out = 0
    return inp, out


def _gemini_finish_of_response(gemini_json: dict) -> str:
    try:
        return str(gemini_json.get("candidates", [])[0].get("finishReason", "STOP"))
    except (IndexError, AttributeError, TypeError):
        return "STOP"


def gemini_to_openai_response(gemini_json: dict, router_model: str) -> dict:
    try:
        parts = gemini_json.get("candidates", [])[0].get("content", {}).get("parts", [])
    except (IndexError, AttributeError, TypeError):
        parts = []
    texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[dict] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("text"), str) and p["text"]:
            (thoughts if p.get("thought") is True else texts).append(p["text"])
        if isinstance(p.get("functionCall"), dict):
            fc = p["functionCall"]
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": str(fc.get("name", "tool")),
                    "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                },
            })
    finish = _gemini_finish_of_response(gemini_json)
    if tool_calls:
        finish_reason = "tool_calls"
    elif finish == "MAX_TOKENS":
        finish_reason = "length"
    else:
        finish_reason = "stop"
    inp, out = _gemini_usage_of_response(gemini_json)
    msg: dict = {"role": "assistant", "content": "".join(texts) if not tool_calls else ("".join(texts) or None)}
    if thoughts:
        msg["reasoning_content"] = "\n".join(thoughts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": router_model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out},
    }


def gemini_to_anthropic_response(gemini_json: dict, router_model: str,
                                 thinking_enabled: bool = False) -> dict:
    try:
        parts = gemini_json.get("candidates", [])[0].get("content", {}).get("parts", [])
    except (IndexError, AttributeError, TypeError):
        parts = []
    content: list[dict] = []
    thought_texts: list[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("text"), str) and p["text"]:
            if p.get("thought") is True:
                thought_texts.append(p["text"])
                continue
            content.append({"type": "text", "text": p["text"]})
        if isinstance(p.get("functionCall"), dict):
            fc = p["functionCall"]
            args = fc.get("args", {})
            if not isinstance(args, dict):
                args = {"_value": args}
            content.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:12]}",
                "name": str(fc.get("name", "tool")),
                "input": args,
            })
    if thought_texts and thinking_enabled:
        # unsigned thinking block: if the client sends it back to us,
        # sanitize converts it to plain text (so upstream does not return 400)
        content.insert(0, {"type": "thinking", "thinking": "\n".join(thought_texts)})
    if not content:
        # empty response: Anthropic does not accept empty content
        content = [{"type": "text", "text": gemini_text_of_response(gemini_json)}]
    finish = _gemini_finish_of_response(gemini_json)
    if any(b.get("type") == "tool_use" for b in content):
        stop_reason = "tool_use"
    elif finish == "MAX_TOKENS":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    inp, out = _gemini_usage_of_response(gemini_json)
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": router_model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": inp, "output_tokens": out},
    }


def estimate_input_tokens(system, messages) -> int:
    """Rough estimate for count_tokens (~4 chars = 1 token)."""
    buf = _safe_str(system if isinstance(system, str) else json.dumps(system or "", ensure_ascii=False))
    try:
        buf += json.dumps(messages or [], ensure_ascii=False)
    except (TypeError, ValueError):
        buf += str(messages)
    return max(1, len(buf) // 4)


def iter_anthropic_sse(anth_resp: dict):
    """Split a completed Anthropic response into SSE events (fake-streaming)."""
    msg_id = anth_resp.get("id", f"msg_{uuid.uuid4().hex[:12]}")
    model = anth_resp.get("model", "")
    usage = anth_resp.get("usage", {}) or {}
    blocks = anth_resp.get("content", []) or []
    stop_reason = anth_resp.get("stop_reason", "end_turn")

    start = {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant",
        "model": model, "content": [], "stop_reason": None,
        "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
    }}
    yield f"event: message_start\ndata: {json.dumps(start, ensure_ascii=False)}\n\n"
    yield "event: ping\ndata: {\"type\": \"ping\"}\n\n"

    for idx, block in enumerate(blocks):
        if block.get("type") == "tool_use":
            cs = {"type": "content_block_start", "index": idx, "content_block": {
                "type": "tool_use", "id": block.get("id"),
                "name": block.get("name"), "input": {}}}
            yield f"event: content_block_start\ndata: {json.dumps(cs, ensure_ascii=False)}\n\n"
            input_json = json.dumps(block.get("input", {}), ensure_ascii=False)
            for i in range(0, max(len(input_json), 1), 30):
                delta = {"type": "content_block_delta", "index": idx, "delta": {
                    "type": "input_json_delta",
                    "partial_json": input_json[i:i + 30]}}
                yield f"event: content_block_delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx}, ensure_ascii=False)}\n\n"
        elif block.get("type") == "thinking":
            text = block.get("thinking", "") if isinstance(block.get("thinking"), str) else ""
            cs = {"type": "content_block_start", "index": idx, "content_block": {
                "type": "thinking", "thinking": ""}}
            yield f"event: content_block_start\ndata: {json.dumps(cs, ensure_ascii=False)}\n\n"
            for i in range(0, max(len(text), 1), 30):
                delta = {"type": "content_block_delta", "index": idx, "delta": {
                    "type": "thinking_delta", "thinking": text[i:i + 30]}}
                yield f"event: content_block_delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n"
            if block.get("signature"):
                sig = {"type": "content_block_delta", "index": idx, "delta": {
                    "type": "signature_delta", "signature": block["signature"]}}
                yield f"event: content_block_delta\ndata: {json.dumps(sig, ensure_ascii=False)}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx}, ensure_ascii=False)}\n\n"
        elif block.get("type") == "redacted_thinking":
            cs = {"type": "content_block_start", "index": idx, "content_block": {
                "type": "redacted_thinking", "data": block.get("data", "")}}
            yield f"event: content_block_start\ndata: {json.dumps(cs, ensure_ascii=False)}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx}, ensure_ascii=False)}\n\n"
        else:
            text = block.get("text", "") if isinstance(block.get("text"), str) else ""
            cs = {"type": "content_block_start", "index": idx, "content_block": {
                "type": "text", "text": ""}}
            yield f"event: content_block_start\ndata: {json.dumps(cs, ensure_ascii=False)}\n\n"
            if text == "":
                pass
            for i in range(0, len(text), 30):
                delta = {"type": "content_block_delta", "index": idx, "delta": {
                    "type": "text_delta", "text": text[i:i + 30]}}
                yield f"event: content_block_delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n"
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx}, ensure_ascii=False)}\n\n"

    md = {"type": "message_delta",
          "delta": {"stop_reason": stop_reason, "stop_sequence": None},
          "usage": {"output_tokens": usage.get("output_tokens", 0)}}
    yield f"event: message_delta\ndata: {json.dumps(md, ensure_ascii=False)}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'}, ensure_ascii=False)}\n\n"


def iter_openai_sse(oai_resp: dict):
    """Split a completed OpenAI response into chat.completion.chunk pieces."""
    model = oai_resp.get("model", "")
    resp_id = oai_resp.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    created = oai_resp.get("created", int(time.time()))
    choices = (oai_resp.get("choices") or [{}])[0]
    msg = choices.get("message", {}) or {}
    finish = choices.get("finish_reason", "stop")
    text = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    tool_calls = msg.get("tool_calls") or []

    def chunk(delta: dict, fin=None):
        return {"id": resp_id, "object": "chat.completion.chunk", "created": created,
                "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}

    yield f"data: {json.dumps(chunk({'role': 'assistant'}), ensure_ascii=False)}\n\n"
    if isinstance(reasoning, str) and reasoning:
        for i in range(0, len(reasoning), 30):
            yield f"data: {json.dumps(chunk({'reasoning_content': reasoning[i:i + 30]}), ensure_ascii=False)}\n\n"
    for i in range(0, len(text), 30):
        yield f"data: {json.dumps(chunk({'content': text[i:i + 30]}), ensure_ascii=False)}\n\n"
    for tc in tool_calls:
        yield f"data: {json.dumps(chunk({'tool_calls': [tc]}), ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps(chunk({}, finish), ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Upstream call layer (protocol + base_url + key rotation)
# ---------------------------------------------------------------------------

def _pool_for_protocol(data: dict, protocol: str) -> list[dict]:
    """Keys speaking the given protocol (including custom)."""
    return [k for k in data.get("provider_keys", [])
            if isinstance(k, dict) and k.get("key")
            and effective_protocol(k) == protocol]


def _ordered_pool(data: dict, protocol: str) -> list[dict]:
    pool = _pool_for_protocol(data, protocol)
    if not pool:
        raise RuntimeError(
            f"No provider key configured for protocol {protocol}. "
            "Add one in /admin first.")
    start = int((data.get("provider_index", {}) or {}).get(protocol, 0) or 0) % len(pool)
    return pool[start:] + pool[:start]


def _advance_provider_index(protocol: str, nkeys: int) -> None:
    with _lock:
        data = load_data()
        idx = data.setdefault("provider_index", {})
        idx[protocol] = (int(idx.get(protocol, 0) or 0) + 1) % max(nkeys, 1)
        save_data(data)


def openai_chat_url(base: str) -> str:
    base = (base or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def anthropic_messages_url(base: str) -> str:
    base = (base or "").rstrip("/")
    if base.endswith("/messages"):
        return base
    return base + "/messages"


def gemini_generate_url(base: str, gemini_model: str) -> str:
    return (base or "").rstrip("/") + f"/v1beta/models/{gemini_model}:generateContent"


def upstream_headers(protocol: str, api_key: str, extra: dict | None = None) -> dict:
    h: dict = {"Content-Type": "application/json"}
    if protocol == "openai":
        h["Authorization"] = f"Bearer {api_key}"
    elif protocol == "anthropic":
        h["x-api-key"] = api_key
        h["anthropic-version"] = ANTHROPIC_VERSION
    if extra:
        h.update({k: v for k, v in extra.items() if v})
    return h


def _close_quietly(resp) -> None:
    try:
        resp.close()
    except Exception:
        pass


def call_upstream(protocol: str, body: dict, gemini_model: str = "",
                  stream: bool = False, timeout=60,
                  extra_headers: dict | None = None):
    """POST to upstream; returns (entry, response).

    Keys speaking the same protocol are tried round-robin; on 401/403/429
    or a connection error the next key is tried. On success (200) the order
    is advanced. A non-200 response is still returned (the handler decides);
    if no attempt can be made at all, an exception is raised.
    """
    with _lock:
        data = load_data()
    pool = _ordered_pool(data, protocol)
    last_exc: Exception | None = None
    for i, entry in enumerate(pool):
        base = effective_base(entry)
        key = entry.get("key", "")
        try:
            if protocol == "openai":
                resp = requests.post(
                    openai_chat_url(base),
                    headers=upstream_headers(protocol, key, extra_headers),
                    json=body, timeout=timeout, stream=stream)
            elif protocol == "anthropic":
                resp = requests.post(
                    anthropic_messages_url(base),
                    headers=upstream_headers(protocol, key, extra_headers),
                    json=body, timeout=timeout, stream=stream)
            else:
                url = gemini_generate_url(base, gemini_model) + f"?key={key}"
                resp = requests.post(url, json=body, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            continue
        if resp.status_code == 200:
            _advance_provider_index(protocol, len(pool))
            return entry, resp
        if resp.status_code in (401, 403, 429) and i < len(pool) - 1:
            _close_quietly(resp)
            last_exc = RuntimeError(f"upstream {resp.status_code}: {resp.text[:300]}")
            continue
        return entry, resp
    raise last_exc or RuntimeError("upstream call failed.")


def call_gemini(gemini_model: str, payload: dict, timeout: int = 60) -> dict:
    """Legacy signature kept: JSON call with the gemini protocol, returns a dict."""
    _entry, resp = call_upstream("gemini", payload, gemini_model=gemini_model,
                                 stream=False, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _rewrite_sse_line(line: bytes, router_model: str) -> bytes:
    try:
        s = line.decode("utf-8", "replace")
    except Exception:
        return line
    if s.startswith("data:"):
        payload = s[5:].strip()
        if payload and payload != "[DONE]":
            try:
                j = json.loads(payload)
            except (ValueError, TypeError):
                return line
            if isinstance(j, dict):
                if isinstance(j.get("model"), str):
                    j["model"] = router_model
                msg = j.get("message")
                if isinstance(msg, dict) and isinstance(msg.get("model"), str):
                    msg["model"] = router_model
                return ("data: " + json.dumps(j, ensure_ascii=False)).encode("utf-8")
    return line


def rewrite_sse_model(chunks, router_model: str):
    """Replace the model name in the upstream SSE stream with the router name (pure generator)."""
    buf = b""
    for chunk in chunks or []:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", "replace")
        buf += chunk or b""
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield _rewrite_sse_line(line, router_model) + b"\n"
    if buf:
        yield _rewrite_sse_line(buf, router_model)


def proxy_sse_response(upstream_resp, router_model: str) -> Response:
    """Relay the upstream SSE stream to the client as-is (model name fixed)."""
    def gen():
        try:
            yield from rewrite_sse_model(
                upstream_resp.iter_content(chunk_size=4096), router_model)
        finally:
            _close_quietly(upstream_resp)

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def forward_upstream_error(protocol: str, resp) -> tuple:
    """Forward the upstream error to the client in the appropriate format (status preserved)."""
    try:
        body = resp.json()
    except (ValueError, TypeError):
        body = None
    snippet = (resp.text or "")[:500]
    if protocol == "anthropic":
        if isinstance(body, dict) and body.get("type") == "error":
            return jsonify(body), resp.status_code
        return jsonify({"type": "error", "error": {
            "type": "api_error", "message": snippet or f"upstream {resp.status_code}"}}), resp.status_code
    if protocol == "gemini":
        if isinstance(body, dict) and body.get("error") is not None:
            return jsonify(body), resp.status_code
        return jsonify({"error": {"message": snippet or f"upstream {resp.status_code}"}}), resp.status_code
    if isinstance(body, dict) and body.get("error") is not None:
        return jsonify(body), resp.status_code
    return jsonify({"error": snippet or f"upstream {resp.status_code}"}), resp.status_code


def add_log(endpoint: str, model: str, ok: bool, latency_ms: int, error: str = "",
            input_tokens: int = 0, output_tokens: int = 0,
            api_key_id: str = "", api_key_label: str = "",
            provider: str = "", thinking: bool = False) -> None:
    try:
        input_tokens = int(input_tokens or 0)
    except (TypeError, ValueError):
        input_tokens = 0
    try:
        output_tokens = int(output_tokens or 0)
    except (TypeError, ValueError):
        output_tokens = 0
    with _lock:
        data = load_data()
        logs = data.get("logs", [])
        logs.append(
            {"ts": _now_iso(), "endpoint": endpoint, "model": model,
             "ok": ok, "latency_ms": latency_ms, "error": (error or "")[:2000],
             "input_tokens": input_tokens, "output_tokens": output_tokens,
             "total_tokens": input_tokens + output_tokens,
             "api_key_id": api_key_id or "", "api_key_label": api_key_label or "",
             "provider": provider or "", "thinking": bool(thinking)}
        )
        data["logs"] = logs[-1000:]
        st = data.setdefault("stats", {"total_requests": 0, "ok": 0, "err": 0,
                                       "input_tokens": 0, "output_tokens": 0})
        st["total_requests"] = int(st.get("total_requests", 0) or 0) + 1
        st["ok" if ok else "err"] = int(st.get("ok" if ok else "err", 0) or 0) + 1
        st["input_tokens"] = int(st.get("input_tokens", 0) or 0) + input_tokens
        st["output_tokens"] = int(st.get("output_tokens", 0) or 0) + output_tokens
        if api_key_id:
            for rk in data.get("router_keys", []):
                if rk.get("id") == api_key_id:
                    rk["request_count"] = int(rk.get("request_count", 0) or 0) + 1
                    rk["last_used_at"] = _now_iso()
                    break
        save_data(data)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    with _lock:
        load_data()  # create the file if missing + migrate the legacy version

    def need_router_auth() -> tuple[bool, dict | None, dict]:
        """Router auth: env ROUTER_API_KEY or panel-generated router_keys.

        Returns: (ok, error_response, key_info). key_info = {"id","label"}.
        If no key is defined at all (local mode), everyone passes and key_info
        is returned empty.
        """
        token = extract_bearer_token(request.headers)
        data = load_data()
        router_keys = data.get("router_keys", []) or []
        if not ROUTER_API_KEY and not router_keys:
            return True, None, {"id": "", "label": ""}
        if ROUTER_API_KEY and token == ROUTER_API_KEY:
            return True, None, {"id": "env", "label": "env:ROUTER_API_KEY"}
        for rk in router_keys:
            if token and token == rk.get("key"):
                return True, None, {"id": rk.get("id", ""), "label": rk.get("label", "")}
        return False, (jsonify({"error": "unauthorized, Bearer token or x-api-key required"}), 401), {"id": "", "label": ""}

    def need_admin_auth():
        if not ADMIN_TOKEN:
            return None
        if request.headers.get("Authorization", "") == f"Bearer {ADMIN_TOKEN}":
            return None
        return jsonify({"error": "unauthorized"}), 401

    @app.before_request
    def _cors_preflight():
        if request.method == "OPTIONS":
            return Response(status=204)

    @app.after_request
    def _cors_headers(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = (
            "Authorization, Content-Type, x-api-key, api-key, "
            "anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access, "
            "x-requested-with"
        )
        resp.headers["Access-Control-Expose-Headers"] = "request-id"
        return resp

    def _anthropic_error(msg: str, status: int = 400, err_type: str = "invalid_request_error"):
        return jsonify({"type": "error", "error": {"type": err_type, "message": msg}}), status

    def _forward_anthropic_beta() -> dict:
        beta = request.headers.get("anthropic-beta", "") or ""
        return {"anthropic-beta": beta.strip()} if beta.strip() else {}

    def _cross_error_message(protocol: str, resp) -> str:
        """Reduce an upstream error in a cross-protocol call to plain text.

        Keeps type/code/param details (not just message) + HTTP status, so
        generic errors like "invalid request" stay diagnosable in /admin logs.
        """
        status = getattr(resp, "status_code", "?")
        try:
            bj = resp.json()
        except (ValueError, TypeError):
            bj = None
        detail = ""
        if isinstance(bj, dict):
            err = bj.get("error")
            if isinstance(err, dict):
                parts = []
                for k in ("type", "code", "param", "message"):
                    v = err.get(k)
                    if v:
                        parts.append(f"{k}={v}")
                if parts:
                    detail = " ".join(str(p) for p in parts)
            elif isinstance(err, str) and err.strip():
                detail = err.strip()
            if not detail:
                # e.g. Anthropic {"type":"error","error":{...}} or zen wrappers
                for k in ("message", "detail"):
                    v = bj.get(k)
                    if isinstance(v, str) and v.strip():
                        detail = v.strip()
                        break
            if not detail:
                try:
                    detail = json.dumps(bj, ensure_ascii=False)[:1000]
                except (TypeError, ValueError):
                    detail = ""
        if not detail:
            try:
                detail = (resp.text or "").strip()[:1000]
            except Exception:
                detail = ""
        if not detail:
            detail = f"upstream {status}"
        return f"Upstream {status}: {detail}"[:2000]

    @app.get("/")
    def index():
        return redirect("/admin")

    @app.get("/health")
    def health():
        data = load_data()
        by_provider: dict[str, int] = {}
        for k in data.get("provider_keys", []):
            p = effective_protocol(k) if isinstance(k, dict) else "?"
            by_provider[p] = by_provider.get(p, 0) + 1
        return jsonify({
            "ok": True, "service": "samet-router",
            "keys": len(data.get("provider_keys", [])),
            "keys_by_protocol": by_provider,
            "models": len(data.get("models", [])),
            "time": _now_iso(),
        })

    @app.get("/admin")
    def admin():
        return render_template("admin.html")

    def _masked_provider_key(k: dict) -> dict:
        return {"id": k.get("id", ""), "label": k.get("label", ""),
                "provider": k.get("provider", "gemini"),
                "protocol": effective_protocol(k),
                "base_url": k.get("base_url", ""),
                "effective_base": effective_base(k),
                "key": mask_key(k.get("key", "")),
                "created_at": k.get("created_at", "")}

    # --- Admin JSON API ---
    @app.get("/admin/api/state")
    def admin_state():
        if (err := need_admin_auth()) is not None:
            return err
        data = load_data()
        return jsonify({
            "provider_keys": [_masked_provider_key(k) for k in data.get("provider_keys", [])],
            "providers": provider_catalog(),
            "router_keys": [mask_router_key(k) for k in data.get("router_keys", [])],
            "models": data.get("models", []),
            "logs": list(reversed(data.get("logs", [])[-100:])),
            "settings": data.get("settings", {"default_system": ""}),
            "stats": compute_stats(data.get("logs", []), data.get("stats")),
            "router_auth_required": bool(ROUTER_API_KEY or data.get("router_keys")),
        })

    @app.get("/admin/api/providers")
    def admin_providers():
        if (err := need_admin_auth()) is not None:
            return err
        return jsonify(provider_catalog())

    @app.get("/admin/api/stats")
    def admin_stats():
        if (err := need_admin_auth()) is not None:
            return err
        data = load_data()
        return jsonify(compute_stats(data.get("logs", []), data.get("stats")))

    @app.delete("/admin/api/stats")
    def admin_reset_stats():
        if (err := need_admin_auth()) is not None:
            return err
        with _lock:
            data = load_data()
            data["stats"] = {"total_requests": 0, "ok": 0, "err": 0,
                             "input_tokens": 0, "output_tokens": 0}
            data["logs"] = []
            save_data(data)
        return jsonify({"ok": True})

    @app.post("/admin/api/router-keys")
    def admin_add_router_key():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        label = str(body.get("label", "")).strip() or "user"
        full, prefix = generate_router_key()
        with _lock:
            data = load_data()
            entry = {"id": uuid.uuid4().hex[:8], "key": full, "prefix": prefix,
                     "label": label, "created_at": _now_iso(),
                     "last_used_at": "", "request_count": 0}
            data.setdefault("router_keys", []).append(entry)
            save_data(data)
        # raw key is returned ONLY in this response (never shown again)
        return jsonify({"ok": True, "id": entry["id"], "key": full,
                        "prefix": prefix, "label": label}), 201

    @app.delete("/admin/api/router-keys/<key_id>")
    def admin_del_router_key(key_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        with _lock:
            data = load_data()
            data["router_keys"] = [k for k in data.get("router_keys", []) if k.get("id") != key_id]
            save_data(data)
        return jsonify({"ok": True})

    def _store_provider_key(body: dict):
        key = str(body.get("key", "")).strip()
        label = str(body.get("label", "")).strip() or "provider-key"
        provider = str(body.get("provider", "gemini")).strip().lower() or "gemini"
        if not key:
            return jsonify({"error": "key must not be empty"}), 400
        if provider not in PROVIDERS:
            return jsonify({"error": f"unknown provider (valid: {', '.join(PROVIDERS)})"}), 400
        try:
            protocol, base_url = normalize_provider_key(
                provider, body.get("protocol"), body.get("base_url"))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        with _lock:
            data = load_data()
            if any(k.get("key") == key for k in data.get("provider_keys", [])):
                return jsonify({"error": "this key is already added"}), 400
            data["provider_keys"].append({
                "id": uuid.uuid4().hex[:8], "key": key, "label": label,
                "provider": provider, "protocol": protocol, "base_url": base_url,
                "created_at": _now_iso(),
            })
            save_data(data)
        return jsonify({"ok": True}), 201

    def _delete_provider_key(key_id: str):
        with _lock:
            data = load_data()
            data["provider_keys"] = [k for k in data.get("provider_keys", [])
                                     if k.get("id") != key_id]
            save_data(data)
        return jsonify({"ok": True})

    @app.post("/admin/api/provider-keys")
    def admin_add_provider_key():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        return _store_provider_key(body)

    @app.delete("/admin/api/provider-keys/<key_id>")
    def admin_del_provider_key(key_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        return _delete_provider_key(key_id)

    @app.post("/admin/api/keys")
    def admin_add_key():
        # Legacy alias: adds a gemini-protocol key (backwards compatibility)
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        body = {**body, "provider": "gemini", "protocol": "gemini", "base_url": ""}
        return _store_provider_key(body)

    @app.delete("/admin/api/keys/<key_id>")
    def admin_del_key(key_id: str):
        # Legacy alias (backwards compatibility)
        if (err := need_admin_auth()) is not None:
            return err
        return _delete_provider_key(key_id)

    @app.post("/admin/api/models")
    def admin_add_model():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        name = str(body.get("name", "")).strip()
        # new schema: provider + upstream_model; legacy: gemini_model
        provider = str(body.get("provider", "gemini")).strip().lower() or "gemini"
        upstream_model = (str(body.get("upstream_model", "")).strip()
                          or str(body.get("gemini_model", "")).strip()
                          or "gemini-2.0-flash")
        desc = str(body.get("description", "")).strip()
        if not name:
            return jsonify({"error": "model name (name) must not be empty"}), 400
        if provider not in ("openai", "anthropic", "gemini"):
            return jsonify({"error": "provider must be openai | anthropic | gemini"}), 400
        with _lock:
            data = load_data()
            data["models"] = [m for m in data["models"] if m.get("name") != name]
            data["models"].append({"name": name, "provider": provider,
                                   "upstream_model": upstream_model, "description": desc})
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

    @app.get("/admin/api/settings")
    def admin_get_settings():
        if (err := need_admin_auth()) is not None:
            return err
        data = load_data()
        return jsonify(data.get("settings", {"default_system": ""}))

    @app.post("/admin/api/settings")
    def admin_set_settings():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        with _lock:
            data = load_data()
            data.setdefault("settings", {})
            if "default_system" in body:
                data["settings"]["default_system"] = str(body.get("default_system") or "")
            save_data(data)
        return jsonify({"ok": True})

    def _model_public(m: dict, now: int, fallback: bool = False) -> dict:
        provider = m.get("provider") or "gemini"
        upstream = m.get("upstream_model") or m.get("gemini_model") or m.get("name")
        meta: dict = {"provider": provider, "upstream_model": upstream}
        if provider == "gemini":
            meta["gemini_model"] = upstream  # for legacy clients
        if fallback:
            meta["fallback"] = True
        return {"id": m["name"], "object": "model", "type": "model",
                "owned_by": "samet-router", "created": now,
                "display_name": m["name"], "meta": meta}

    # --- Router: OpenAI + Anthropic compatible model list ---
    @app.get("/v1/models")
    def list_models():
        ok, err, _key = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        data = load_data()
        now = int(time.time())
        return jsonify({
            "object": "list",
            "data": [_model_public(m, now) for m in data.get("models", [])
                     if isinstance(m, dict) and m.get("name")],
        })

    @app.get("/v1/models/<model_id>")
    def get_model(model_id: str):
        ok, err, _key = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        data = load_data()
        now = int(time.time())
        for m in data.get("models", []):
            if isinstance(m, dict) and m.get("name") == model_id:
                return jsonify(_model_public(m, now))
        # unknown model (e.g. claude-*) still returns 200: Claude Code
        # checks model existence, falls back to the fallback target.
        target = resolve_target(data, model_id)
        return jsonify(_model_public({"name": model_id, "provider": target["provider"],
                                      "upstream_model": target["upstream_model"]},
                                     now, fallback=True))

    # --- Router: OpenAI compatible ---
    @app.post("/v1/chat/completions")
    def chat_completions():
        ok, err, key_info = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        body = request.get_json(force=True, silent=True) or {}
        router_model = str(body.get("model", "")).strip() or "samet-flash"
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        t0 = time.time()
        thinking = body_has_thinking("openai", body)
        ep = "openai:/v1/chat/completions"
        kid, klabel = key_info.get("id", ""), key_info.get("label", "")
        try:
            with _lock:
                data = load_data()
            default_system = (data.get("settings", {}) or {}).get("default_system", "")
            target = resolve_target(data, router_model)
            provider, upstream = target["provider"], target["upstream_model"]

            if provider == "openai":
                # one-to-one passthrough (tools, reasoning_effort carried over as-is)
                up_body = dict(body)
                up_body["model"] = upstream
                up_body["stream"] = stream
                if (default_system or "").strip():
                    up_body["messages"] = [
                        {"role": "system", "content": default_system.strip()},
                        *(messages or [])]
                _entry, resp = call_upstream(
                    "openai", up_body, stream=stream,
                    timeout=STREAM_TIMEOUT if stream else 60)
                if resp.status_code == 400 and not stream:
                    retry_body, removed = sanitize_openai_body_for_retry(up_body)
                    if removed:
                        _close_quietly(resp)
                        _entry2, resp2 = call_upstream(
                            "openai", retry_body, stream=False, timeout=60)
                        if resp2.status_code == 200:
                            resp = resp2
                            _entry = _entry2
                        else:
                            _close_quietly(resp)
                            resp = resp2
                if resp.status_code != 200:
                    add_log(ep, router_model, False, int((time.time() - t0) * 1000),
                            _cross_error_message("openai", resp),
                            0, 0, kid, klabel, provider, thinking)
                    return forward_upstream_error("openai", resp)
                if stream:
                    add_log(ep, router_model, True, int((time.time() - t0) * 1000),
                            "", 0, 0, kid, klabel, provider, thinking)
                    return proxy_sse_response(resp, router_model)
                j = resp.json()
                j["model"] = router_model
                u = j.get("usage", {}) or {}
                thinking = thinking or bool(
                    ((j.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content"))
                add_log(ep, router_model, True, int((time.time() - t0) * 1000), "",
                        int(u.get("prompt_tokens", 0) or 0),
                        int(u.get("completion_tokens", 0) or 0),
                        kid, klabel, provider, thinking)
                return jsonify(j)

            if provider == "anthropic":
                abody, thinking_req = openai_to_anthropic_body(body, default_system)
                thinking = thinking or thinking_req
                abody["model"] = upstream
                # upstream Anthropic SSE cannot be converted to OpenAI SSE; buffer and
                # fake-stream it (correct but delayed)
                _entry, resp = call_upstream(
                    "anthropic", abody, stream=False, timeout=60,
                    extra_headers=_forward_anthropic_beta())
                if resp.status_code != 200:
                    add_log(ep, router_model, False, int((time.time() - t0) * 1000),
                            _cross_error_message("anthropic", resp),
                            0, 0, kid, klabel, provider, thinking)
                    return jsonify({"error": _cross_error_message("anthropic", resp)}), resp.status_code
                oai = anthropic_response_to_openai(resp.json(), router_model)
                thinking = thinking or bool(
                    (oai.get("choices") or [{}])[0].get("message", {}).get("reasoning_content"))
                u = oai.get("usage", {}) or {}
                add_log(ep, router_model, True, int((time.time() - t0) * 1000), "",
                        int(u.get("prompt_tokens", 0) or 0),
                        int(u.get("completion_tokens", 0) or 0),
                        kid, klabel, provider, thinking)
                if stream:
                    return Response(
                        stream_with_context(iter_openai_sse(oai)),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                return jsonify(oai)

            # provider == "gemini" (legacy behavior)
            gemini_model = upstream
            system_text, contents = openai_messages_to_gemini(messages)
            system_text = _merge_system(default_system, system_text)
            max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
            stop = body.get("stop")
            if isinstance(stop, str):
                stop = [stop]
            budget = None
            if isinstance(body.get("reasoning_effort"), str):
                budget = budget_for_effort(body["reasoning_effort"])
                thinking = True
            payload = build_gemini_payload(
                system_text, contents,
                max_tokens=max_tokens,
                temperature=body.get("temperature"),
                stop_sequences=stop if isinstance(stop, list) else None,
                top_p=body.get("top_p"),
                top_k=None,
                tools=openai_tools_to_gemini(body.get("tools")),
                tool_config=openai_tool_choice_to_gemini(body.get("tool_choice")),
                thinking_budget=budget,
                include_thoughts=budget is not None,
            )
            g = call_gemini(gemini_model, payload)
            oai = gemini_to_openai_response(g, router_model)
            u = (oai.get("usage") or {})
            thinking = thinking or bool(
                (oai.get("choices") or [{}])[0].get("message", {}).get("reasoning_content"))
            add_log(ep, router_model, True,
                    int((time.time() - t0) * 1000), "",
                    int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0),
                    kid, klabel, provider, thinking)
            if stream:
                return Response(
                    stream_with_context(iter_openai_sse(oai)),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return jsonify(oai)
        except Exception as exc:  # noqa: BLE001 - error is returned to the client in simplified form
            add_log(ep, router_model, False,
                    int((time.time() - t0) * 1000), str(exc),
                    0, 0, kid, klabel, "", thinking)
            return jsonify({"error": str(exc)[:500]}), 502

    # --- Router: Anthropic compatible ---
    @app.post("/v1/messages")
    def anthropic_messages():
        ok, err, key_info = need_router_auth()
        if not ok:
            # return 401 in Anthropic format (Claude Code expects this)
            return jsonify({"type": "error", "error": {
                "type": "authentication_error",
                "message": "unauthorized, x-api-key or Bearer required"}}), 401
        body = request.get_json(force=True, silent=True) or {}
        router_model = str(body.get("model", "")).strip() or "samet-flash"
        stream = bool(body.get("stream", False))
        t0 = time.time()
        ep = "anthropic:/v1/messages"
        kid, klabel = key_info.get("id", ""), key_info.get("label", "")
        thinking = body_has_thinking("anthropic", body)
        try:
            with _lock:
                data = load_data()
            default_system = (data.get("settings", {}) or {}).get("default_system", "")
            target = resolve_target(data, router_model)
            provider, upstream = target["provider"], target["upstream_model"]

            if provider == "anthropic":
                # one-to-one passthrough (thinking + signatures carried over as-is)
                abody = dict(body)
                abody["model"] = upstream
                abody["stream"] = stream
                merged = _merge_anthropic_system(default_system, body.get("system"))
                if merged:
                    abody["system"] = merged
                elif "system" in abody:
                    abody.pop("system", None)
                abody["messages"] = sanitize_anthropic_messages_for_upstream(
                    body.get("messages", []))
                _entry, resp = call_upstream(
                    "anthropic", abody, stream=stream,
                    timeout=STREAM_TIMEOUT if stream else 60,
                    extra_headers=_forward_anthropic_beta())
                if resp.status_code != 200:
                    add_log(ep, router_model, False, int((time.time() - t0) * 1000),
                            _cross_error_message("anthropic", resp),
                            0, 0, kid, klabel, provider, thinking)
                    return forward_upstream_error("anthropic", resp)
                if stream:
                    add_log(ep, router_model, True, int((time.time() - t0) * 1000),
                            "", 0, 0, kid, klabel, provider, thinking)
                    return proxy_sse_response(resp, router_model)
                j = resp.json()
                j["model"] = router_model
                u = j.get("usage", {}) or {}
                thinking = thinking or any(
                    isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
                    for b in j.get("content", []) or [])
                add_log(ep, router_model, True, int((time.time() - t0) * 1000), "",
                        int(u.get("input_tokens", 0) or 0),
                        int(u.get("output_tokens", 0) or 0),
                        kid, klabel, provider, thinking)
                return jsonify(j)

            if provider == "openai":
                obody, _effort = anthropic_to_openai_body(body, default_system)
                obody["model"] = upstream
                # upstream OpenAI SSE cannot be converted to Anthropic SSE; buffer it
                _entry, resp = call_upstream("openai", obody, stream=False, timeout=60)
                if resp.status_code == 400:
                    # Strict gateways (opencode zen, Ollama, vLLM) reject
                    # reasoning_effort / temperature / max_tokens variants with
                    # a generic 400. Retry once with a sanitized payload.
                    retry_body, removed = sanitize_openai_body_for_retry(obody)
                    if removed:
                        _close_quietly(resp)
                        _entry2, resp2 = call_upstream(
                            "openai", retry_body, stream=False, timeout=60)
                        if resp2.status_code == 200:
                            resp = resp2
                            _entry = _entry2
                        else:
                            first_err = _cross_error_message("openai", resp)
                            _close_quietly(resp)
                            resp = resp2
                            second_err = _cross_error_message("openai", resp)
                            hint = (f"retry without {removed} also failed. "
                                    f"payload=({describe_openai_body(obody)})")
                            msg = f"{second_err} [first: {first_err[:300]}] [{hint}]"
                            add_log(ep, router_model, False, int((time.time() - t0) * 1000),
                                    msg, 0, 0, kid, klabel, provider, thinking)
                            return jsonify({"type": "error", "error": {
                                "type": "api_error", "message": msg}}), resp.status_code
                if resp.status_code != 200:
                    msg = (_cross_error_message("openai", resp)
                           + f" [payload=({describe_openai_body(obody)})]")
                    add_log(ep, router_model, False, int((time.time() - t0) * 1000),
                            msg,
                            0, 0, kid, klabel, provider, thinking)
                    return jsonify({"type": "error", "error": {
                        "type": "api_error",
                        "message": msg}}), resp.status_code
                anth = openai_response_to_anthropic(resp.json(), router_model)
                thinking = thinking or any(
                    isinstance(b, dict) and b.get("type") == "thinking"
                    for b in anth.get("content", []) or [])
                u = anth.get("usage", {}) or {}
                add_log(ep, router_model, True, int((time.time() - t0) * 1000), "",
                        int(u.get("input_tokens", 0) or 0),
                        int(u.get("output_tokens", 0) or 0),
                        kid, klabel, provider, thinking)
                if stream:
                    return Response(
                        stream_with_context(iter_anthropic_sse(anth)),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                return jsonify(anth)

            # provider == "gemini" (legacy behavior + thinking mapping)
            gemini_model = upstream
            thinking_on, budget = anthropic_thinking_of_body(body)
            thinking = thinking or thinking_on
            client_system, contents = anthropic_messages_to_gemini(
                body.get("system"), body.get("messages", [])
            )
            system_text = _merge_system(default_system, client_system)
            max_tokens = body.get("max_tokens") or 1024
            stop = body.get("stop_sequences")
            if isinstance(stop, str):
                stop = [stop]
            payload = build_gemini_payload(
                system_text, contents,
                max_tokens=max_tokens,
                temperature=body.get("temperature"),
                stop_sequences=stop if isinstance(stop, list) else None,
                top_p=body.get("top_p"),
                top_k=body.get("top_k"),
                tools=anthropic_tools_to_gemini(body.get("tools")),
                tool_config=anthropic_tool_choice_to_gemini(body.get("tool_choice")),
                thinking_budget=budget if thinking_on else None,
                include_thoughts=thinking_on,
            )
            g = call_gemini(gemini_model, payload)
            anth = gemini_to_anthropic_response(g, router_model,
                                                thinking_enabled=thinking_on)
            u = (anth.get("usage") or {})
            add_log(ep, router_model, True,
                    int((time.time() - t0) * 1000), "",
                    int(u.get("input_tokens", 0) or 0), int(u.get("output_tokens", 0) or 0),
                    kid, klabel, provider, thinking)
            if stream:
                return Response(
                    stream_with_context(iter_anthropic_sse(anth)),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return jsonify(anth)
        except Exception as exc:  # noqa: BLE001
            add_log(ep, router_model, False,
                    int((time.time() - t0) * 1000), str(exc),
                    0, 0, kid, klabel, "", thinking)
            return jsonify({"type": "error", "error": {"type": "api_error",
                            "message": str(exc)[:500]}}), 502

    @app.post("/v1/messages/count_tokens")
    def anthropic_count_tokens():
        ok, err, _key = need_router_auth()
        if not ok:
            return jsonify({"type": "error", "error": {
                "type": "authentication_error", "message": "unauthorized"}}), 401
        body = request.get_json(force=True, silent=True) or {}
        # model is not required; default to it if missing (Claude Code sometimes omits it)
        tokens = estimate_input_tokens(body.get("system"), body.get("messages", []))
        # tools also take tokens; add a rough amount
        try:
            tools_str = json.dumps(body.get("tools", []), ensure_ascii=False)
            tokens += len(tools_str) // 4
        except (TypeError, ValueError):
            pass
        return jsonify({"input_tokens": tokens})

    # --- Router: Gemini compatible ---
    def _gemini_call(model_name: str):
        ok, err, key_info = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        body = request.get_json(force=True, silent=True) or {}
        t0 = time.time()
        ep = f"gemini:/v1beta/models/{model_name}:generateContent"
        kid, klabel = key_info.get("id", ""), key_info.get("label", "")
        thinking = body_has_thinking("gemini", body)
        try:
            with _lock:
                data = load_data()
            target = resolve_target(data, model_name)
            provider, upstream = target["provider"], target["upstream_model"]
            if provider == "gemini":
                g = call_gemini(upstream, body or {"contents": []})
                inp, out = _gemini_usage_of_response(g)
                add_log(ep, model_name, True, int((time.time() - t0) * 1000), "",
                        inp, out, kid, klabel, provider, thinking)
                return jsonify(g)
            # gemini protocol bound to an openai/anthropic provider:
            # first reduce to OpenAI format, then send to the target
            obody = gemini_request_to_openai(body or {"contents": []})
            obody["model"] = upstream
            if provider == "openai":
                _entry, resp = call_upstream("openai", obody, stream=False, timeout=60)
                if resp.status_code != 200:
                    add_log(ep, model_name, False, int((time.time() - t0) * 1000),
                            _cross_error_message("openai", resp),
                            0, 0, kid, klabel, provider, thinking)
                    return jsonify({"error": {
                        "message": _cross_error_message("openai", resp)}}), resp.status_code
                g = openai_response_to_gemini(resp.json())
            else:
                abody, _ = openai_to_anthropic_body(obody)
                abody["model"] = upstream
                _entry, resp = call_upstream(
                    "anthropic", abody, stream=False, timeout=60,
                    extra_headers=_forward_anthropic_beta())
                if resp.status_code != 200:
                    add_log(ep, model_name, False, int((time.time() - t0) * 1000),
                            _cross_error_message("anthropic", resp),
                            0, 0, kid, klabel, provider, thinking)
                    return jsonify({"error": {
                        "message": _cross_error_message("anthropic", resp)}}), resp.status_code
                g = anthropic_response_to_gemini(resp.json())
            inp, out = _gemini_usage_of_response(g)
            add_log(ep, model_name, True, int((time.time() - t0) * 1000), "",
                    inp, out, kid, klabel, provider, thinking)
            return jsonify(g)
        except Exception as exc:  # noqa: BLE001
            add_log(ep, model_name, False, int((time.time() - t0) * 1000), str(exc),
                    0, 0, kid, klabel, "", thinking)
            return jsonify({"error": {"message": str(exc)[:500]}}), 502

    @app.post("/v1beta/models/<path:model_name>:generateContent")
    def gemini_passthrough(model_name: str):
        return _gemini_call(model_name)

    @app.post("/v1beta/models/<path:model_name>:streamGenerateContent")
    def gemini_stream_passthrough(model_name: str):
        # Non-stream fallback instead of a real SSE proxy: the client receives
        # the JSON as a single chunk. Claude Code does not use this endpoint.
        return _gemini_call(model_name)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
