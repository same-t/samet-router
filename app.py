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
  POST   /admin/api/providers/opencode/opencode/connect/start  -> device-code login
  POST   /admin/api/providers/opencode/opencode/connect/poll   -> device-code poll
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import secrets
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
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
COMMANDCODE_BASE = "https://api.commandcode.ai/provider/v1"

PROVIDERS = ("opencode", "openai", "anthropic", "gemini", "commandcode", "custom")
PROTOCOLS = ("openai", "anthropic", "gemini")
OPENCODE_BASE = "https://opencode.ai/zen/v1"
# Since OpenCode v2 (2026-10-14) the keyless free tier moved off /zen/v1 to
# /inference/v1 and only accepts the v2 client fingerprint. Paid/keyed Zen
# requests still use OPENCODE_BASE - see opencode_is_free_model().
OPENCODE_INFERENCE_BASE = "https://opencode.ai/inference/v1"
OPENCODE_V2_UA = "opencode/latest/2.0.26/cli"
OPENCODE_V2_CLIENT = "cli"
OPENCODE_V2_PROJECT = "global"
# v2 rejects x-opencode-request (v1 required it), so we never send it.
OPENCODE_FINGERPRINT_TOOLS = ("read", "shell")
# OpenCode Console account login (device-code OAuth), mirrors opencode's CLI flow:
#   POST {console}/auth/device/code  -> {device_code, user_code, verification_uri_complete, interval}
#   POST {console}/auth/device/token (device_code / refresh_token grant) -> tokens
OPENCODE_CONSOLE = "https://opencode.ai/console"
OPENCODE_CLIENT_ID = "opencode-cli"
OPENCODE_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
OPENCODE_UA = "opencode-cli/1.0 (+samet-router)"

# provider -> defaults (protocol, base_url, auth scheme)
PROVIDER_DEFAULTS: dict[str, dict] = {
    "opencode": {"protocol": "openai", "base_url": OPENCODE_BASE, "auth": "bearer",
                 "key_hint": "public or sk-...", "key_url": "https://opencode.ai/auth"},
    "openai": {"protocol": "openai", "base_url": OPENAI_BASE, "auth": "bearer",
               "key_hint": "sk-...", "key_url": "https://platform.openai.com/api-keys"},
    "anthropic": {"protocol": "anthropic", "base_url": ANTHROPIC_BASE, "auth": "x-api-key",
                  "key_hint": "sk-ant-...", "key_url": "https://console.anthropic.com/"},
    "gemini": {"protocol": "gemini", "base_url": GEMINI_BASE, "auth": "query",
               "key_hint": "AIza...", "key_url": "https://aistudio.google.com/apikey"},
    # Command Code Provider API: one base, /chat/completions + /messages + /models.
    # Routed on the openai protocol, so Claude models (which are /messages-only) are out.
    "commandcode": {"protocol": "openai", "base_url": COMMANDCODE_BASE, "auth": "bearer",
                    "key_hint": "Command Code API key",
                    "key_url": "https://commandcode.ai/settings/keys"},
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
        "combos": [],  # [{id, name, model, description, created_at}] - name -> one existing model
        "models": [  # router model name -> {provider, upstream_model}
            {"name": "samet-flash", "provider": "gemini", "upstream_model": "gemini-2.0-flash",
             "description": "Default fast model"},
            {"name": "samet-pro", "provider": "gemini", "upstream_model": "gemini-2.0-flash",
             "description": "Pro alias (same backend)"},
        ],
        "logs": [],  # [{ts, endpoint, model, provider, thinking, ok, latency_ms, error, ...}]
        "provider_index": {},  # provider -> round-robin order
        "proxy_pools": [],  # [{id, name, proxies: ["http://user:pass@host:port"], created_at}]
        "proxy_index": {},  # rotation cursor per provider/pool
        "provider_settings": {},  # provider -> {proxy_pool, rotation, thinking, headers_active, custom_headers: []}
        "settings": {"default_system": "", "round_robin": True},
        # cumulative counters (unaffected by log rotation)
        "stats": {"total_requests": 0, "ok": 0, "err": 0,
                  "input_tokens": 0, "output_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Storage: SQLite (WAL) with a JSON-file import for existing installs
# ---------------------------------------------------------------------------
# load_data()/save_data() keep their original dict interface, so nothing above
# this layer knows the store changed. SQLite gives real transactions, so a
# crash mid-write can no longer truncate or corrupt the file the old store
# rewrote wholesale on every request.

SCHEMA_VERSION = 1

_SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS provider_keys (id TEXT PRIMARY KEY, ord INTEGER NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS router_keys   (id TEXT PRIMARY KEY, ord INTEGER NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS models        (name TEXT PRIMARY KEY, ord INTEGER NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS combos        (name TEXT PRIMARY KEY, ord INTEGER NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS proxy_pools   (id TEXT PRIMARY KEY, ord INTEGER NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS logs          (seq INTEGER PRIMARY KEY, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv            (k TEXT PRIMARY KEY, doc TEXT NOT NULL);
"""

_LIST_TABLES = {
    "provider_keys": "id",
    "router_keys": "id",
    "models": "name",
    "combos": "name",
    "proxy_pools": "id",
}
_KV_KEYS = ("settings", "stats", "provider_index", "proxy_index", "provider_settings")
_db_local = threading.local()
_db_lock = threading.Lock()


def db_path() -> Path:
    override = os.environ.get("SAMET_ROUTER_DB", "").strip()
    if override:
        return Path(override)
    # derive from DATA_FILE so tests (and SAMET_ROUTER_DATA) keep isolating
    base = Path(str(DATA_FILE))
    return base.with_suffix(".db")


def _connect(path: Path) -> "sqlite3.Connection":
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=15, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SQL_SCHEMA)
    conn.commit()
    return conn


def db() -> "sqlite3.Connection":
    """Per-thread connection, keyed by path so reassigning DATA_FILE reopens."""
    path = db_path()
    conns = getattr(_db_local, "conns", None)
    if conns is None:
        conns = _db_local.conns = {}
    with _db_lock:
        conn = conns.get(path)
        if conn is None:
            if len(conns) >= 4:  # bound memory across many test temp paths
                for c in conns.values():
                    try:
                        c.close()
                    except sqlite3.Error:
                        pass
                conns.clear()
            conn = conns[path] = _connect(path)
        return conn


def _row_id(doc: dict, key: str, fallback: str) -> str:
    return str(doc.get(key) or fallback)


def _write_store(data: dict) -> None:
    """Persist the whole dict in a single transaction (upsert by id)."""
    conn = db()
    with conn:  # one transaction; rolls back as a unit on any error
        for table, key in _LIST_TABLES.items():
            rows = data.get(table) or []
            conn.execute(f"DELETE FROM {table}")
            conn.executemany(
                f"INSERT OR REPLACE INTO {table} ({key}, ord, doc) VALUES (?,?,?)",
                [(_row_id(r, key, f"{table}-{i}"), i, json.dumps(r, ensure_ascii=False))
                 for i, r in enumerate(rows) if isinstance(r, dict)])
        logs = data.get("logs") or []
        conn.execute("DELETE FROM logs")
        conn.executemany(
            "INSERT INTO logs (seq, doc) VALUES (?,?)",
            [(i, json.dumps(r, ensure_ascii=False))
             for i, r in enumerate(logs) if isinstance(r, dict)])
        conn.execute("DELETE FROM kv")
        conn.executemany(
            "INSERT INTO kv (k, doc) VALUES (?,?)",
            [(k, json.dumps(data.get(k), ensure_ascii=False)) for k in _KV_KEYS])
        conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES ('schema_version', ?)",
                     (str(SCHEMA_VERSION),))


def _read_store() -> dict | None:
    conn = db()
    try:
        row = conn.execute("SELECT v FROM meta WHERE k='schema_version'").fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None  # empty store: caller decides seed vs import
    data = default_data()
    for table, key in _LIST_TABLES.items():
        out = []
        for (doc,) in conn.execute(f"SELECT doc FROM {table} ORDER BY ord"):
            try:
                out.append(json.loads(doc))
            except (ValueError, TypeError):
                continue
        data[table] = out
    data["logs"] = [json.loads(doc) for (doc,) in
                    conn.execute("SELECT doc FROM logs ORDER BY seq")
                    if _safe_json(doc) is not None]
    for k in _KV_KEYS:
        for (doc,) in conn.execute("SELECT doc FROM kv WHERE k=?", (k,)):
            try:
                data[k] = json.loads(doc)
            except (ValueError, TypeError):
                pass
    return data


def _safe_json(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _import_json_file(data: dict) -> bool:
    """One-time import of an existing data.json into the seed dict."""
    if not DATA_FILE.exists():
        return False
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            legacy = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(legacy, dict) or not legacy:
        return False
    data.update(legacy)
    try:  # keep the original as a backup next to the db
        DATA_FILE.replace(DATA_FILE.with_suffix(".json.bak"))
    except OSError:
        pass
    return True


def load_data() -> dict:
    stored = _read_store()
    if stored is not None:
        return _normalize_data(stored)
    seed = default_data()
    if _import_json_file(seed):
        # normalize before persisting: the legacy migration renames keys that
        # the sqlite schema has no table for, so it must happen while we still
        # have the raw dict in hand.
        seed = _normalize_data(seed)
    _write_store(seed)
    return seed


def save_data(data: dict) -> None:
    _write_store(data)


def _normalize_data(data: dict) -> dict:
    if not isinstance(data, dict):
        data = default_data()
    for k, v in default_data().items():
        data.setdefault(k, v)
    # also migrate the settings content
    if not isinstance(data.get("settings"), dict):
        data["settings"] = {"default_system": "", "round_robin": True}
    data["settings"].setdefault("default_system", "")
    data["settings"].setdefault("round_robin", True)
    if not isinstance(data.get("router_keys"), list):
        data["router_keys"] = []
    if not isinstance(data.get("provider_keys"), list):
        data["provider_keys"] = []
    if not isinstance(data.get("provider_index"), dict):
        data["provider_index"] = {}
    if not isinstance(data.get("proxy_pools"), list):
        data["proxy_pools"] = []
    if not isinstance(data.get("proxy_index"), dict):
        data["proxy_index"] = {}
    if not isinstance(data.get("provider_settings"), dict):
        data["provider_settings"] = {}
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


# ---------------------------------------------------------------------------
# Rate-limit cooldown + stale-response fallback
# ---------------------------------------------------------------------------
# OpenCode answers a rate-limited request with 429 and, in some paths, an
# explicit Retry-After: seconds-until-the-next-00:00-UTC for the daily free
# quota, or a short window for a burst ceiling. When the header is present we
# honour it (capped); otherwise we fall back to the configured cooldown.

RATE_LIMIT_COOLDOWN = 300.0        # 5 minutes, per the routing rule
RATE_LIMIT_COOLDOWN_MAX = 86400.0  # never park a key longer than a day
CACHE_MAX_ENTRIES = 64
CACHE_MAX_AGE = 3600.0

_cooldowns: dict[str, float] = {}       # key id -> epoch seconds when usable again
_response_cache: dict[tuple, dict] = {}  # (provider, model) -> {"json", "at"}


class AllKeysCoolingDown(RuntimeError):
    """Every key for this target is serving a rate-limit cooldown."""

    def __init__(self, retry_after: float, keys: int):
        self.retry_after = retry_after
        self.keys = keys
        super().__init__(f"all {keys} key(s) cooling down for up to {retry_after:.0f}s")


def _cooldown_key(entry: dict) -> str:
    return str(entry.get("id") or entry.get("key") or "")


def prune_cooldowns(now: float | None = None) -> None:
    now = time.time() if now is None else now
    for k in [k for k, until in _cooldowns.items() if until <= now]:
        _cooldowns.pop(k, None)


def key_is_cooling(entry: dict, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    until = _cooldowns.get(_cooldown_key(entry))
    return bool(until and until > now)


def key_cooldown_remaining(entry: dict, now: float | None = None) -> float:
    now = time.time() if now is None else now
    return max(0.0, _cooldowns.get(_cooldown_key(entry), 0.0) - now)


def mark_key_cooling(entry: dict, seconds: float | None = None) -> float:
    """Park a key. Returns the seconds applied."""
    secs = RATE_LIMIT_COOLDOWN if seconds is None else float(seconds)
    secs = max(RATE_LIMIT_COOLDOWN, min(secs, RATE_LIMIT_COOLDOWN_MAX))
    ident = _cooldown_key(entry)
    if not ident:
        return 0.0
    _cooldowns[ident] = time.time() + secs
    prune_cooldowns()
    return secs


def retry_after_seconds(resp) -> float | None:
    """Parse Retry-After / retry-after-ms / x-ratelimit-reset from a 429."""
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    try:
        # requests' CaseInsensitiveDict already normalises, but a plain dict in
        # tests or a hand-rolled response should work too.
        lower = {str(k).lower(): v for k, v in headers.items()}
    except (AttributeError, TypeError):
        return None
    raw = lower.get("retry-after-ms")
    if raw not in (None, ""):
        try:
            return max(0.0, float(raw) / 1000.0)
        except (TypeError, ValueError):
            pass
    raw = lower.get("retry-after")
    if raw not in (None, ""):
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            import email.utils as _eu
            try:
                dt = _eu.parsedate_to_datetime(str(raw))
                if dt is not None:
                    return max(0.0, dt.timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    return None


def cooling_key_report(provider: str) -> list[dict]:
    """Cooling keys for a provider, newest first (for the admin UI)."""
    now = time.time()
    prune_cooldowns(now)
    with _lock:
        data = load_data()
    out = []
    for entry in data.get("provider_keys", []):
        if not isinstance(entry, dict) or (entry.get("provider") or "") != provider:
            continue
        left = key_cooldown_remaining(entry, now)
        if left > 0:
            out.append({"id": entry.get("id"), "label": entry.get("label"),
                        "retry_after": round(left, 1)})
    return sorted(out, key=lambda r: -r["retry_after"])


def cache_store_response(provider: str, model: str, payload: dict) -> None:
    """Remember the last good non-streaming response for a target."""
    if not isinstance(payload, dict) or not provider or not model:
        return
    _response_cache[(provider, model)] = {"json": payload, "at": time.time()}
    if len(_response_cache) > CACHE_MAX_ENTRIES:
        oldest = min(_response_cache.items(), key=lambda kv: kv[1]["at"])[0]
        _response_cache.pop(oldest, None)


def cache_get_response(provider: str, model: str) -> dict | None:
    hit = _response_cache.get((provider, model))
    if not hit:
        return None
    if time.time() - hit["at"] > CACHE_MAX_AGE:
        return None
    return hit


# ---------------------------------------------------------------------------
# Per-key request accounting
# ---------------------------------------------------------------------------
# Measured 2026-10-18: a successful 200 carries no rate-limit headers at all
# (only cloudflare plumbing plus x-opencode-log-id and x-zen-model), and the
# console exposes no usage/balance/limits endpoint - /api/user is the only
# authenticated route and returns just id/email/role. Upstream published a
# feature request for exactly this (anomalyco/opencode#55816). So remaining
# quota cannot be read from the wire; these counters are the local proxy for it.

_key_usage: dict[str, dict] = {}
USAGE_KEYS = ("requests_since_reset", "requests_total", "last_429_at",
              "last_429_retry_after", "last_ok_at", "upstream_log_id",
              "upstream_model", "last_status")


def _usage_slot(entry: dict) -> dict:
    ident = _cooldown_key(entry)
    slot = _key_usage.get(ident)
    if slot is None:
        slot = _key_usage[ident] = {
            "requests_since_reset": 0, "requests_total": 0,
            "last_429_at": 0, "last_429_retry_after": 0.0,
            "last_ok_at": 0, "upstream_log_id": "", "upstream_model": "",
            "last_status": 0,
        }
    return slot


def note_request_sent(entry: dict) -> None:
    slot = _usage_slot(entry)
    slot["requests_since_reset"] += 1
    slot["requests_total"] += 1


def note_rate_limited(entry: dict, retry_after: float) -> None:
    slot = _usage_slot(entry)
    slot["last_429_at"] = time.time()
    slot["last_429_retry_after"] = retry_after
    slot["last_status"] = 429
    slot["requests_since_reset"] = 0  # quota window just rolled over


def note_upstream_result(entry: dict, resp) -> None:
    """Record what upstream said, including the two headers it does send."""
    slot = _usage_slot(entry)
    status = int(getattr(resp, "status_code", 0) or 0)
    slot["last_status"] = status
    headers = getattr(resp, "headers", None)
    if headers:
        try:
            lower = {str(k).lower(): v for k, v in headers.items()}
        except (AttributeError, TypeError):
            lower = {}
        # trace id is what opencode support asks for
        if lower.get("x-opencode-log-id"):
            slot["upstream_log_id"] = str(lower["x-opencode-log-id"])
        # echoes the model upstream actually served, so a silent substitution shows up
        if lower.get("x-zen-model"):
            slot["upstream_model"] = str(lower["x-zen-model"])
    if status == 200:
        slot["last_ok_at"] = time.time()


def key_usage_report(provider: str) -> list[dict]:
    """Per-key accounting for the admin UI."""
    with _lock:
        data = load_data()
    now = time.time()
    out = []
    for entry in data.get("provider_keys", []):
        if not isinstance(entry, dict) or (entry.get("provider") or "") != provider:
            continue
        ident = _cooldown_key(entry)
        slot = _key_usage.get(ident)
        row = {
            "id": entry.get("id"), "label": entry.get("label"),
            "requests_since_reset": (slot or {}).get("requests_since_reset", 0),
            "requests_total": (slot or {}).get("requests_total", 0),
            "last_status": (slot or {}).get("last_status", 0),
            "upstream_log_id": (slot or {}).get("upstream_log_id", ""),
            "upstream_model": (slot or {}).get("upstream_model", ""),
            "last_429_at": (slot or {}).get("last_429_at", 0),
            "last_429_in": (max(0.0, now - (slot or {}).get("last_429_at", 0))
                            if (slot or {}).get("last_429_at") else 0),
            "last_ok_in": (max(0.0, now - (slot or {}).get("last_ok_at", 0))
                           if (slot or {}).get("last_ok_at") else 0),
        }
        out.append(row)
    return out


def served_from_cache(provider_id: str, upstream_model: str, router_model: str,
                      cooling: AllKeysCoolingDown, anthropic: bool = False):
    """Serve the last good response while every key is rate-limited.

    This is a liveness shim, not a cache: the payload answers an earlier
    prompt, not this one. It is flagged in the body and in X-Samet-Router so a
    caller can tell. With nothing cached we return an honest 429 that carries
    the remaining cooldown so the client can back off.
    """
    wait = max(1, int(cooling.retry_after))
    stale = cache_get_response(provider_id, upstream_model)
    if stale is None:
        message = ("every key for %s is rate limited; retry in about %ds"
                   % (provider_id or "this provider", wait))
        payload = ({"type": "error", "error": {"type": "rate_limit_error",
                                               "message": message}} if anthropic
                   else {"error": {"message": message, "type": "rate_limit_error"}})
        resp = jsonify(payload)
        resp.status_code = 429
        resp.headers["Retry-After"] = str(wait)
        resp.headers["X-Samet-Router"] = "all-keys-cooling"
        return resp

    out = dict(stale["json"])
    if anthropic:
        out["type"] = "message"
    out["model"] = router_model
    out["samet_router"] = {
        "stale": True,
        "reason": "all provider keys are rate limited",
        "served_at": int(stale["at"]),
        "age_seconds": int(time.time() - stale["at"]),
        "retry_after": wait,
    }
    resp = jsonify(out)
    resp.headers["X-Samet-Router"] = "stale-response"
    resp.headers["Retry-After"] = str(wait)
    return resp


def provider_catalog() -> list[dict]:
    """Provider catalog for the admin UI (pure function)."""
    out = []
    for pid in PROVIDERS:
        d = PROVIDER_DEFAULTS[pid]
        out.append({"id": pid, "protocol": d["protocol"], "base_url": d["base_url"],
                    "auth": d["auth"], "key_hint": d["key_hint"], "key_url": d["key_url"]})
    return out


PROVIDER_DISPLAY = {"opencode": "OpenCode Free", "commandcode": "Command Code"}
PROVIDER_SHORT = {"opencode": "oc", "openai": "oai", "anthropic": "ant",
                  "gemini": "gem", "commandcode": "cmd"}

# curated free OpenCode Zen models (shown as suggestions on the provider page)
OPENCODE_SUGGESTIONS = [
    "big-pickle", "beam-free", "mimo-v2.6-flash-free",
    "nemotron-3.5-super-v1-free", "nemotron-3.5-lightning-free",
    "longcat-2.5-preview-free", "deepseek-v4.1-flash-free",
]

# Command Code models that answer on /chat/completions (Claude is /messages-only)
# Note: the model page slug (commandcode.ai/models/beam-free) is NOT the api id;
# the api id for Beam 1.0 is "reflection/beam:free".
COMMANDCODE_SUGGESTIONS = [
    "reflection/beam:free",
    "deepseek/deepseek-v4-flash", "moonshotai/Kimi-K2.7-Code",
    "z-ai/glm-5.3-flashx", "Qwen/Qwen3.8-Max", "xiaomi/mimo-v2.6-pro",
    "google/gemini-3.8-flash", "gpt-5.5", "MiniMaxAI/MiniMax-M3",
]
PROVIDER_SUGGESTIONS = {"opencode": OPENCODE_SUGGESTIONS,
                        "commandcode": COMMANDCODE_SUGGESTIONS}

DEFAULT_PROVIDER_SETTINGS = {
    "proxy_pool": "none",       # proxy pool id or "none" (direct)
    "rotation": "none",         # none | round_robin | random
    "thinking": "auto",         # auto | on | off
    "headers_active": False,
    "custom_headers": [],       # [{name, value, enabled}]
}


def provider_display_name(provider: str) -> str:
    provider = (provider or "").strip().lower()
    return PROVIDER_DISPLAY.get(provider, provider.capitalize() or provider)


def provider_short(provider: str) -> str:
    provider = (provider or "").strip().lower()
    return PROVIDER_SHORT.get(provider, provider[:3] or "m")


def route_protocol_for_provider(provider: str) -> str:
    """Map a provider id to the protocol used for routing (opencode -> openai)."""
    provider = (provider or "").strip().lower()
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["gemini"])["protocol"]


def get_provider_settings(data: dict, provider: str) -> dict:
    stored = (data.get("provider_settings", {}) or {}).get(provider, {}) or {}
    merged = {**DEFAULT_PROVIDER_SETTINGS, **stored}
    if not isinstance(merged.get("custom_headers"), list):
        merged["custom_headers"] = []
    return merged


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


def router_auth_enabled(data: dict) -> bool:
    """Whether a request without a valid key is rejected.

    The stored setting wins when present. When it is absent the panel behaves
    exactly as it did before the toggle existed — auth on iff a key exists —
    so upgrading never changes who can reach the API. The toggle exists for the
    case the old rule could not express: keys you create for tracking without
    closing the router.
    """
    st = data.get("settings") or {}
    if "require_api_key" in st:
        return bool(st.get("require_api_key"))
    return bool(ROUTER_API_KEY or data.get("router_keys"))


# --- usage & analytics ------------------------------------------------------
# `range` -> (bucket size, window). `None` means "everything retained", which is
# capped by the logs ring buffer rather than by the window.
USAGE_RANGES: dict[str, tuple[str, timedelta | None]] = {
    "1h":  ("hour", timedelta(hours=1)),
    "24h": ("hour", timedelta(hours=24)),
    "7d":  ("day",  timedelta(days=7)),
    "30d": ("day",  timedelta(days=30)),
    "60d": ("day",  timedelta(days=60)),
    "all": ("day",  None),
}


def _parse_ts(value: object) -> datetime | None:
    """Log timestamps are ISO strings written by _now_iso()."""
    s = str(value or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _bucket_key(dt: datetime, bucket: str) -> str:
    if bucket == "hour":
        return dt.strftime("%Y-%m-%dT%H:00")
    return dt.strftime("%Y-%m-%d")


def _blank_point(key: str) -> dict:
    return {"t": key, "requests": 0, "ok": 0, "err": 0,
            "input_tokens": 0, "output_tokens": 0}


def _fill_gaps(points: dict[str, dict], start: datetime, end: datetime,
               bucket: str) -> list[dict]:
    """A gap in the series is a zero, not a missing row.

    Without this a chart silently draws a straight line across an hour in which
    nothing was requested, which reads as traffic that never happened.
    """
    out, cur = [], start
    step = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
    if bucket == "hour":
        cur = cur.replace(minute=0, second=0, microsecond=0)
    else:
        cur = cur.replace(hour=0, minute=0, second=0, microsecond=0)
    guard = 0
    while cur <= end and guard < 2000:
        key = _bucket_key(cur, bucket)
        out.append(points.get(key) or _blank_point(key))
        cur += step
        guard += 1
    return out


def _cost(pricing: dict, model: str, inp: int, out: int) -> float | None:
    """Cost from the operator's own price list. None when unpriced.

    There is no bundled price table on purpose: a made-up rate would turn every
    cost figure on the page into a number that means nothing. An entry that
    happens to price everything at zero is a real answer, so "absent" and
    "present but zero" have to stay distinguishable — hence no `or {}`.
    """
    if not isinstance(pricing, dict):
        return None
    entry = pricing.get(model)
    if entry is None:
        entry = pricing.get("*")
    if not isinstance(entry, dict) or not entry:
        return None
    try:
        return round(inp / 1e6 * float(entry.get("input", 0))
                     + out / 1e6 * float(entry.get("output", 0)), 6)
    except (TypeError, ValueError):
        return None


def build_usage(data: dict, rng: str) -> dict:
    """Everything the Usage page renders, derived from the logs ring buffer."""
    bucket, window = USAGE_RANGES[rng]
    logs = data.get("logs", []) or []
    settings = data.get("settings") or {}
    pricing = settings.get("pricing") or {}
    now = datetime.now(timezone.utc)
    cutoff = now - window if window else None

    rows = []
    for l in logs:
        dt = _parse_ts(l.get("ts"))
        if dt is None:
            continue
        if cutoff and dt < cutoff:
            continue
        rows.append((dt, l))

    points: dict[str, dict] = {}
    models: dict[str, dict] = {}
    providers: dict[str, dict] = {}
    totals = {"requests": 0, "ok": 0, "err": 0,
              "input_tokens": 0, "output_tokens": 0}
    lat: list[int] = []

    for dt, l in rows:
        key = _bucket_key(dt, bucket)
        p = points.setdefault(key, _blank_point(key))
        ok = bool(l.get("ok"))
        try:
            inp = int(l.get("input_tokens", 0) or 0)
            out = int(l.get("output_tokens", 0) or 0)
        except (TypeError, ValueError):
            inp = out = 0

        p["requests"] += 1
        p["ok" if ok else "err"] += 1
        p["input_tokens"] += inp
        p["output_tokens"] += out
        totals["requests"] += 1
        totals["ok" if ok else "err"] += 1
        totals["input_tokens"] += inp
        totals["output_tokens"] += out
        if isinstance(l.get("latency_ms"), (int, float)):
            lat.append(int(l["latency_ms"]))

        mname = str(l.get("model", "") or "?")
        m = models.setdefault(mname, {
            "model": mname, "provider": str(l.get("provider", "") or "?"),
            "requests": 0, "ok": 0, "err": 0,
            "input_tokens": 0, "output_tokens": 0, "lat_sum": 0, "lat_n": 0,
            "last_used": "",
        })
        m["requests"] += 1
        m["ok" if ok else "err"] += 1
        m["input_tokens"] += inp
        m["output_tokens"] += out
        if isinstance(l.get("latency_ms"), (int, float)):
            m["lat_sum"] += int(l["latency_ms"]); m["lat_n"] += 1
        iso = str(l.get("ts", "") or "")
        if iso > m["last_used"]:
            m["last_used"] = iso

        pv = str(l.get("provider", "") or "?")
        pr = providers.setdefault(pv, {
            "provider": pv, "requests": 0, "input_tokens": 0,
            "output_tokens": 0, "last_used": "",
        })
        pr["requests"] += 1
        pr["input_tokens"] += inp
        pr["output_tokens"] += out
        if iso > pr["last_used"]:
            pr["last_used"] = iso

    by_model = []
    for m in models.values():
        m = dict(m)
        m["total_tokens"] = m["input_tokens"] + m["output_tokens"]
        m["avg_latency_ms"] = int(m["lat_sum"] / m["lat_n"]) if m["lat_n"] else 0
        m.pop("lat_sum", None); m.pop("lat_n", None)
        m["cost"] = _cost(pricing, m["model"], m["input_tokens"], m["output_tokens"])
        by_model.append(m)
    by_model.sort(key=lambda m: -m["requests"])

    by_provider = sorted(providers.values(),
                         key=lambda p: -p["requests"])
    for p in by_provider:
        p["total_tokens"] = p["input_tokens"] + p["output_tokens"]

    start = min((dt for dt, _ in rows), default=now - (window or timedelta(days=7)))
    series = _fill_gaps(points, start, now, bucket)

    total_cost = None
    if by_model and all(m["cost"] is not None for m in by_model):
        total_cost = round(sum(m["cost"] for m in by_model), 4)

    recent = []
    for dt, l in sorted(rows, key=lambda r: r[0], reverse=True)[:60]:
        try:
            ri = int(l.get("input_tokens", 0) or 0)
            ro = int(l.get("output_tokens", 0) or 0)
        except (TypeError, ValueError):
            ri = ro = 0
        recent.append({
            "ts": l.get("ts", ""), "model": l.get("model", ""),
            "provider": l.get("provider", ""), "ok": bool(l.get("ok")),
            "input_tokens": ri, "output_tokens": ro,
            "latency_ms": l.get("latency_ms"), "error": l.get("error", ""),
            "api_key_label": l.get("api_key_label", ""),
            "endpoint": l.get("endpoint", ""),
        })

    return {
        "range": rng,
        "bucket": bucket,
        "generated_at": now.isoformat(),
        "summary": {
            "total_requests": totals["requests"],
            "ok": totals["ok"],
            "err": totals["err"],
            "success_rate": round(100.0 * totals["ok"] / totals["requests"], 1)
                            if totals["requests"] else 0,
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "total_tokens": totals["input_tokens"] + totals["output_tokens"],
            "avg_latency_ms": int(sum(lat) / len(lat)) if lat else 0,
            "cost": total_cost,
        },
        "series": series,
        "by_model": by_model,
        "by_provider": by_provider,
        "recent": recent,
        "pricing_configured": bool(pricing),
        "retained_logs": len(logs),
    }


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


def apply_thinking_default(body: dict, protocol: str, mode: str) -> None:
    """Mutate `body` to honor a provider-level thinking default (in place).

    mode: "auto" (leave as-is) | "on" (enable if the client did not) |
    "off" (force disable).
    """
    mode = (mode or "auto").strip().lower()
    if mode == "auto" or not isinstance(body, dict):
        return
    if protocol == "openai":
        if mode == "on":
            if not body.get("reasoning_effort"):
                body["reasoning_effort"] = "medium"
        else:
            body.pop("reasoning_effort", None)
            body.pop("reasoning", None)
            for m in body.get("messages", []) or []:
                if isinstance(m, dict):
                    m.pop("reasoning_content", None)
    elif protocol == "anthropic":
        t = body.get("thinking")
        if mode == "on":
            if not (isinstance(t, dict) and t.get("type") == "enabled"):
                body["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        else:
            body.pop("thinking", None)
    else:  # gemini
        gc = body.get("generationConfig")
        if mode == "on":
            if not isinstance(gc, dict):
                gc = {}
                body["generationConfig"] = gc
            if not gc.get("thinkingConfig"):
                gc["thinkingConfig"] = {"thinkingBudget": 4096, "includeThoughts": True}
        elif isinstance(gc, dict):
            gc.pop("thinkingConfig", None)


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


# DeepSeek/OpenCode-style thinking mode requires `reasoning_content` to be
# passed back on assistant messages. OpenAI SDKs drop this non-standard field
# when rebuilding history, so the next turn after a tool call fails with
# "The `reasoning_content` in the thinking mode must be passed back to the API".
_REASONING_MODEL_HINTS = ("deepseek", "reasoner", "qwq", "thinking", "thinking-free")


def _openai_wants_reasoning_replay(body: dict, messages: list, upstream_model: str,
                                   provider_id: str) -> bool:
    if body.get("reasoning_effort"):
        return True
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if "reasoning_content" in msg:
            return True
        content = msg.get("content")
        if isinstance(content, list) and any(
                isinstance(p, dict) and p.get("type") in ("reasoning", "reasoning_content")
                for p in content):
            return True
    if (provider_id or "").strip().lower() in ("opencode", "commandcode"):
        return True
    model = (upstream_model or "").lower()
    return any(h in model for h in _REASONING_MODEL_HINTS)


def ensure_openai_reasoning_replay(body: dict, upstream_model: str = "",
                                   provider_id: str = "", mode: str = "auto") -> bool:
    """Inject a placeholder `reasoning_content` on assistant messages (in place).

    Thinking-mode upstreams (DeepSeek, OpenCode Zen) validate that every
    assistant message carries the field, even when the client dropped it.
    An empty string satisfies the validator; returns True when we added one.
    `mode == "off"` means the provider disabled thinking, so leave the body.
    """
    if not isinstance(body, dict) or (mode or "auto").strip().lower() == "off":
        return False
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    if not _openai_wants_reasoning_replay(body, messages, upstream_model, provider_id):
        return False
    injected = False
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant" \
                and msg.get("reasoning_content") is None:
            msg["reasoning_content"] = ""
            injected = True
    return injected


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


COMBO_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def list_combos(data: dict) -> list[dict]:
    """Enabled combos, oldest first."""
    return [c for c in (data.get("combos") or [])
            if isinstance(c, dict) and str(c.get("name") or "").strip()]


def find_combo(data: dict, name: str) -> dict | None:
    name = str(name or "").strip()
    if not name:
        return None
    for c in list_combos(data):
        if str(c.get("name") or "").strip() == name:
            return c
    return None


def resolve_target(data: dict, router_model: str) -> dict:
    """Resolve a router alias to its target -> {name, provider, upstream_model}.

    An unknown name starting with gemini-* passes through as-is (passthrough); in
    other cases it falls back to the first model (legacy claude-* fallback behavior).
    Combo names are resolved first: they alias exactly one existing model.
    """
    router_model = (router_model or "").strip() or "samet-flash"
    combo = find_combo(data, router_model)
    if combo and combo.get("model"):
        inner = resolve_target(data, str(combo["model"]))
        return {**inner, "name": router_model, "combo_model": str(combo["model"])}
    models = [m for m in data.get("models", [])
              if isinstance(m, dict) and not m.get("disabled")]
    for m in models:
        if m.get("name") == router_model:
            provider = route_protocol_for_provider(m.get("provider") or "gemini")
            upstream = m.get("upstream_model") or m.get("gemini_model") or router_model
            return {"name": m["name"], "provider": provider,
                    "provider_id": m.get("provider") or "gemini",
                    "upstream_model": str(upstream)}
    low = router_model.lower()
    if low.startswith("gemini-") or low.startswith("gemma-") or low.startswith("models/"):
        return {"name": router_model, "provider": "gemini", "provider_id": "gemini",
                "upstream_model": router_model.replace("models/", "")}
    if models:
        m0 = models[0]
        provider = route_protocol_for_provider(m0.get("provider") or "gemini")
        return {"name": router_model, "provider": provider,
                "provider_id": m0.get("provider") or "gemini",
                "upstream_model": str(m0.get("upstream_model") or m0.get("gemini_model")
                                      or "gemini-2.0-flash")}
    return {"name": router_model, "provider": "gemini", "provider_id": "gemini",
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

def _pool_for_protocol(data: dict, protocol: str, provider: str = "") -> list[dict]:
    """Keys speaking the given protocol (including custom).

    When `provider` is given the pool is narrowed to that provider, so a model
    mapped to e.g. commandcode is not sent to another openai-protocol vendor.
    Falls back to every key on that protocol if the provider has none, which
    keeps resolve_target()'s unknown-model fallback working.
    """
    keys = [k for k in data.get("provider_keys", [])
            if isinstance(k, dict) and k.get("key")
            and effective_protocol(k) == protocol]
    provider = (provider or "").strip().lower()
    if provider:
        pinned = [k for k in keys if (k.get("provider") or "") == provider]
        if pinned:
            return pinned
    return keys


def _ordered_pool(data: dict, protocol: str, provider: str = "") -> list[dict]:
    pool = _pool_for_protocol(data, protocol, provider)
    if not pool:
        scope = f" for provider {provider}" if provider else ""
        raise RuntimeError(
            f"No provider key configured for protocol {protocol}{scope}. "
            "Add one in /admin first.")
    # lower priority number = tried first (1 is highest priority)
    pool = sorted(pool, key=lambda k: int(k.get("priority", 1) or 1))
    prune_cooldowns()
    usable = [k for k in pool if not key_is_cooling(k)]
    if not usable:
        # keys exist but every one is serving a rate-limit cooldown
        worst = max((key_cooldown_remaining(k) for k in pool), default=0.0)
        raise AllKeysCoolingDown(worst, len(pool))
    round_robin = bool((data.get("settings", {}) or {}).get("round_robin", True))
    if not round_robin:
        return usable
    cursor = pool_cursor(protocol, provider)
    start = int((data.get("provider_index", {}) or {}).get(cursor, 0) or 0) % len(usable)
    return usable[start:] + usable[:start]


def pool_cursor(protocol: str, provider: str = "") -> str:
    """provider_index key: per-provider when pinned, else shared per protocol."""
    provider = (provider or "").strip().lower()
    return f"{protocol}:{provider}" if provider else protocol


def _advance_provider_index(protocol: str, nkeys: int, provider: str = "") -> None:
    with _lock:
        data = load_data()
        idx = data.setdefault("provider_index", {})
        cur = pool_cursor(protocol, provider)
        idx[cur] = (int(idx.get(cur, 0) or 0) + 1) % max(nkeys, 1)
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


def _proxy_dict(proxy: str | None):
    proxy = (proxy or "").strip()
    return {"http": proxy, "https": proxy} if proxy else None


def _pick_proxy(data: dict, provider: str) -> str | None:
    """Select an outbound proxy for a provider (rotation aware).

    Mutates `data["proxy_index"]`; the caller persists it. `rotation`:
    none (single selected pool, round-robin within it) | round_robin
    (rotate across all pools) | random.
    """
    settings = get_provider_settings(data, provider)
    pools = [p for p in data.get("proxy_pools", []) if isinstance(p, dict)]
    pool_map = {p.get("id"): p for p in pools}
    rotation = str(settings.get("rotation") or "none").lower()
    index = data.setdefault("proxy_index", {})
    if rotation in ("round_robin", "random") and pools:
        entries = [str(px).strip() for p in pools
                   for px in (p.get("proxies") or []) if str(px).strip()]
        if not entries:
            return None
        if rotation == "random":
            return entries[random.randrange(len(entries))]
        key = f"rot:{provider}"
        i = int(index.get(key, 0) or 0) % len(entries)
        index[key] = i + 1
        return entries[i]
    pool = pool_map.get(str(settings.get("proxy_pool") or "none"))
    if not pool:
        return None
    proxies = [str(px).strip() for px in (pool.get("proxies") or []) if str(px).strip()]
    if not proxies:
        return None
    key = f"pool:{pool.get('id')}"
    i = int(index.get(key, 0) or 0) % len(proxies)
    index[key] = i + 1
    return proxies[i]


def provider_custom_headers(data: dict, provider: str) -> dict:
    """Active custom headers for a provider (applied to upstream requests)."""
    settings = get_provider_settings(data, provider)
    if not settings.get("headers_active"):
        return {}
    out: dict = {}
    for h in settings.get("custom_headers") or []:
        if isinstance(h, dict) and h.get("enabled", True) and str(h.get("name", "")).strip():
            out[str(h["name"]).strip()] = str(h.get("value", ""))
    return out


def _opencode_console_headers(token: str | None = None, extra: dict | None = None) -> dict:
    h: dict = {"Content-Type": "application/json", "Accept": "application/json",
               "User-Agent": OPENCODE_UA}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if extra:
        h.update({k: v for k, v in extra.items() if v})
    return h


# ---------------------------------------------------------------------------
# OpenCode v2 free-tier fingerprint (keyless /inference/v1 lane)
# ---------------------------------------------------------------------------
# Upstream compares description AND parameters byte-for-byte against the v2 CLI,
# so do not reword, reorder or reformat these two definitions.
OPENCODE_V2_TOOL_DEFS: dict[str, dict] = {
    "read": {
        "description": (
            "Read the contents of a file or directory. Supports text files, images, and PDFs. "
            "Images and PDFs are presented directly to the model. Each text line is prefixed by "
            "its 1-based line number as <line>: <content>. The prefix is for reference and is not "
            "part of the file content. Directory entries are returned one per line. Use offset and "
            "limit to read large files or directories in sections. Prefer one larger read over many "
            "small slices, and use grep to find specific content in large files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory to read"},
                "offset": {"type": "integer", "minimum": 0,
                           "description": "The line or directory entry to start reading from (1-based)"},
                "limit": {"type": "integer", "minimum": 0,
                          "description": "The maximum number of lines or directory entries to read "
                                         "(defaults to and capped at 2000)"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "shell": {
        "description": (
            "Execute a shell command and return its output. Commands run on Linux using bash. "
            "Quote file paths containing spaces or special characters. Prefer dedicated tools over "
            "shell commands when possible. When output is large, the full result is saved to a file "
            "and a truncated preview is returned. Rely on automatic truncation unless filtering the "
            "output is more useful. Commands accept an optional timeout, background commands have no "
            "timeout by default. Background commands return immediately, and you will be notified "
            "when they complete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command string to execute"},
                "workdir": {"type": "string",
                            "description": "Working directory to execute the command in. Defaults to "
                                           "the current working directory. When possible, avoid "
                                           "changing directories in the command and set the working "
                                           "directory here instead."},
                "timeout": {"type": "integer", "minimum": 0,
                            "description": "Timeout in milliseconds. Set to 0 to disable the timeout. "
                                           "Defaults to 120000 for foreground commands. Background "
                                           "commands have no timeout by default."},
                "background": {"type": "boolean",
                               "description": "Run the command in the background and return immediately "
                                              "(useful for dev servers and long-running builds). You do "
                                              "not need to use '&' at the end of the command when using "
                                              "this parameter. You will be notified when it completes. "
                                              "DO NOT poll for completion."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_OC_PREFIX_MASK = (1 << 48) - 1
_OC_SESSIONS: dict[str, tuple[str, float]] = {}
_OC_SESSION_TTL = 3600.0


def opencode_is_free_model(model: str) -> bool:
    """True for the keyless free-tier lane (v2 /inference/v1 + fingerprint)."""
    base = opencode_base_model(model)
    return base.endswith("-free") or ":free" in base


def opencode_base_model(model: str) -> str:
    """Strip the 'model(level)' thinking suffix 9router uses for catalog lookup."""
    return str(model or "").split("(")[0].strip().lower()


# Upstream publishes no endpoint metadata (/models returns only id/object/created/
# owned_by), so this was measured by replaying the v2 fingerprint against the
# live endpoint. It is a hint, not a contract: call_upstream also retries the
# other chat endpoints when upstream answers ModelProtocolUnsupported, so a
# model moving between them self-heals without a code change.
OPENCODE_SYSTEMONE_PREFIXES = ("jev",)
OPENCODE_RESPONSES_HINTS = ("muse-spark",)


def opencode_model_endpoint(model: str) -> str:
    """Which /inference/v1 route a model answers on: chat | responses | systemone."""
    base = opencode_base_model(model)
    if base.startswith(OPENCODE_SYSTEMONE_PREFIXES):
        # System One decision models take {state, questions} and return
        # probabilities, not chat messages. Not proxyable through /v1/chat.
        return "systemone"
    if any(h in base for h in OPENCODE_RESPONSES_HINTS):
        return "responses"
    return "chat"


def opencode_endpoint_candidates(model: str) -> list[str]:
    """Routes to try, the hinted one first, without duplicates."""
    first = opencode_model_endpoint(model)
    if first == "systemone":
        return ["systemone"]
    rest = [e for e in ("chat", "responses", "messages") if e != first]
    return [first] + rest


def opencode_route_path(endpoint: str) -> str:
    return {"chat": "chat/completions", "responses": "responses",
            "messages": "messages", "systemone": "systemone"}[endpoint]


def is_model_protocol_unsupported(resp) -> bool:
    """True for upstream's "wrong endpoint" signal (measured: 400 + this type)."""
    if getattr(resp, "status_code", 0) != 400:
        return False
    try:
        body = resp.json()
    except (ValueError, TypeError):
        return False
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        return False
    blob = json.dumps(err).lower()
    return "modelprotocolunsupported" in blob or "does not support this protocol" in blob


def _opencode_session_id() -> str:
    """ses_<12 hex of ~(ms*0x1000+1)><14 base62>.

    Upstream decodes the prefix as an inverted timestamp and rejects a large
    counter, so the counter is pinned to 1 exactly like the CLI.
    """
    ms = int(time.time() * 1000)
    prefix = (~(ms * 0x1000 + 1)) & _OC_PREFIX_MASK
    tail = "".join(_BASE62[secrets.randbelow(256) % 62] for _ in range(14))
    return "ses_%012x%s" % (prefix, tail)


def opencode_session(entry: dict) -> str:
    """One stable session per connection.

    Upstream meters the free tier per session; minting a fresh id per request
    burns the quota and surfaces as 429 FreeUsageLimitError.
    """
    ident = str(entry.get("id") or entry.get("key") or "default")
    now = time.time()
    hit = _OC_SESSIONS.get(ident)
    if hit and now - hit[1] < _OC_SESSION_TTL:
        _OC_SESSIONS[ident] = (hit[0], now)
        return hit[0]
    sid = _opencode_session_id()
    if len(_OC_SESSIONS) > 500:
        _OC_SESSIONS.pop(next(iter(_OC_SESSIONS)), None)
    _OC_SESSIONS[ident] = (sid, now)
    return sid


def opencode_v2_headers(ses: str) -> dict:
    """v2 identity. x-opencode-request is deliberately absent (upstream 403s on it)."""
    return {
        "Authorization": "Bearer public",
        "User-Agent": OPENCODE_V2_UA,
        "x-opencode-client": OPENCODE_V2_CLIENT,
        "x-opencode-session": ses,
        "x-opencode-session-id": ses,
        "x-session-affinity": ses,
        "x-session-id": ses,
        "x-opencode-project": OPENCODE_V2_PROJECT,
    }


def _oc_tool_name(tool: dict) -> str:
    if not isinstance(tool, dict):
        return ""
    if isinstance(tool.get("name"), str) and tool["name"].strip():
        return tool["name"].strip()
    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"].strip()
    return ""


def _oc_tool_def(tool: dict) -> dict:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return fn if isinstance(fn, dict) else (tool if isinstance(tool, dict) else {})


def _oc_set_tool_name(tool: dict, name: str) -> dict:
    if isinstance(tool.get("function"), dict):
        return {**tool, "function": {**tool["function"], "name": name}}
    return {**tool, "name": name}


def apply_opencode_v2_tools(body: dict) -> dict:
    """Force stream, alias clashing caller tools, append the fingerprint pair.

    Returns {sent_name: original_name} for caller tools we had to rename.
    """
    if not isinstance(body, dict):
        return {}
    # The free lane rejects non-streaming requests outright.
    body["stream"] = True
    # Upstream answers tool_choice "none" with 403, so never forward that.
    if body.get("tool_choice") == "none":
        body.pop("tool_choice", None)

    tools = body.get("tools")
    if not isinstance(tools, list):
        tools = []
    tools = [t for t in tools if isinstance(t, dict)]

    taken = {_oc_tool_name(t) for t in tools}
    rename: dict[str, str] = {}
    kept_fp: set[str] = set()
    out: list[dict] = []
    for tool in tools:
        name = _oc_tool_name(tool)
        if name not in OPENCODE_FINGERPRINT_TOOLS:
            out.append(tool)
            continue
        if name in kept_fp:
            continue  # duplicate declarations of one name are rejected
        want = OPENCODE_V2_TOOL_DEFS[name]
        got = _oc_tool_def(tool)
        if got.get("description") == want["description"] and \
                got.get("parameters") == want["parameters"]:
            kept_fp.add(name)
            out.append(tool)
            continue
        # Caller ships its own `read`/`shell`: move it aside so the verbatim
        # fingerprint definition can still be appended under the real name.
        alias = name + "_client"
        i = 2
        while alias in taken:
            alias = "%s_client%d" % (name, i)
            i += 1
        taken.add(alias)
        rename[alias] = name
        out.append(_oc_set_tool_name(tool, alias))

    appended: list[str] = []
    for name in OPENCODE_FINGERPRINT_TOOLS:
        if name in kept_fp:
            continue
        d = OPENCODE_V2_TOOL_DEFS[name]
        out.append({"type": "function",
                    "function": {"name": name, "description": d["description"],
                                 "parameters": d["parameters"]}})
        appended.append(name)
    body["tools"] = out

    if appended:
        # Without the old "do not use" wording the model actually calls these;
        # "not available" made it also refuse the caller's own tools.
        listed = " or ".join("`%s`" % n for n in appended)
        note = ("Do not call the %s tools; they are placeholders that do nothing. "
                "Use the other tools you have been given instead." % listed)
        msgs = body.get("messages")
        if isinstance(msgs, list):
            if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system" \
                    and isinstance(msgs[0].get("content"), str):
                msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + note}
            else:
                body["messages"] = [{"role": "system", "content": note}] + msgs
        elif isinstance(body.get("system"), str):
            body["system"] = body["system"] + "\n\n" + note
    return rename


def _oc_restore_choice(choice, rename: dict):
    if not isinstance(rename, dict) or not isinstance(choice, dict):
        return choice
    if isinstance(choice.get("name"), str):
        original = rename.get(choice["name"].strip())
        return {**choice, "name": original} if original else choice
    fn = choice.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        original = rename.get(fn["name"].strip())
        if original:
            return {**choice, "function": {**fn, "name": original}}
    return choice


def restore_opencode_tool_names(payload, rename: dict):
    """Put the caller's original tool spellings back in a response/SSE payload."""
    if not rename or not isinstance(payload, dict):
        return payload
    changed = False
    out = dict(payload)

    choices = out.get("choices")
    if isinstance(choices, list):
        new_choices = []
        for choice in choices:
            if not isinstance(choice, dict):
                new_choices.append(choice)
                continue
            new_choice = dict(choice)
            for holder in ("delta", "message"):
                msg = new_choice.get(holder)
                if not isinstance(msg, dict):
                    continue
                calls = msg.get("tool_calls")
                if not isinstance(calls, list) or not calls:
                    continue
                new_calls = []
                for call in calls:
                    fn = call.get("function") if isinstance(call, dict) else None
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str) \
                            and fn["name"] in rename:
                        new_calls.append({**call, "function": {**fn, "name": rename[fn["name"]]}})
                        changed = True
                    else:
                        new_calls.append(call)
                new_choice[holder] = {**msg, "tool_calls": new_calls}
            new_choices.append(new_choice)
        if changed:
            out["choices"] = new_choices

    content = out.get("content")
    if isinstance(content, list):
        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" \
                    and isinstance(block.get("name"), str) and block["name"] in rename:
                new_content.append({**block, "name": rename[block["name"]]})
                changed = True
            else:
                new_content.append(block)
        if changed:
            out["content"] = new_content

    output = out.get("output")
    if isinstance(output, list):
        new_output = []
        for item in output:
            if isinstance(item, dict) and item.get("type") == "function_call" \
                    and isinstance(item.get("name"), str) and item["name"] in rename:
                new_output.append({**item, "name": rename[item["name"]]})
                changed = True
            else:
                new_output.append(item)
        if changed:
            out["output"] = new_output

    inner = out.get("item")
    if isinstance(inner, dict) and inner.get("type") == "function_call" \
            and isinstance(inner.get("name"), str) and inner["name"] in rename:
        out["item"] = {**inner, "name": rename[inner["name"]]}
        changed = True

    block = out.get("content_block")
    if isinstance(block, dict) and block.get("type") == "tool_use" \
            and isinstance(block.get("name"), str) and block["name"] in rename:
        out["content_block"] = {**block, "name": rename[block["name"]]}
        changed = True

    if "tool_choice" in out and out["tool_choice"] != "none":
        restored = _oc_restore_choice(out["tool_choice"], rename)
        if restored is not out["tool_choice"]:
            out["tool_choice"] = restored

    return out if changed else payload


def aggregate_openai_sse(resp, router_model: str, rename: dict | None = None) -> dict:
    """Collapse an upstream SSE chat.completion stream into one JSON response."""
    content: list[str] = []
    reasoning: list[str] = []
    finish = "stop"
    usage: dict = {}
    created = int(time.time())
    resp_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    model = router_model
    tool_calls: dict[int, dict] = {}
    order: list[int] = []
    whole: dict = {}

    def absorb(ev: dict) -> None:
        nonlocal finish, created, model, usage, resp_id
        if isinstance(ev.get("id"), str) and ev["id"]:
            resp_id = ev["id"]
        if isinstance(ev.get("model"), str) and ev["model"]:
            model = ev["model"]
        if isinstance(ev.get("created"), (int, float)):
            created = int(ev["created"])
        if isinstance(ev.get("usage"), dict) and ev["usage"]:
            usage = ev["usage"]
        for choice in (ev.get("choices") or []):
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                delta = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning.append(delta["reasoning_content"])
            for call in (delta.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                idx = call.get("index")
                idx = idx if isinstance(idx, int) else (order[-1] if order else 0)
                slot = tool_calls.get(idx)
                if slot is None:
                    slot = {"index": idx, "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""}}
                    tool_calls[idx] = slot
                    order.append(idx)
                if isinstance(call.get("id"), str) and call["id"]:
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if isinstance(fn, dict):
                    if isinstance(fn.get("name"), str) and fn["name"]:
                        slot["function"]["name"] += fn["name"]
                    if isinstance(fn.get("arguments"), str):
                        slot["function"]["arguments"] += fn["arguments"]

    iter_lines = getattr(resp, "iter_lines", None)
    if not callable(iter_lines):
        # no streaming body available on this response object
        try:
            j = resp.json()
        except (ValueError, TypeError, AttributeError):
            j = {}
        if isinstance(j, dict):
            j = dict(j)
            j["model"] = router_model
            return restore_opencode_tool_names(j, rename) if rename else j

    for raw in resp.iter_lines(decode_unicode=False):
        if not raw:
            continue
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        line = line.strip()
        if not line.startswith("data:"):
            # upstream ignored stream:true and sent a plain JSON body
            if line.startswith("{"):
                try:
                    ev = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(ev, dict):
                    if ev.get("choices") and not any(
                            isinstance(c, dict) and "delta" in c
                            for c in ev["choices"]):
                        whole = ev  # already a finished chat.completion
                    else:
                        absorb(ev)
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except (ValueError, TypeError):
            continue
        if isinstance(ev, dict):
            absorb(ev)

    if whole and not content and not tool_calls:
        whole = dict(whole)
        whole["model"] = router_model
        return restore_opencode_tool_names(whole, rename) if rename else whole

    message: dict = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        if not message["content"]:
            message["content"] = None

    out = {
        "id": resp_id, "object": "chat.completion", "created": created,
        "model": router_model, "choices": [
            {"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if rename:
        out = restore_opencode_tool_names(out, rename)
    return out


def opencode_device_start(server: str = OPENCODE_CONSOLE) -> dict:
    """Start an OpenCode Console device-code login. Returns the JSON response."""
    resp = requests.post(
        f"{server.rstrip('/')}/auth/device/code",
        headers=_opencode_console_headers(),
        json={"client_id": OPENCODE_CLIENT_ID, "supports_org_scope": True},
        timeout=30)
    resp.raise_for_status()
    j = resp.json()
    return j if isinstance(j, dict) else {}


def opencode_device_poll(device_code: str, server: str = OPENCODE_CONSOLE):
    """Poll the device-code token endpoint. Returns the raw requests.Response."""
    return requests.post(
        f"{server.rstrip('/')}/auth/device/token",
        headers=_opencode_console_headers(),
        json={"grant_type": OPENCODE_DEVICE_GRANT,
              "device_code": device_code, "client_id": OPENCODE_CLIENT_ID},
        timeout=30)


def opencode_fetch_catalog(proxy: dict | None = None) -> list[dict]:
    """Fetch the live model catalog from OpenCode and annotate each entry.

    Upstream returns only id/object/created/owned_by, so the fields the UI
    needs (lane, endpoint, chat-compatibility) are derived here rather than
    trusted from the response.
    """
    out: list[dict] = []
    seen: set[str] = set()
    bases = [(OPENCODE_INFERENCE_BASE, True), (OPENCODE_BASE, False)]
    for base, free_lane in bases:
        try:
            resp = requests.get(f"{base}/models", headers={"Authorization": "Bearer public"},
                                timeout=20, proxies=proxy)
        except requests.RequestException:
            continue
        if resp.status_code != 200:
            continue
        try:
            payload = resp.json()
        except (ValueError, TypeError):
            continue
        rows = payload.get("data") if isinstance(payload, dict) else payload
        for row in (rows or []):
            mid = str((row or {}).get("id") or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            endpoint = opencode_model_endpoint(mid)
            out.append({
                "id": mid,
                "free": opencode_is_free_model(mid),
                "endpoint": endpoint,
                "lane": "free" if opencode_is_free_model(mid) else "paid",
                "chat_compatible": endpoint != "systemone",
                "owned_by": (row or {}).get("owned_by") or "",
            })
        if out:  # one good catalog is enough
            break
    out.sort(key=lambda m: (not m["free"], m["id"]))
    return out


def opencode_console_get(path: str, token: str, server: str = OPENCODE_CONSOLE):
    """Authenticated GET against the OpenCode Console API."""
    return requests.get(
        f"{server.rstrip('/')}{path}",
        headers=_opencode_console_headers(token),
        timeout=30)


def opencode_account_metadata(tokens: dict | None = None, user: dict | None = None,
                              orgs: list | None = None) -> dict:
    """Build connection metadata from token/user/org responses (defensive to shapes)."""
    tokens = tokens or {}
    user = user or {}
    orgs = orgs or []
    org = orgs[0] if orgs and isinstance(orgs[0], dict) else {}

    def _pick(d: dict, *keys):
        for k in keys:
            v = d.get(k)
            if v:
                return str(v)
        return ""

    return {
        "account_id": _pick(tokens, "account_id", "accountID")
        or _pick(user, "id", "accountID", "account_id"),
        "email": _pick(user, "email", "primary_email") or _pick(tokens, "email"),
        "org_id": _pick(tokens, "org_id", "orgID") or _pick(org, "id", "orgID"),
        "org_name": _pick(org, "name", "slug") or _pick(tokens, "org_name"),
    }


def opencode_needs_refresh(entry: dict) -> bool:
    """True when an oauth OpenCode connection's access token is (nearly) expired."""
    if not isinstance(entry, dict) or entry.get("auth_type") != "oauth":
        return False
    if not entry.get("key"):
        return False
    try:
        exp = int(entry.get("expires_at", 0) or 0)
    except (TypeError, ValueError):
        exp = 0
    if exp <= 0:
        return False
    return exp - int(time.time()) <= 60


def opencode_refresh_access(entry: dict, proxy: dict | None = None,
                            server: str | None = None) -> bool:
    """Refresh an oauth connection's access token in place. Returns True on success."""
    server = (server or entry.get("console_server") or OPENCODE_CONSOLE).rstrip("/")
    refresh_token = str(entry.get("refresh_token") or "")
    if not refresh_token:
        return False
    try:
        resp = requests.post(
            f"{server}/auth/device/token",
            headers=_opencode_console_headers(),
            json={"grant_type": "refresh_token", "refresh_token": refresh_token,
                  "client_id": OPENCODE_CLIENT_ID},
            timeout=30, proxies=proxy)
    except requests.RequestException:
        return False
    if resp.status_code != 200:
        return False
    try:
        j = resp.json()
    except (ValueError, TypeError):
        return False
    access = (j or {}).get("access_token")
    if not access:
        return False
    entry["key"] = access
    if j.get("refresh_token"):
        entry["refresh_token"] = j["refresh_token"]
    try:
        expires_in = int(j.get("expires_in", 3600) or 3600)
    except (TypeError, ValueError):
        expires_in = 3600
    entry["expires_at"] = int(time.time()) + expires_in
    return True


def _close_quietly(resp) -> None:
    try:
        resp.close()
    except Exception:
        pass


class _RefusalResponse:
    """Synthetic non-200 response for a request we decline to send upstream.

    Lets call_upstream refuse centrally instead of every handler duplicating the
    check, so all three client dialects get the same clear message.
    """

    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return iter(())

    def iter_content(self, chunk_size=None):
        return iter(())


def call_upstream(protocol: str, body: dict, gemini_model: str = "",
                  stream: bool = False, timeout=60,
                  extra_headers: dict | None = None, provider: str = "",
                  oc_state: dict | None = None):
    """POST to upstream; returns (entry, response).

    Keys speaking the same protocol are tried round-robin; on 401/403/429
    or a connection error the next key is tried. On success (200) the order
    is advanced. A non-200 response is still returned (the handler decides);
    if no attempt can be made at all, an exception is raised.

    `provider` pins the pool to one provider (the model mapping's provider) so
    a request for a commandcode model never lands on another openai-protocol key.
    `oc_state`, when a dict is passed, receives OpenCode v2 free-lane details
    ({rename, session, forced_stream}) so the handler can aggregate/restore.
    """
    with _lock:
        data = load_data()
        pool = _ordered_pool(data, protocol, provider)
        proxies_for: dict[str, str | None] = {}
        for entry in pool:
            prov = entry.get("provider") or ""
            if prov not in proxies_for:
                proxies_for[prov] = _pick_proxy(data, prov)
        # OpenCode Console connections (device-code OAuth): refresh expired tokens
        for entry in pool:
            if opencode_needs_refresh(entry):
                opencode_refresh_access(
                    entry, _proxy_dict(proxies_for.get(entry.get("provider") or "")))
        save_data(data)  # persist proxy rotation cursors + refreshed tokens
    last_exc: Exception | None = None
    for i, entry in enumerate(pool):
        base = effective_base(entry)
        key = entry.get("key", "")
        entry_provider = entry.get("provider") or ""
        proxy = _proxy_dict(proxies_for.get(entry_provider))
        merged_extra = {**(extra_headers or {}),
                        **provider_custom_headers(data, entry_provider)}

        # OpenCode v2: keyless free tier lives on /inference/v1 behind the v2
        # fingerprint. Paid/keyed Zen stays on /zen/v1 untouched.
        oc_endpoint = ""
        if entry_provider == "opencode" and opencode_is_free_model(body.get("model")):
            oc_endpoint = opencode_model_endpoint(body.get("model"))
            if oc_endpoint == "systemone":
                # A System One decision model takes {state, questions} and returns
                # probabilities. Sending it a chat body would only produce an
                # opaque upstream 400, so refuse with something actionable.
                return entry, _RefusalResponse(400, {"error": {
                    "message": ("'%s' is a System One decision model. It is served on "
                                "/inference/v1/systemone and takes {state, questions}, so it "
                                "cannot be proxied through /v1/chat/completions."
                                % body.get("model")),
                    "type": "invalid_request_error", "code": "systemone_model"}})
            base = OPENCODE_INFERENCE_BASE
            ses = opencode_session(entry)
            merged_extra = {**opencode_v2_headers(ses),
                            **{k: v for k, v in merged_extra.items() if v}}
            # upstream insists on stream:true; the handler aggregates if needed
            rename = apply_opencode_v2_tools(body)
            oc_endpoint = opencode_model_endpoint(body.get("model"))
            if oc_state is not None:
                oc_state.update({"rename": rename, "session": ses, "forced_stream": True,
                                 "endpoint": oc_endpoint})
            merged_extra.setdefault("Accept", "text/event-stream")

        def _post(url: str):
            return requests.post(
                url, headers=upstream_headers(protocol, key, merged_extra),
                json=body, timeout=timeout, stream=stream, proxies=proxy)

        try:
            if protocol == "openai":
                note_request_sent(entry)
                if oc_endpoint and oc_endpoint != "chat":
                    resp = _post(f"{base}/{opencode_route_path(oc_endpoint)}")
                    # upstream publishes no endpoint map, so on its explicit
                    # "wrong endpoint" signal walk the remaining chat routes.
                    for alt in opencode_endpoint_candidates(body.get("model")):
                        if alt == oc_endpoint or alt == "systemone" or resp.status_code != 400:
                            continue
                        if not is_model_protocol_unsupported(resp):
                            break
                        _close_quietly(resp)
                        oc_endpoint = alt
                        resp = _post(f"{base}/{opencode_route_path(alt)}")
                else:
                    resp = _post(openai_chat_url(base))
            elif protocol == "anthropic":
                note_request_sent(entry)
                resp = _post(anthropic_messages_url(base))
            else:
                url = gemini_generate_url(base, gemini_model) + f"?key={key}"
                kwargs: dict = {"json": body, "timeout": timeout}
                if merged_extra:
                    kwargs["headers"] = upstream_headers("gemini", key, merged_extra)
                if proxy:
                    kwargs["proxies"] = proxy
                note_request_sent(entry)
                resp = requests.post(url, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            continue
        note_upstream_result(entry, resp)
        if resp.status_code == 200:
            _advance_provider_index(protocol, len(pool), provider)
            return entry, resp
        if resp.status_code == 429:
            # rate limited: park this key so later requests skip it. honour an
            # explicit Retry-After when upstream sent one (OpenCode's daily free
            # quota resets at 00:00 UTC and reports ~19h, far past our 5 min).
            secs = mark_key_cooling(entry, retry_after_seconds(resp))
            note_rate_limited(entry, secs)
            if i < len(pool) - 1:
                _close_quietly(resp)
                last_exc = RuntimeError(f"upstream 429: {resp.text[:300]}")
                continue
            # pool held only usable keys, so parking this one empties it. Raise so
            # the handler can replay its cached response instead of surfacing a
            # bare 429 to the caller.
            _close_quietly(resp)
            raise AllKeysCoolingDown(secs, len(pool))
        if resp.status_code in (401, 403) and i < len(pool) - 1:
            _close_quietly(resp)
            last_exc = RuntimeError(f"upstream {resp.status_code}: {resp.text[:300]}")
            continue
        return entry, resp
    raise last_exc or RuntimeError("upstream call failed.")


def call_gemini(gemini_model: str, payload: dict, timeout: int = 60,
                provider: str = "") -> dict:
    """Legacy signature kept: JSON call with the gemini protocol, returns a dict."""
    _entry, resp = call_upstream("gemini", payload, gemini_model=gemini_model,
                                 stream=False, timeout=timeout, provider=provider)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _rewrite_sse_line(line: bytes, router_model: str, rename: dict | None = None) -> bytes:
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
                if rename:
                    j = restore_opencode_tool_names(j, rename)
                return ("data: " + json.dumps(j, ensure_ascii=False)).encode("utf-8")
    return line


def rewrite_sse_model(chunks, router_model: str, rename: dict | None = None):
    """Replace the model name in the upstream SSE stream with the router name (pure generator)."""
    buf = b""
    for chunk in chunks or []:
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8", "replace")
        buf += chunk or b""
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield _rewrite_sse_line(line, router_model, rename) + b"\n"
    if buf:
        yield _rewrite_sse_line(buf, router_model)


def proxy_sse_response(upstream_resp, router_model: str, rename: dict | None = None) -> Response:
    """Relay the upstream SSE stream to the client as-is (model name fixed)."""
    def gen():
        try:
            yield from rewrite_sse_model(
                upstream_resp.iter_content(chunk_size=4096), router_model, rename)
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
        `require_api_key` in settings decides whether a key is *required*; a
        matching token is always *accepted* so attribution survives the toggle
        being off. With the setting absent, the historical rule applies: auth
        is on as soon as any key exists.
        """
        token = extract_bearer_token(request.headers)
        data = load_data()
        router_keys = data.get("router_keys", []) or []
        if ROUTER_API_KEY and token == ROUTER_API_KEY:
            return True, None, {"id": "env", "label": "env:ROUTER_API_KEY"}
        for rk in router_keys:
            if token and token == rk.get("key"):
                return True, None, {"id": rk.get("id", ""), "label": rk.get("label", "")}
        if router_auth_enabled(data):
            return False, (jsonify({"error": "unauthorized, Bearer token or x-api-key required"}), 401), {"id": "", "label": ""}
        return True, None, {"id": "", "label": ""}

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

    @app.get("/admin/providers/<provider_id>")
    def admin_provider_page(provider_id: str):
        return render_template("provider.html", provider_id=provider_id)

    def _masked_provider_key(k: dict) -> dict:
        return {"id": k.get("id", ""), "label": k.get("label", ""),
                "provider": k.get("provider", "gemini"),
                "protocol": effective_protocol(k),
                "base_url": k.get("base_url", ""),
                "effective_base": effective_base(k),
                "priority": int(k.get("priority", 1) or 1),
                "proxy_pool": k.get("proxy_pool", "none") or "none",
                "auth_type": k.get("auth_type", "key"),
                "account": k.get("account", {}) or {},
                "expires_at": k.get("expires_at", 0) or 0,
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

    # ------------------------------------------------------------------
    # Router auth switch
    # ------------------------------------------------------------------
    @app.get("/admin/api/router-auth")
    def admin_get_router_auth():
        if (err := need_admin_auth()) is not None:
            return err
        data = load_data()
        return jsonify({
            "require_api_key": router_auth_enabled(data),
            "explicit": "require_api_key" in (data.get("settings") or {}),
            "key_count": len(data.get("router_keys", []) or []),
            "env_key": bool(ROUTER_API_KEY),
        })

    @app.post("/admin/api/router-auth")
    def admin_set_router_auth():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        want = bool(body.get("require_api_key"))
        data = load_data()
        # Turning this on with nothing to authenticate against is the one
        # irreversible mistake available in this panel, so it is refused.
        if want and not (data.get("router_keys") or []) and not ROUTER_API_KEY:
            return jsonify({
                "error": "no API keys exist — create one before requiring it, "
                         "otherwise every client is locked out"
            }), 409
        with _lock:
            data = load_data()
            data.setdefault("settings", {})["require_api_key"] = want
            save_data(data)
        return jsonify({"ok": True, "require_api_key": want,
                        "router_auth_required": router_auth_enabled(data)})

    # ------------------------------------------------------------------
    # Usage & analytics
    # ------------------------------------------------------------------
    @app.get("/admin/api/usage")
    def admin_usage():
        if (err := need_admin_auth()) is not None:
            return err
        data = load_data()
        rng = (request.args.get("range") or "7d").strip().lower()
        if rng not in USAGE_RANGES:
            rng = "7d"
        return jsonify(build_usage(data, rng))

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
        try:
            priority = int(body.get("priority", 1) or 1)
        except (TypeError, ValueError):
            priority = 1
        proxy_pool = str(body.get("proxy_pool", "none") or "none").strip() or "none"
        with _lock:
            data = load_data()
            if any(k.get("key") == key for k in data.get("provider_keys", [])):
                return jsonify({"error": "this key is already added"}), 400
            data["provider_keys"].append({
                "id": uuid.uuid4().hex[:8], "key": key, "label": label,
                "provider": provider, "protocol": protocol, "base_url": base_url,
                "priority": priority, "proxy_pool": proxy_pool,
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

    def _probe_provider_key(protocol: str, base: str, key: str,
                            extra_headers: dict | None = None,
                            proxy: dict | None = None) -> dict:
        """Lightweight validity probe: GET the upstream model list (never raises)."""
        base = (base or "").rstrip("/")
        if not base:
            return {"ok": False, "status": 0, "message": "base URL is empty", "models": None}
        extra = extra_headers or {}
        try:
            if protocol == "anthropic":
                headers = {"anthropic-version": ANTHROPIC_VERSION}
                if key:
                    headers["x-api-key"] = key
                headers.update(extra)
                resp = requests.get(base + "/models", headers=headers, timeout=15,
                                    proxies=proxy)
            elif protocol == "gemini":
                kwargs: dict = {"timeout": 15}
                if extra:
                    kwargs["headers"] = extra
                if proxy:
                    kwargs["proxies"] = proxy
                resp = requests.get(base + f"/v1beta/models?key={key}", **kwargs)
            else:
                headers = {"Authorization": f"Bearer {key}"} if key else {}
                headers.update(extra)
                resp = requests.get(base + "/models", headers=headers, timeout=15,
                                    proxies=proxy)
        except requests.RequestException as exc:
            return {"ok": False, "status": 0,
                    "message": f"connection error: {exc}"[:300], "models": None}
        ok = resp.status_code == 200
        count = None
        if ok:
            try:
                j = resp.json()
                data = j.get("data") if isinstance(j, dict) else None
                if isinstance(data, list):
                    count = len(data)
            except (ValueError, TypeError):
                count = None
        message = "key is valid" if ok else _cross_error_message(protocol, resp)
        return {"ok": ok, "status": resp.status_code, "message": message, "models": count}

    @app.post("/admin/api/provider-keys/check")
    def admin_check_provider_key():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        key = str(body.get("key", "")).strip()
        provider = str(body.get("provider", "opencode")).strip().lower() or "opencode"
        try:
            protocol, base_url = normalize_provider_key(
                provider, body.get("protocol"), body.get("base_url"))
        except ValueError as exc:
            return jsonify({"ok": False, "status": 0, "message": str(exc)}), 400
        base = base_url or PROVIDER_DEFAULTS.get(provider, {}).get("base_url", "")
        with _lock:
            data = load_data()
            proxy = _proxy_dict(_pick_proxy(data, provider))
            save_data(data)
        return jsonify(_probe_provider_key(protocol, base, key,
                                           provider_custom_headers(data, provider), proxy))

    @app.post("/admin/api/provider-keys/test-all")
    def admin_test_all_provider_keys():
        if (err := need_admin_auth()) is not None:
            return err
        with _lock:
            data = load_data()
            proxies: dict[str, dict | None] = {}
            for k in data.get("provider_keys", []):
                prov = (k.get("provider") or "") if isinstance(k, dict) else ""
                if prov not in proxies:
                    proxies[prov] = _proxy_dict(_pick_proxy(data, prov))
            save_data(data)
        results = []
        for k in data.get("provider_keys", []):
            if not isinstance(k, dict):
                continue
            prov = k.get("provider") or ""
            probe = _probe_provider_key(effective_protocol(k), effective_base(k),
                                        k.get("key", ""),
                                        provider_custom_headers(data, prov),
                                        proxies.get(prov))
            results.append({"id": k.get("id", ""), "label": k.get("label", ""),
                            "ok": probe.get("ok", False),
                            "status": probe.get("status", 0),
                            "message": probe.get("message", "")})
        return jsonify({"results": results})

    # --- Provider detail page API (OpenCode connections + settings + models) ---

    def _proxy_pool_public(p: dict) -> dict:
        return {"id": p.get("id", ""), "name": p.get("name", ""),
                "proxies": [str(x) for x in (p.get("proxies") or [])]}

    def _provider_detail(provider_id: str) -> dict:
        pid = (provider_id or "").strip().lower()
        data = load_data()
        keys = [k for k in data.get("provider_keys", [])
                if isinstance(k, dict) and (k.get("provider") or "") == pid]
        models = [m for m in data.get("models", [])
                  if isinstance(m, dict) and (m.get("provider") or "") == pid]
        have = {str(m.get("upstream_model") or "") for m in models}
        suggestions = [s for s in PROVIDER_SUGGESTIONS.get(pid, []) if s not in have]
        return {
            "id": pid,
            "name": provider_display_name(pid),
            "short": provider_short(pid),
            "protocol": route_protocol_for_provider(pid),
            "base_url": PROVIDER_DEFAULTS.get(pid, {}).get("base_url", ""),
            "auth_required": pid not in ("opencode",),
            "connected": bool(keys),
            "count": len(keys),
            "settings": get_provider_settings(data, pid),
            "connections": [_masked_provider_key(k) for k in keys],
            "cooling": cooling_key_report(pid),
            "usage": key_usage_report(pid),
            "models": [{"name": m.get("name", ""),
                        "upstream_model": str(m.get("upstream_model") or m.get("gemini_model") or ""),
                        "description": m.get("description", ""),
                        "disabled": bool(m.get("disabled"))} for m in models],
            "suggestions": suggestions,
            "pools": [_proxy_pool_public(p) for p in data.get("proxy_pools", [])
                      if isinstance(p, dict)],
        }

    @app.get("/admin/api/providers/<provider_id>")
    def admin_provider_detail(provider_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        return jsonify(_provider_detail(provider_id))

    @app.post("/admin/api/providers/<provider_id>/settings")
    def admin_provider_settings(provider_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        pid = (provider_id or "").strip().lower()
        if pid not in PROVIDERS:
            return jsonify({"error": f"unknown provider: {pid}"}), 400
        body = request.get_json(force=True, silent=True) or {}
        with _lock:
            data = load_data()
            ps = data.setdefault("provider_settings", {}).setdefault(pid, {})
            if "proxy_pool" in body:
                ps["proxy_pool"] = str(body.get("proxy_pool") or "none")
            if "rotation" in body:
                rot = str(body.get("rotation") or "none").lower()
                if rot not in ("none", "round_robin", "random"):
                    return jsonify({"error": "rotation must be none|round_robin|random"}), 400
                ps["rotation"] = rot
            if "thinking" in body:
                th = str(body.get("thinking") or "auto").lower()
                if th not in ("auto", "on", "off"):
                    return jsonify({"error": "thinking must be auto|on|off"}), 400
                ps["thinking"] = th
            if "headers_active" in body:
                ps["headers_active"] = bool(body.get("headers_active"))
            if "custom_headers" in body:
                clean = []
                for h in body.get("custom_headers") or []:
                    if isinstance(h, dict) and str(h.get("name", "")).strip():
                        clean.append({"name": str(h["name"]).strip(),
                                      "value": str(h.get("value", "")),
                                      "enabled": bool(h.get("enabled", True))})
                ps["custom_headers"] = clean
            save_data(data)
        return jsonify({"ok": True, "settings": get_provider_settings(data, pid)})

    # --- OpenCode Console account connection (device-code OAuth) ---

    def _opencode_finish_connect(tokens: dict):
        access = str(tokens.get("access_token") or "")
        refresh_token = str(tokens.get("refresh_token") or "")
        try:
            expires_in = int(tokens.get("expires_in", 3600) or 3600)
        except (TypeError, ValueError):
            expires_in = 3600
        user: dict = {}
        orgs: list = []
        try:
            ur = opencode_console_get("/api/user", access)
            if ur.status_code == 200:
                uj = ur.json()
                user = uj if isinstance(uj, dict) else {}
        except (requests.RequestException, ValueError, TypeError):
            user = {}
        try:
            orr = opencode_console_get("/api/orgs", access)
            if orr.status_code == 200:
                oj = orr.json()
                if isinstance(oj, dict):
                    oj = oj.get("orgs") or oj.get("data") or []
                orgs = oj if isinstance(oj, list) else []
        except (requests.RequestException, ValueError, TypeError):
            orgs = []
        meta = opencode_account_metadata(tokens, user, orgs)
        label = meta.get("email") or meta.get("org_name") or meta.get("account_id") \
            or "OpenCode account"
        with _lock:
            data = load_data()
            acct_id = meta.get("account_id") or ""
            if acct_id:  # replace an existing connection for the same account
                data["provider_keys"] = [
                    k for k in data.get("provider_keys", [])
                    if not (isinstance(k, dict) and k.get("auth_type") == "oauth"
                            and (k.get("account") or {}).get("account_id") == acct_id)]
            entry = {
                "id": uuid.uuid4().hex[:8], "key": access, "label": label,
                "provider": "opencode", "protocol": "openai", "base_url": "",
                "priority": 1, "proxy_pool": "none", "auth_type": "oauth",
                "refresh_token": refresh_token,
                "expires_at": int(time.time()) + expires_in,
                "console_server": OPENCODE_CONSOLE, "account": meta,
                "created_at": _now_iso(),
            }
            data.setdefault("provider_keys", []).append(entry)
            save_data(data)
        return jsonify({"status": "complete",
                        "connection": _masked_provider_key(entry)})

    @app.post("/admin/api/providers/<provider_id>/opencode/connect/start")
    def admin_opencode_connect_start(provider_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        if (provider_id or "").strip().lower() != "opencode":
            return jsonify({"error": "only the opencode provider supports account login"}), 400
        try:
            j = opencode_device_start()
        except requests.RequestException as exc:
            return jsonify({"error": f"connection error: {exc}"[:300]}), 502
        except (ValueError, TypeError):
            return jsonify({"error": "invalid response from OpenCode console"}), 502
        if not j.get("device_code"):
            return jsonify({"error": "OpenCode console did not return a device code"}), 502
        return jsonify({
            "device_code": str(j.get("device_code", "")),
            "user_code": str(j.get("user_code", "")),
            "verification_uri": str(j.get("verification_uri", "")),
            "verification_uri_complete": str(
                j.get("verification_uri_complete") or j.get("verification_uri") or ""),
            "interval": int(j.get("interval", 5) or 5),
            "expires_in": int(j.get("expires_in", 900) or 900),
        })

    @app.post("/admin/api/providers/<provider_id>/opencode/connect/poll")
    def admin_opencode_connect_poll(provider_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        if (provider_id or "").strip().lower() != "opencode":
            return jsonify({"error": "only the opencode provider supports account login"}), 400
        body = request.get_json(force=True, silent=True) or {}
        device_code = str(body.get("device_code", "")).strip()
        if not device_code:
            return jsonify({"error": "device_code required"}), 400
        try:
            resp = opencode_device_poll(device_code)
        except requests.RequestException as exc:
            return jsonify({"status": "error",
                            "message": f"connection error: {exc}"[:300]}), 502
        try:
            j = resp.json()
        except (ValueError, TypeError):
            j = {}
        if isinstance(j, dict) and j.get("access_token"):
            return _opencode_finish_connect(j)
        err_code = ""
        if isinstance(j, dict):
            err_code = str(j.get("error", "") or "")
            if not err_code and j.get("message"):
                err_code = str(j["message"])
        if err_code == "authorization_pending":
            status = "pending"
        elif err_code == "slow_down":
            status = "slow_down"
        elif err_code in ("expired_token", "access_denied"):
            status = "expired"
        else:
            status = "error"
        return jsonify({"status": status,
                        "message": err_code or f"http {resp.status_code}"})

    @app.get("/admin/api/proxy-pools")
    def admin_list_proxy_pools():
        if (err := need_admin_auth()) is not None:
            return err
        data = load_data()
        return jsonify([_proxy_pool_public(p) for p in data.get("proxy_pools", [])
                        if isinstance(p, dict)])

    @app.get("/admin/api/combos")
    def admin_list_combos():
        if (err := need_admin_auth()) is not None:
            return err
        data = load_data()
        now = int(time.time())
        combos = [_combo_public(data, c, now) for c in list_combos(data)]
        return jsonify({"combos": combos,
                        "pickable": [m["name"] for m in data.get("models", [])
                                     if isinstance(m, dict) and m.get("name")
                                     and not m.get("disabled")]})

    @app.post("/admin/api/combos")
    def admin_save_combo():
        """Create or update a combo. A combo aliases exactly one existing model."""
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        name = str(body.get("name", "")).strip()
        model = str(body.get("model", "")).strip()
        desc = str(body.get("description", "")).strip()
        if not name:
            return jsonify({"error": "combo name must not be empty"}), 400
        if not COMBO_NAME_RE.match(name):
            return jsonify({"error": "combo name may only contain letters, numbers, "
                                     "'.', '_' and '-' (max 64)"}), 400
        if not model:
            return jsonify({"error": "combo must point at one model"}), 400
        with _lock:
            data = load_data()
            models = {m["name"] for m in data.get("models", [])
                      if isinstance(m, dict) and m.get("name")}
            if model not in models:
                return jsonify({"error": f"unknown model: {model}"}), 400
            if find_combo(data, model):
                return jsonify({"error": f"'{model}' is already used as a combo name"}), 400
            if any(m.get("name") == name for m in data.get("models", [])
                   if isinstance(m, dict)):
                return jsonify({"error": f"'{name}' is already a model name"}), 400
            combos = [c for c in (data.get("combos") or []) if isinstance(c, dict)]
            existing = next((c for c in combos
                             if str(c.get("name") or "").strip() == name), None)
            if existing is None and len(combos) >= 50:
                return jsonify({"error": "too many combos (max 50)"}), 400
            if existing:
                existing.update({"model": model, "description": desc})
            else:
                combos.append({"id": uuid.uuid4().hex[:8], "name": name,
                               "model": model, "description": desc,
                               "created_at": _now_iso()})
            data["combos"] = combos
            save_data(data)
        return jsonify({"ok": True, "name": name}), 201

    @app.delete("/admin/api/combos/<name>")
    def admin_delete_combo(name: str):
        if (err := need_admin_auth()) is not None:
            return err
        with _lock:
            data = load_data()
            combos = list_combos(data)
            if not any(str(c.get("name")) == name for c in combos):
                return jsonify({"error": f"unknown combo: {name}"}), 404
            data["combos"] = [c for c in combos if str(c.get("name")) != name]
            save_data(data)
        return jsonify({"ok": True})

    @app.post("/admin/api/proxy-pools")
    def admin_add_proxy_pool():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        name = str(body.get("name", "")).strip() or "pool"
        raw = body.get("proxies")
        if isinstance(raw, str):
            raw = raw.splitlines()
        proxies = [str(x).strip() for x in (raw or []) if str(x).strip()]
        with _lock:
            data = load_data()
            entry = {"id": uuid.uuid4().hex[:8], "name": name, "proxies": proxies,
                     "created_at": _now_iso()}
            data.setdefault("proxy_pools", []).append(entry)
            save_data(data)
        return jsonify({"ok": True, "id": entry["id"], "pool": _proxy_pool_public(entry)}), 201

    @app.delete("/admin/api/proxy-pools/<pool_id>")
    def admin_del_proxy_pool(pool_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        with _lock:
            data = load_data()
            data["proxy_pools"] = [p for p in data.get("proxy_pools", [])
                                   if p.get("id") != pool_id]
            save_data(data)
        return jsonify({"ok": True})

    @app.post("/admin/api/providers/<provider_id>/models/fetch")
    def admin_fetch_provider_models(provider_id: str):
        """Pull the live model catalog for a provider (OpenCode today)."""
        if (err := need_admin_auth()) is not None:
            return err
        pid = (provider_id or "").strip().lower()
        if pid not in PROVIDERS:
            return jsonify({"error": f"unknown provider: {pid}"}), 400
        if pid != "opencode":
            return jsonify({"error": f"catalog fetch is not implemented for {pid}",
                            "hint": "add models manually from the Models tab"}), 400
        with _lock:
            data = load_data()
            proxy = _proxy_dict(_pick_proxy(data, pid))
            save_data(data)
        try:
            models = opencode_fetch_catalog(proxy)
        except Exception as exc:  # noqa: BLE001 - upstream can be anything
            return jsonify({"error": f"catalog fetch failed: {exc}"[:300]}), 502
        if not models:
            return jsonify({"error": "upstream returned no models",
                            "hint": "check connectivity or proxy settings"}), 502
        have = {str(m.get("upstream_model") or "") for m in data.get("models", [])
                if isinstance(m, dict) and (m.get("provider") or "") == pid}
        for m in models:
            m["added"] = m["id"] in have
        return jsonify({"models": models,
                        "free": sum(1 for m in models if m["free"]),
                        "chat": sum(1 for m in models if m["chat_compatible"])})

    @app.post("/admin/api/providers/<provider_id>/models")
    def admin_provider_add_model(provider_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        pid = (provider_id or "").strip().lower()
        if pid not in PROVIDERS:
            return jsonify({"error": f"unknown provider: {pid}"}), 400
        body = request.get_json(force=True, silent=True) or {}
        upstream = str(body.get("upstream_model", "")).strip()
        if not upstream:
            return jsonify({"error": "upstream_model required"}), 400
        name = str(body.get("name", "")).strip() or f"{provider_short(pid)}/{upstream}"
        with _lock:
            data = load_data()
            data["models"] = [m for m in data.get("models", []) if m.get("name") != name]
            data["models"].append({"name": name, "provider": pid,
                                   "upstream_model": upstream,
                                   "description": str(body.get("description", "")).strip(),
                                   "disabled": False})
            save_data(data)
        return jsonify({"ok": True, "name": name}), 201

    @app.delete("/admin/api/providers/<provider_id>/models/<path:name>")
    def admin_provider_del_model(provider_id: str, name: str):
        if (err := need_admin_auth()) is not None:
            return err
        with _lock:
            data = load_data()
            data["models"] = [m for m in data.get("models", []) if m.get("name") != name]
            save_data(data)
        return jsonify({"ok": True})

    @app.post("/admin/api/providers/<provider_id>/models/disable-all")
    def admin_provider_disable_all(provider_id: str):
        if (err := need_admin_auth()) is not None:
            return err
        pid = (provider_id or "").strip().lower()
        with _lock:
            data = load_data()
            own = [m for m in data.get("models", [])
                   if isinstance(m, dict) and (m.get("provider") or "") == pid]
            any_enabled = any(not m.get("disabled") for m in own)
            for m in own:
                m["disabled"] = any_enabled
            save_data(data)
        return jsonify({"ok": True, "disabled": any_enabled})

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
        if provider not in PROVIDERS:
            return jsonify({"error": f"provider must be one of: {', '.join(PROVIDERS)}"}), 400
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
            if "round_robin" in body:
                data["settings"]["round_robin"] = bool(body.get("round_robin"))
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
        out = {"id": m["name"], "object": "model", "type": "model",
               "owned_by": "samet-router", "created": now,
               "display_name": m["name"], "meta": meta}
        if m.get("combo"):
            out["combo"] = True
            out["combo_model"] = m.get("combo_model") or ""
            out["description"] = m.get("description") or ""
        return out

    def _combo_public(data: dict, c: dict, now: int) -> dict:
        target = resolve_target(data, str(c.get("model") or ""))
        return _model_public({
            "name": c.get("name"),
            # report the provider id (like a normal model entry), not the wire protocol
            "provider": target.get("provider_id") or target["provider"],
            "upstream_model": target["upstream_model"],
            "combo": True,
            "combo_model": c.get("model") or "",
            "description": c.get("description") or "",
        }, now)

    # --- Router: OpenAI + Anthropic compatible model list ---
    @app.get("/v1/models")
    def list_models():
        ok, err, _key = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        data = load_data()
        now = int(time.time())
        rows = [_model_public(m, now) for m in data.get("models", [])
                if isinstance(m, dict) and m.get("name") and not m.get("disabled")]
        rows += [_combo_public(data, c, now) for c in list_combos(data)]
        return jsonify({"object": "list", "data": rows})

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
        combo = find_combo(data, model_id)
        if combo:
            return jsonify(_combo_public(data, combo, now))
        # unknown model (e.g. claude-*) still returns 200: Claude Code
        # checks model existence, falls back to the fallback target.
        target = resolve_target(data, model_id)
        return jsonify(_model_public({"name": model_id, "provider": target["provider"],
                                      "upstream_model": target["upstream_model"]},
                                     now, fallback=True))

    # --- Router: OpenAI compatible ---
    def _chat_core(body: dict, key_info: dict, ep: str, forward_beta: dict | None = None):
        forward_beta = forward_beta or {}
        router_model = str(body.get("model", "")).strip() or "samet-flash"
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        t0 = time.time()
        thinking = body_has_thinking("openai", body)
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
                _tmode = get_provider_settings(
                    data, target.get("provider_id") or "").get("thinking")
                apply_thinking_default(up_body, "openai", _tmode)
                thinking = ensure_openai_reasoning_replay(
                    up_body, upstream, target.get("provider_id") or "", _tmode) or thinking
                oc: dict = {}
                _entry, resp = call_upstream(
                    "openai", up_body, stream=stream or bool(oc.get("forced_stream")),
                    timeout=STREAM_TIMEOUT if stream else 60,
                    provider=target.get("provider_id") or "", oc_state=oc)
                # OpenCode free lane always streams upstream; aggregate for
                # clients that asked for a single JSON response.
                oc_streamed = bool(oc.get("forced_stream")) and not stream
                if resp.status_code == 400 and not stream and not oc_streamed:
                    retry_body, removed = sanitize_openai_body_for_retry(up_body)
                    if removed:
                        _close_quietly(resp)
                        _entry2, resp2 = call_upstream(
                            "openai", retry_body, stream=False, timeout=60,
                            provider=target.get("provider_id") or "", oc_state=oc)
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
                rename = oc.get("rename") or {}
                if stream:
                    add_log(ep, router_model, True, int((time.time() - t0) * 1000),
                            "", 0, 0, kid, klabel, provider, thinking)
                    return proxy_sse_response(resp, router_model, rename)
                if oc_streamed:
                    try:
                        j = aggregate_openai_sse(resp, router_model, rename)
                    except Exception as exc:
                        _close_quietly(resp)
                        add_log(ep, router_model, False, int((time.time() - t0) * 1000),
                                f"upstream stream error: {exc}"[:300],
                                0, 0, kid, klabel, provider, thinking)
                        return jsonify({"error": {"message": f"upstream stream error: {exc}",
                                                  "type": "server_error"}}), 502
                    u = j.get("usage", {}) or {}
                    thinking = thinking or bool(
                        ((j.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content"))
                    add_log(ep, router_model, True, int((time.time() - t0) * 1000), "",
                            int(u.get("prompt_tokens", 0) or 0),
                            int(u.get("completion_tokens", 0) or 0),
                            kid, klabel, provider, thinking)
                    cache_store_response(target.get("provider_id") or "", upstream, j)
                    return jsonify(j)
                j = resp.json()
                j["model"] = router_model
                u = j.get("usage", {}) or {}
                thinking = thinking or bool(
                    ((j.get("choices") or [{}])[0].get("message") or {}).get("reasoning_content"))
                add_log(ep, router_model, True, int((time.time() - t0) * 1000), "",
                        int(u.get("prompt_tokens", 0) or 0),
                        int(u.get("completion_tokens", 0) or 0),
                        kid, klabel, provider, thinking)
                if not stream:
                    cache_store_response(target.get("provider_id") or "", upstream, j)
                return jsonify(j)

            if provider == "anthropic":
                abody, thinking_req = openai_to_anthropic_body(body, default_system)
                thinking = thinking or thinking_req
                abody["model"] = upstream
                apply_thinking_default(
                    abody, "anthropic",
                    get_provider_settings(data, target.get("provider_id") or "").get("thinking"))
                # upstream Anthropic SSE cannot be converted to OpenAI SSE; buffer and
                # fake-stream it (correct but delayed)
                _entry, resp = call_upstream(
                    "anthropic", abody, stream=False, timeout=60,
                    extra_headers=forward_beta,
                    provider=target.get("provider_id") or "")
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
            _tmode = get_provider_settings(data, target.get("provider_id") or "").get("thinking")
            if _tmode == "on" and budget is None:
                budget = 4096
                thinking = True
            elif _tmode == "off":
                budget = None
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
            g = call_gemini(gemini_model, payload,
                             provider=target.get("provider_id") or "")
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
        except AllKeysCoolingDown as cooling:
            # every key is parked by a rate limit: serve the last good answer if
            # we have one, otherwise an honest 429 carrying Retry-After.
            stale = cache_get_response(target.get("provider_id") or "", upstream) is not None
            add_log(ep, router_model, stale,
                    int((time.time() - t0) * 1000),
                    "" if stale else str(cooling),
                    0, 0, kid, klabel, provider, thinking)
            return served_from_cache(target.get("provider_id") or "", upstream,
                                     router_model, cooling)
        except Exception as exc:  # noqa: BLE001 - error is returned to the client in simplified form
            add_log(ep, router_model, False,
                    int((time.time() - t0) * 1000), str(exc),
                    0, 0, kid, klabel, "", thinking)
            return jsonify({"error": str(exc)[:500]}), 502

    @app.post("/v1/chat/completions")
    def chat_completions():
        ok, err, key_info = need_router_auth()
        if not ok:
            return err  # type: ignore[return-value]
        body = request.get_json(force=True, silent=True) or {}
        return _chat_core(body, key_info, "openai:/v1/chat/completions",
                          _forward_anthropic_beta())

    @app.post("/admin/api/playground/chat")
    def admin_playground_chat():
        if (err := need_admin_auth()) is not None:
            return err
        body = request.get_json(force=True, silent=True) or {}
        return _chat_core(body, {"id": "playground", "label": "playground"},
                          "playground", {})

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
                apply_thinking_default(
                    abody, "anthropic",
                    get_provider_settings(data, target.get("provider_id") or "").get("thinking"))
                _entry, resp = call_upstream(
                    "anthropic", abody, stream=stream,
                    timeout=STREAM_TIMEOUT if stream else 60,
                    extra_headers=_forward_anthropic_beta(),
                    provider=target.get("provider_id") or "")
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
                _tmode = get_provider_settings(
                    data, target.get("provider_id") or "").get("thinking")
                apply_thinking_default(obody, "openai", _tmode)
                thinking = ensure_openai_reasoning_replay(
                    obody, upstream, target.get("provider_id") or "", _tmode) or thinking
                # upstream OpenAI SSE cannot be converted to Anthropic SSE; buffer it
                oc: dict = {}
                _entry, resp = call_upstream("openai", obody, stream=False, timeout=60,
                                        provider=target.get("provider_id") or "",
                                        oc_state=oc)
                oc_streamed = bool(oc.get("forced_stream"))
                if resp.status_code == 400 and not oc_streamed:
                    # Strict gateways (opencode zen, Ollama, vLLM) reject
                    # reasoning_effort / temperature / max_tokens variants with
                    # a generic 400. Retry once with a sanitized payload.
                    retry_body, removed = sanitize_openai_body_for_retry(obody)
                    if removed:
                        _close_quietly(resp)
                        _entry2, resp2 = call_upstream(
                            "openai", retry_body, stream=False, timeout=60,
                            provider=target.get("provider_id") or "")
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
                oai_json = (aggregate_openai_sse(resp, router_model, oc.get("rename"))
                            if oc_streamed else resp.json())
                anth = openai_response_to_anthropic(oai_json, router_model)
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
            _tmode = get_provider_settings(data, target.get("provider_id") or "").get("thinking")
            if _tmode == "on" and not thinking_on:
                thinking_on, budget = True, 4096
                thinking = True
            elif _tmode == "off":
                thinking_on, budget = False, None
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
            g = call_gemini(gemini_model, payload,
                             provider=target.get("provider_id") or "")
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
                apply_thinking_default(
                    body, "gemini",
                    get_provider_settings(data, target.get("provider_id") or "").get("thinking"))
                g = call_gemini(upstream, body or {"contents": []},
                                 provider=target.get("provider_id") or "")
                inp, out = _gemini_usage_of_response(g)
                add_log(ep, model_name, True, int((time.time() - t0) * 1000), "",
                        inp, out, kid, klabel, provider, thinking)
                return jsonify(g)
            # gemini protocol bound to an openai/anthropic provider:
            # first reduce to OpenAI format, then send to the target
            obody = gemini_request_to_openai(body or {"contents": []})
            obody["model"] = upstream
            if provider == "openai":
                _entry, resp = call_upstream("openai", obody, stream=False, timeout=60,
                                        provider=target.get("provider_id") or "")
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
                    extra_headers=_forward_anthropic_beta(),
                    provider=target.get("provider_id") or "")
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
    port = int(os.environ.get("PORT", "20003"))
    app.run(host="0.0.0.0", port=port, debug=True)
