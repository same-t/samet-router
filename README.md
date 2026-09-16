# ◈ Samet Router

Multi-provider, **OpenAI / Anthropic / Gemini compatible** mini LLM router + admin panel with sidebar.

```
Client (OpenAI SDK / Anthropic SDK / ham REST)
   → Samet Router (Flask, bu repo)
   → Upstream: OpenAI / Anthropic / Gemini / OpenAI-uyumlu custom sunucu
```

For each **provider key**, configure the provider (whose key it is), the protocol (which format is spoken:
`openai` | `anthropic` | `gemini`) and the `base URL` (where requests are
sent). Each **model alias** is linked to a provider; the request
is converted to the target format and routed to the matching pool.
The Dashboard shows total/successful/failed requests, input/output/total tokens,
and model/key/endpoint/provider/daily breakdowns.

## Quick start

```powershell
pip install -r requirements.txt
python app.py
# http://localhost:8000/admin
```

1. In `/admin` → **Provider Keys** tab, add an upstream key:
   - `provider=openai` → OpenAI key (`sk-...`), protocol `openai`
   - `provider=anthropic` → Anthropic key (`sk-ant-...`), protocol `anthropic`
   - `provider=gemini` → Google AI Studio key (`AIza...`), protocol `gemini`
   - `provider=custom` + `base URL` → OpenAI-compatible servers such as
     Ollama / vLLM / DeepSeek / LM Studio (e.g. `http://localhost:11434/v1`)
2. In the **Models** tab, map a router name → provider + upstream model
   (e.g. `gpt-mini` → `openai`/`gpt-4o-mini`, `claude-alias` → `anthropic`/`claude-sonnet-4-5`).
3. In the **User Keys** tab, generate a separate key per user (`sk-samet-...`).
   The raw key is shown only at creation time — copy it immediately.
4. Use that key in the client (`Authorization: Bearer ...` or `x-api-key: ...`).

## Endpoints

| Endpoint | Compatibility |
|---|---|
| `GET /health` | service status |
| `GET /v1/models` | OpenAI + Anthropic `list models` (dual format) |
| `GET /v1/models/<id>` | single model (returns fallback 200 for unknown `claude-*` id) |
| `POST /v1/chat/completions` | OpenAI chat (including tools, `tool_choice`, `stop`, `stream:SSE`) |
| `POST /v1/messages` | Anthropic messages (including tools, `tool_use`/`tool_result`, images, `stop_sequences`, `stream:SSE`) |
| `POST /v1/messages/count_tokens` | Anthropic token counting (estimated) |
| `POST /v1beta/models/<model>:generateContent` | Gemini passthrough |
| `POST /v1beta/models/<model>:streamGenerateContent` | Gemini stream fallback (returns JSON) |
| `GET /admin` | admin panel with sidebar (Dashboard, User Keys, Provider Keys, Models, Prompt, Recent Requests, Docs) |
| `GET/POST/DELETE /admin/api/...` | admin JSON API (including `/settings`, `/router-keys`, `/stats`) |
| `GET /admin/api/providers` | provider catalog: protocol/base_url/auth defaults |
| `POST /admin/api/provider-keys` | add upstream key (`{key, label, provider, protocol?, base_url?}`; `base_url` required for `custom`) |
| `DELETE /admin/api/provider-keys/<id>` | delete upstream key |
| `GET /admin/api/stats` | dashboard summary: total/ok/err, success %, input/output/total tokens, avg. latency, thinking count, model/key/endpoint/provider/daily breakdown |
| `DELETE /admin/api/stats` | reset statistics + logs |
| `POST /admin/api/router-keys` | generates a user key, returns the raw key **once** (`{"label": "..."}`) |
| `DELETE /admin/api/router-keys/<id>` | revoke key (this key now returns 401) |

## Examples

```python
# OpenAI compatible (api_key = the sk-samet-... key generated in the panel)
# "gpt-mini" is mapped to openai/gpt-4o-mini under Models
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-samet-...")
r = c.chat.completions.create(model="gpt-mini",
    messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content, r.usage)
```

```python
# Anthropic compatible (the format Claude Code uses)
# "claude-alias" is mapped to anthropic/claude-sonnet-4-5 under Models
import anthropic
c = anthropic.Anthropic(base_url="http://localhost:8000", api_key="x")
m = c.messages.create(model="claude-alias", max_tokens=200,
    system="Write briefly",
    messages=[{"role": "user", "content": "Hello"}],
    tools=[{"name": "get_weather", "description": "weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}])
print(m.content)
```

### Claude Code connection

```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:8000"
$env:ANTHROPIC_AUTH_TOKEN = "sk-samet-..."   # key generated in the panel → User Keys
$env:ANTHROPIC_MODEL = "claude-alias"        # alias bound to anthropic under Models in the panel
claude
```

> If an unknown model name arrives, it falls back to the first model in the list. Since Claude Code
> sends `claude-*`, add an alias via the shortcut in the Prompt tab
> (provider can be selected: `anthropic`, `openai` or `gemini`).

### Thinking (reasoning) support

```python
# Extended thinking: passed through to the Anthropic provider untouched
m = c.messages.create(model="claude-alias", max_tokens=2000,
    thinking={"type": "enabled", "budget_tokens": 1000},
    messages=[{"role": "user", "content": "Think deeply"}])
```

| Direction | Behavior |
|---|---|
| Anthropic thinking → Anthropic provider | Direct passthrough (`thinking` parameter, signed `thinking`, `redacted_thinking`, `anthropic-beta` header) |
| Anthropic thinking → OpenAI provider | Budget is mapped to a `reasoning_effort` heuristic; `thinking` blocks cannot be carried in OpenAI input and are dropped |
| Anthropic thinking → Gemini provider | Budget becomes `thinkingConfig.thinkingBudget`, `includeThoughts` is enabled; response `thought` parts become a `thinking` block if thinking was requested |
| OpenAI `reasoning_effort` → Anthropic provider | Converted to a budget and becomes the `thinking` parameter |
| OpenAI `reasoning_content` → Anthropic | Becomes an unsigned `thinking` block (converted to text if sent back to Anthropic upstream, preventing a 400) |
| Gemini `thought` → OpenAI/Anthropic | Becomes `reasoning_content` / `thinking`; dropped if thinking was not requested (hidden reasoning never leaks to the user) |

```powershell
# Gemini compatible (raw REST)
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/v1beta/models/samet-flash:generateContent" `
  -ContentType "application/json" `
  -Body '{"contents":[{"role":"user","parts":[{"text":"Hello"}]}]}'
```

## Environment variables

| Variable | Description |
|---|---|
| `PORT` | default `8000` |
| `SAMET_ROUTER_DATA` | JSON data file path (default `./data.json`) |
| `ROUTER_API_KEY` | optional single server key. If set **or** a user key exists in the panel, auth is required: `Authorization: Bearer ...` **or** `x-api-key: ...` (Claude Code sends `x-api-key`). If neither exists, it is open in local mode |
| `ADMIN_TOKEN` | if set, `/admin/api/*` requires Bearer |

## Tests

```powershell
pytest -q
```

Tests do not send real requests to Gemini (`requests.post` is mocked).

## Notes

- Keys speaking the same protocol are pooled; requests are distributed round-robin, trying the next key on 401/403/429 or connection errors. If the pool is empty, it returns 502 + a "no provider key configured" error.
- User keys remain valid until revoked; a revoked key returns 401. Last usage and the request counter are on the key row; the token/request breakdown is on the Dashboard.
- Token counts come from the upstream's real counters (`usageMetadata`, `usage`); logs keep the last 1000 requests, totals are cumulative counters unaffected by rotation. In stream passthrough, tokens are unknown (logged as 0).
- `stream:true`: with the same protocol, upstream SSE is relayed verbatim (model name corrected); across protocols it is buffered, converted to the target format, then streamed.
- On passthrough paths (OpenAI→OpenAI, Anthropic→Anthropic) the upstream error code + body are forwarded as-is; on cross-protocol paths they are converted to the target format.
- Tools: converted between formats (Anthropic `tool_use` ↔ OpenAI `tool_calls` ↔ Gemini `functionCall`); `stop_reason`/`finish_reason` are mapped correctly.
- System/user: Anthropic `system` (string or block list with `cache_control`), OpenAI `system/developer` roles, `image` blocks, and the default system prompt from `/admin` → **Prompt** tab are merged before sending (`default + client`).
- `cache_control`, `metadata`, `top_p/top_k`, `stop_sequences` are either converted or silently ignored, never returning 400.
- Old `data.json` files are migrated automatically (`gemini_keys` → provider key, `gemini_model` → provider/upstream mapping).
- This is a hobby/startup prototype; add auth + rate limiting + DB for production.
