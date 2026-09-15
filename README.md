# ◈ Samet Router

Gemini backend'li, **OpenAI / Anthropic / Gemini uyumlu** mini LLM router + sidebar'lı admin panel.

```
Client (OpenAI SDK / Anthropic SDK / ham REST)
   → Samet Router (Flask, bu repo)
   → Google Gemini API (arkadaki gerçek LLM)
```

Admin panelden Gemini API key ve model eşlemesi eklenir. İstemciler standart SDK'larla router'a istek atar.

## Hızlı başlat

```powershell
pip install -r requirements.txt
python app.py
# http://localhost:8000/admin
```

Önce `/admin` → **API Keys** sekmesinden en az 1 Gemini key ekle (`AIza...`).
Google AI Studio'dan ücretsiz key alınır: https://aistudio.google.com/apikey

## Endpoint'ler

| Endpoint | Uyumluluk |
|---|---|
| `GET /health` | servis durumu |
| `GET /v1/models` | OpenAI `list models` |
| `POST /v1/chat/completions` | OpenAI chat |
| `POST /v1/messages` | Anthropic messages |
| `POST /v1beta/models/<model>:generateContent` | Gemini passthrough |
| `GET /admin` | sidebar'lı admin panel |
| `GET/POST/DELETE /admin/api/...` | admin JSON API |

## Örnekler

```python
# OpenAI uyumlu
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
r = c.chat.completions.create(model="samet-flash",
    messages=[{"role": "user", "content": "Merhaba"}])
print(r.choices[0].message.content)
```

```python
# Anthropic uyumlu
import anthropic
c = anthropic.Anthropic(base_url="http://localhost:8000", api_key="x")
m = c.messages.create(model="samet-flash", max_tokens=200,
    messages=[{"role": "user", "content": "Merhaba"}])
print(m.content[0].text)
```

```powershell
# Gemini uyumlu (ham REST)
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/v1beta/models/samet-flash:generateContent" `
  -ContentType "application/json" `
  -Body '{"contents":[{"role":"user","parts":[{"text":"Merhaba"}]}]}'
```

## Ortam değişkenleri

| Değişken | Açıklama |
|---|---|
| `PORT` | default `8000` |
| `SAMET_ROUTER_DATA` | JSON veri dosyası yolu (default `./data.json`) |
| `ROUTER_API_KEY` | doluysa istemciler `Authorization: Bearer ...` göndermek zorunda |
| `ADMIN_TOKEN` | doluysa `/admin/api/*` Bearer ister |

## Test

```powershell
pytest -q
```

Testler Gemini'ye gerçek istek atmaz (`requests.post` mock'lanır).

## Notlar

- Key'ler round-robin ile dönülür; 401/403/429 alınırsa sonraki key denenir.
- `stream:true` henüz desteklenmiyor.
- Bu bir hobi/startup prototipidir; production için auth + rate-limit + DB ekleyin.
