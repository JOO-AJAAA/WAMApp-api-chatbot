# WAMApp Chatbot API

> FastAPI backend for weather-aware AI chat, retrieval context, and notification-aware responses.

## Repository

- **Backend Repo URL:** `https://github.com/<your-username>/<your-backend-repo>`

## Overview

This service provides the chatbot API used by the WAMApp mobile client. It handles token issuance, request validation, context building (weather + history + notifications), and LLM fallback orchestration.

## Features

- Chat access token endpoint (`/api/chat/token`)
- Chat response endpoint (`/api/chat`)
- JWT verification and device binding
- Rate limiting per device/IP
- Retrieval-Augmented Generation (RAG) via Supabase RPC
- Multi-model fallback (Gemini → Llama → Qwen)
- Notification context integration for better chat responses

## Tech Stack

- Python
- FastAPI + Uvicorn
- Supabase Python Client
- Hugging Face Hub
- Google GenAI (Gemini)
- PyJWT

## Project Structure

```text
WAMApp-api-chatbot/
├─ lib/
├─ main.py
├─ requirements.txt
└─ Procfile
```

## Setup

### Prerequisites

- Python 3.10+
- `pip`

### Install

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run (Local)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

Create `.env` in this folder:

```env
SUPABASE_URL=
SUPABASE_KEY=

EXPO_SUPABASE_URL=
EXPO_SUPABASE_ANON_KEY=

JWT_SECRET=
JWT_EXPIRE_SECONDS=600
WAMAPP_CLIENT_KEY=

CHAT_RATE_LIMIT_REQUESTS=25
CHAT_RATE_LIMIT_WINDOW_SECONDS=60

HF_TOKEN=
GEMINI_API_KEY=
```

## API Summary

| Method | Endpoint          | Description                           |
|-------:|-------------------|---------------------------------------|
| POST   | `/api/chat/token` | Issue temporary chat access token     |
| POST   | `/api/chat`       | Generate AI reply with contextual RAG |

## Request Example

```json
{
  "message": "Will it rain this evening?",
  "device_id": "your-device-id",
  "weather": {
    "locationName": "Jakarta",
    "temperatureC": 30.5,
    "weatherCode": 61
  },
  "timezone_name": "Asia/Jakarta",
  "utc_offset_minutes": 420
}
```

## Deployment Note

`Procfile` uses:

```txt
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

## License

Add your project license here.
