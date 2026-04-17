import os
import time
from uuid import UUID
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from supabase import create_client, Client
from huggingface_hub import InferenceClient
import jwt

from typing import Optional, Any, Dict, List
import json
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- AUTH & RATE LIMIT CONFIG ---
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_ISSUER = "wamapp-api"
JWT_AUDIENCE = "wamapp-mobile"
JWT_EXPIRE_SECONDS = int(os.getenv("JWT_EXPIRE_SECONDS", "600"))
WAMAPP_CLIENT_KEY = os.getenv("WAMAPP_CLIENT_KEY", "")

CHAT_RATE_LIMIT_REQUESTS = int(os.getenv("CHAT_RATE_LIMIT_REQUESTS", "25"))
CHAT_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60"))
_chat_rate_bucket: Dict[str, List[float]] = {}
DEVICE_ID_VALIDATION_TABLES = [
    t.strip() for t in os.getenv("DEVICE_ID_VALIDATION_TABLES", "devices,notifications,marked_locations").split(",") if t.strip()
]
DEVICE_ID_VALIDATION_COLUMN = os.getenv("DEVICE_ID_VALIDATION_COLUMN", "device_id")

# --- SETUP SUPABASE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EXPO_SUPABASE_URL = os.getenv("EXPO_SUPABASE_URL")
EXPO_SUPABASE_KEY = os.getenv("EXPO_SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is not configured in environment")

if not EXPO_SUPABASE_URL or not EXPO_SUPABASE_KEY:
    raise RuntimeError("EXPO_SUPABASE_URL or EXPO_SUPABASE_ANON_KEY is not configured in environment")

backend_supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
expo_supabase: Client = create_client(EXPO_SUPABASE_URL, EXPO_SUPABASE_KEY)

# --- SETUP HUGGING FACE (Untuk Embedding) ---
hf_token = os.getenv("HF_TOKEN")
hf_client = InferenceClient(provider="hf-inference", api_key=hf_token)

# --- SETUP HUGGING FACE ROUTER (Untuk Llama & Qwen via OpenAI SDK) ---
if hf_token:
    hf_openai_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )
else:
    print("WARNING: HF_TOKEN belum diset!")
    hf_openai_client = None

# --- SETUP GEMINI (Utama) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY belum diset!")
    gemini_client = None

class ChatRequest(BaseModel):
    message: str
    device_id: str

    weather: Optional[Dict[str, Any]] = None
    system_instructions: Optional[str] = None
    assistant_instructions: Optional[str] = None


class ChatTokenRequest(BaseModel):
    device_id: str


def _get_client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if req.client and req.client.host:
        return req.client.host
    return "unknown_ip"


def _enforce_rate_limit(device_id: str, ip: str) -> None:
    now = time.time()
    key = f"{device_id}:{ip}"
    window_start = now - CHAT_RATE_LIMIT_WINDOW_SECONDS

    timestamps = _chat_rate_bucket.get(key, [])
    timestamps = [ts for ts in timestamps if ts > window_start]

    if len(timestamps) >= CHAT_RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Terlalu banyak request. Coba lagi sebentar.")

    timestamps.append(now)
    _chat_rate_bucket[key] = timestamps


def _normalize_device_id(device_id: str) -> str:
    cleaned = (device_id or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="device_id wajib diisi.")

    try:
        # Paksa format UUID agar tidak menerima device_id random.
        return str(UUID(cleaned))
    except ValueError:
        raise HTTPException(status_code=400, detail="Format device_id tidak valid.")


def _is_registered_device_id(device_id: str) -> bool:
    for table_name in DEVICE_ID_VALIDATION_TABLES:
        try:
            response = (
                expo_supabase.table(table_name)
                .select(DEVICE_ID_VALIDATION_COLUMN)
                .eq(DEVICE_ID_VALIDATION_COLUMN, device_id)
                .limit(1)
                .execute()
            )
            rows = response.data if hasattr(response, "data") else response.get("data", [])
            if rows:
                return True
        except Exception as exc:
            print(f"WARNING: Validasi device_id ke tabel '{table_name}' gagal: {exc}")

    return False


def _assert_registered_device_id(device_id: str) -> str:
    normalized_device_id = _normalize_device_id(device_id)
    if not _is_registered_device_id(normalized_device_id):
        raise HTTPException(status_code=403, detail="device_id tidak terdaftar.")
    return normalized_device_id


def _create_chat_access_token(device_id: str) -> str:
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET belum dikonfigurasi di server.")

    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=JWT_EXPIRE_SECONDS)

    payload = {
        "sub": device_id,
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _verify_chat_access_token(authorization: Optional[str]) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Token tidak ditemukan.")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Format Authorization harus Bearer token.")

    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token kosong.")

    try:
        decoded = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER,
        )
        return decoded
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token tidak valid.")


@app.post("/api/chat/token")
async def issue_chat_token(
    request: ChatTokenRequest,
    x_wamapp_client_key: Optional[str] = Header(default=None),
):
    if not WAMAPP_CLIENT_KEY:
        raise HTTPException(status_code=500, detail="WAMAPP_CLIENT_KEY belum dikonfigurasi di server.")

    if not x_wamapp_client_key or x_wamapp_client_key != WAMAPP_CLIENT_KEY:
        raise HTTPException(status_code=403, detail="Akses ditolak.")

    device_id = _assert_registered_device_id(request.device_id)

    token = _create_chat_access_token(device_id)
    return {
        "token": token,
        "token_type": "Bearer",
        "expires_in": JWT_EXPIRE_SECONDS,
    }

@app.post("/api/chat")
async def chat_rag(
    request: ChatRequest,
    http_request: Request,
    authorization: Optional[str] = Header(default=None),
):
    try:
        user_text = request.message
        device_id = _assert_registered_device_id(request.device_id)

        token_payload = _verify_chat_access_token(authorization)
        token_device_id = str(token_payload.get("sub", "")).strip()
        if not token_device_id or token_device_id != device_id:
            raise HTTPException(status_code=403, detail="Token tidak cocok dengan device_id.")

        client_ip = _get_client_ip(http_request)
        _enforce_rate_limit(device_id=device_id, ip=client_ip)
        
        # 1. EMBEDDING
        query_text = f"query: {user_text}"
        hasil_vektor = hf_client.feature_extraction(
            query_text,
            model="intfloat/multilingual-e5-large"
        )
        # Menggunakan caramu yang benar untuk e5-large
        embedding_vektor = hasil_vektor.tolist()

        # 2. RETRIEVAL
        response = backend_supabase.rpc("match_documents", {
            "query_embedding": embedding_vektor,
            "match_threshold": 0.65,
            "match_count": 4
        }).execute()

        dokumen_ditemukan = response.data
        
        # Menyusun konteks sekaligus mencari tahu KATEGORI UTAMA
        konteks_list = []
        kategori_utama = "umum" # Default persona
        
        if dokumen_ditemukan:
            # Mengambil kategori dari dokumen yang paling relevan (urutan pertama)
            kategori_utama = dokumen_ditemukan[0].get("kategori", "umum")
            
            for doc in dokumen_ditemukan:
                kategori_doc = doc.get("kategori", "umum")
                isi_doc = doc.get("content", "")
                konteks_list.append(f"[Kategori: {kategori_doc}]\n{isi_doc}")
            
            konteks_gabungan = "\n\n".join(konteks_list)
        else:
            konteks_gabungan = "[TIDAK ADA KONTEKS SPESIFIK DARI DATABASE UNTUK PERTANYAAN INI]"

        # --- DYNAMIC PERSONA MAPPING ---
        persona_mapping = {
            "mitigasi": "seorang ahli mitigasi iklim yang tegas, tenang, dan sangat peduli pada keselamatan. Nada bicaramu serius namun menenangkan, memberikan instruksi protektif yang jelas dan mudah diikuti.",
            "kesehatan": "seorang konsultan kesehatan cuaca yang penuh empati. Nada bicaramu sangat hangat dan peduli, fokus memberikan saran perawatan diri terbaik agar pengguna tetap sehat menghadapi cuaca.",
            "aktivitas": "seorang pemandu aktivitas dan agrikultur yang energik dan bersemangat. Nada bicaramu antusias dan memotivasi, memberikan saran yang praktis untuk kegiatan di luar ruangan.",
            "edukasi": "seorang ilmuwan cuaca yang cerdas dan sangat antusias. Nada bicaramu menginspirasi rasa ingin tahu, menjelaskan fenomena alam dengan bahasa yang menarik, layaknya sedang bercerita.",
            "umum": "asisten ahli meteorologi yang ramah, santai, dan suportif."
        }
        
        peran_spesifik = persona_mapping.get(kategori_utama, persona_mapping["edukasi"])

        instruksi_persona = f"""Kamu adalah WAMchat, kamu berada dalam aplikasi yang disebut dengan WAMApp data cuaca user berasal dari situ. Kamu adalah {peran_spesifik}

ATURAN PENTING:
1. JANGAN PERNAH menggunakan frasa kaku seperti 'berdasarkan teks', 'menurut database', 'berdasarkan informasi yang saya miliki', atau yang sejenisnya. Jawablah mengalir seolah pengetahuan itu murni dari pikiranmu.
2. Gunakan referensi [Konteks Database] yang diberikan sebagai acuan fakta untuk menjawab.
3. Jika [Konteks Database] menunjukkan TIDAK ADA KONTEKS atau pertanyaan melenceng jauh dari topik cuaca/alam, tetaplah jawab dengan sopan memakai pengetahuan umummu, TAPI berikan sedikit klarifikasi ramah dengan gaya personamu bahwa kamu asisten cuaca."""

        # Merge incoming system/assistant prompts (if provided) and include weather snapshot
        additional_system = ""
        if getattr(request, 'system_instructions', None):
            additional_system += "\n\n" + request.system_instructions
        if getattr(request, 'assistant_instructions', None):
            # assistant_instructions are treated as additional system guidance
            additional_system += "\n\n" + request.assistant_instructions

        combined_system_instruction = instruksi_persona + additional_system

        # --- MENGAMBIL RIWAYAT CHAT SEBELUMNYA ---
        history_response = backend_supabase.table("chat_history") \
            .select("role, content") \
            .eq("device_id", device_id) \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        
        history_data = history_response.data[::-1] # Balik agar urut dari terlama ke terbaru
        teks_history = ""
        if history_data:
            teks_history = "Riwayat Percakapan Sebelumnya:\n"
            for h in history_data:
                nama = "User" if h['role'] == 'user' else "WAMchat"
                teks_history += f"{nama}: {h['content']}\n"
            teks_history += "\n"

        # --- MENYUSUN KONTEN PENGGUNA ---
        weather_text = ""
        if getattr(request, 'weather', None):
            try:
                weather_text = "External Weather Snapshot:\n" + json.dumps(request.weather, ensure_ascii=False) + "\n\n"
            except Exception:
                weather_text = "External Weather Snapshot: (unserializable)\n\n"

        # Gabungkan Riwayat + Cuaca + Konteks + Pertanyaan
        user_content_lengkap = f"{teks_history}{weather_text}Konteks Database:\n{konteks_gabungan}\n\nPertanyaan User: {user_text}"

        jawaban_akhir = ""
        system_message = {"role": "system", "content": combined_system_instruction}
        user_message = {"role": "user", "content": user_content_lengkap}

        # --- PERCOBAAN 1: GEMINI ---
        try:
            print(f"Mencoba Gemini dengan persona: {kategori_utama.upper()}...")
            if not gemini_client:
                raise Exception("Kunci API Gemini tidak tersedia")

            # Konfigurasi system prompt
            gemini_config = types.GenerateContentConfig(
                system_instruction=combined_system_instruction,
                temperature=0.4, 
            )
            
            # PENTING: Pakai model gemini-2.5-flash agar stabil
            respons_gemini = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_content_lengkap, 
                config=gemini_config
            )
            jawaban_akhir = respons_gemini.text
            print("Berhasil menggunakan Gemini!")

        except Exception as e_gemini:
            print(f"Gemini gagal: {e_gemini}. Beralih ke Llama 3.1...")
            
            # --- PERCOBAAN 2: LLAMA 3.1 ---
            try:
                if not hf_openai_client:
                    raise Exception("Client HF Router tidak tersedia")

                respons_llama = hf_openai_client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct:novita",
                    messages=[system_message, user_message],
                    temperature=0.4
                )
                jawaban_akhir = respons_llama.choices[0].message.content
                print("Berhasil menggunakan Llama 3.1!")

            except Exception as e_llama:
                print(f"Llama gagal: {e_llama}. Beralih ke Qwen 2.5...")
                
                # --- PERCOBAAN 3: QWEN 2.5 ---
                try:
                    respons_qwen = hf_openai_client.chat.completions.create(
                        model="Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
                        messages=[system_message, user_message],
                        temperature=0.4
                    )
                    jawaban_akhir = respons_qwen.choices[0].message.content
                    print("Berhasil menggunakan Qwen 2.5!")
                    
                except Exception as e_qwen:
                    print(f"Qwen juga gagal: {e_qwen}.")
                    jawaban_akhir = "Waduh, sepertinya radar komunikasi saya sedang terganggu badai nih. Boleh coba kirim pesannya lagi sebentar lagi?"

        # --- SIMPAN KE DATABASE SETELAH AI MENJAWAB ---
        if jawaban_akhir and not jawaban_akhir.startswith("Waduh, sepertinya"):
            backend_supabase.table("chat_history").insert([
                {"device_id": device_id, "role": "user", "content": user_text},
                {"device_id": device_id, "role": "bot", "content": jawaban_akhir}
            ]).execute()

        return {"reply": jawaban_akhir, "context_used": dokumen_ditemukan}

    except Exception as e:
        print("--- DEBUG ERROR START ---")
        import traceback
        traceback.print_exc() 
        print(f"ERROR MESSAGE: {str(e)}")
        print("--- DEBUG ERROR END ---")
        
        raise HTTPException(status_code=500, detail=str(e))