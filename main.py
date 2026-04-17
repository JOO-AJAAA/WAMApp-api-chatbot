import os
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from supabase import create_client, Client
from huggingface_hub import InferenceClient

from typing import Optional, Any, Dict, List
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

from lib.chat_prompt import build_llm_context
from lib.chat_security import (
    create_chat_access_token,
    enforce_rate_limit,
    get_client_ip,
    verify_chat_access_token,    
)
from lib.getNotification import DeviceIdUnavailableError, NotificationFetchError, get_notifications_for_device

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

# --- SETUP SUPABASE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY is not configured in environment")

backend_supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    timezone_name: Optional[str] = None
    utc_offset_minutes: Optional[int] = None


class ChatTokenRequest(BaseModel):
    device_id: str


@app.post("/api/chat/token")
async def issue_chat_token(
    request: ChatTokenRequest,
    x_wamapp_client_key: Optional[str] = Header(default=None),
):
    if not WAMAPP_CLIENT_KEY:
        raise HTTPException(status_code=500, detail="WAMAPP_CLIENT_KEY belum dikonfigurasi di server.")

    if not x_wamapp_client_key or x_wamapp_client_key != WAMAPP_CLIENT_KEY:
        raise HTTPException(status_code=403, detail="Akses ditolak.")

    device_id = (request.device_id or "").strip()
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id wajib diisi.")

    token = create_chat_access_token(
        device_id,
        JWT_SECRET,
        JWT_EXPIRE_SECONDS,
        JWT_ISSUER,
        JWT_AUDIENCE,
        JWT_ALGORITHM,
    )
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
        device_id = (request.device_id or "").strip()
        if not device_id:
            raise HTTPException(status_code=400, detail="device_id wajib diisi.")

        token_payload = verify_chat_access_token(
            authorization,
            JWT_SECRET,
            JWT_ISSUER,
            JWT_AUDIENCE,
            JWT_ALGORITHM,
        )
        token_device_id = str(token_payload.get("sub", "")).strip()
        if not token_device_id or token_device_id != device_id:
            raise HTTPException(status_code=403, detail="Token tidak cocok dengan device_id.")

        client_ip = get_client_ip(http_request)
        enforce_rate_limit(device_id=device_id, ip=client_ip, max_requests=CHAT_RATE_LIMIT_REQUESTS, window_seconds=CHAT_RATE_LIMIT_WINDOW_SECONDS)
        
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
        
        # --- MENGAMBIL RIWAYAT CHAT SEBELUMNYA ---
        history_response = backend_supabase.table("chat_history") \
            .select("role, content") \
            .eq("device_id", device_id) \
            .order("created_at", desc=True) \
            .limit(10) \
            .execute()
        
        history_data = history_response.data[::-1] # Balik agar urut dari terlama ke terbaru
        try:
            notifications = get_notifications_for_device(device_id=device_id, limit=10)
        except DeviceIdUnavailableError as notif_device_err:
            raise HTTPException(
                status_code=400,
                detail="Gagal mengambil context notifikasi: device_id tidak tersedia/invalid.",
            ) from notif_device_err
        except NotificationFetchError as notif_fetch_err:
            print(f"Warning: notifikasi gagal diambil, lanjut tanpa context notifikasi: {notif_fetch_err}")
            notifications = []
        print("notification\n", notifications)
                
        context_bundle = build_llm_context(
            user_text=user_text,
            weather=getattr(request, "weather", None),
            system_instructions=getattr(request, "system_instructions", None),
            assistant_instructions=getattr(request, "assistant_instructions", None),
            notifications_for_llm=notifications,
            history_data=history_data,
            dokumen_ditemukan=dokumen_ditemukan,
            timezone_name=getattr(request, "timezone_name", None),
            utc_offset_minutes=getattr(request, "utc_offset_minutes", None),
        )

        combined_system_instruction = context_bundle["combined_system_instruction"]
        user_content_lengkap = context_bundle["user_content_lengkap"]

        jawaban_akhir = ""
        system_message = context_bundle["system_message"]
        user_message = context_bundle["user_message"]

        # --- PERCOBAAN 1: GEMINI ---
        try:
            print(f"Mencoba Gemini dengan persona: {context_bundle['kategori_utama'].upper()}...")
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

        if isinstance(e, HTTPException):
            raise e

        raise HTTPException(status_code=500, detail=str(e))