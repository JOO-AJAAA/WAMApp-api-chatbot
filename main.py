import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from huggingface_hub import InferenceClient
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# --- SETUP SUPABASE ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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

@app.post("/api/chat")
async def chat_rag(request: ChatRequest):
    try:
        user_text = request.message
        
        # 1. EMBEDDING (Hugging Face E5)
        query_text = f"query: {user_text}"
        hasil_vektor = hf_client.feature_extraction(
            query_text,
            model="intfloat/multilingual-e5-large"
        )
        embedding_vektor = hasil_vektor.tolist()

        # 2. RETRIEVAL (Supabase pgvector)
        response = supabase.rpc("match_documents", {
            "query_embedding": embedding_vektor,
            "match_threshold": 0.65,
            "match_count": 4
        }).execute()

        dokumen_ditemukan = response.data
        konteks_gabungan = "\n\n".join([doc.get("content", "") for doc in dokumen_ditemukan])
        
        if not konteks_gabungan.strip():
            konteks_gabungan = "Tidak ada informasi terkait di dalam database."

        # 3. GENERATION DENGAN FALLBACK (Gemini -> Llama -> Qwen)
        jawaban_akhir = ""

        # Pesan sistem universal untuk Llama dan Qwen
        system_message = {
            "role": "system", 
            "content": "Kamu adalah WAMchat, asisten AI pintar. Jawab pertanyaan HANYA berdasarkan konteks yang diberikan. Jika jawaban tidak ada di konteks, katakan kamu tidak tahu dengan sopan. Jawabannya jangan ada kata seperti 'berdasarkan informasi yang saya miliki' atau 'menurut data yang tersedia'. Tapi jangan bilang kamu tidak tahu jika tidak ada di konteks, jawab saja dengan informasi yang tersedia tapi beri sedikit ralat dan klarifikasinya."
        }
        user_message = {
            "role": "user", 
            "content": f"Konteks Database:\n{konteks_gabungan}\n\nPertanyaan User: {user_text}"
        }

        # --- PERCOBAAN 1: GEMINI (Model Utama) ---
        try:
            print("Mencoba generate jawaban dengan Gemini...")
            if not gemini_client:
                raise Exception("Kunci API Gemini tidak tersedia")

            gemini_config = types.GenerateContentConfig(
                system_instruction="Kamu adalah WAMchat, asisten AI pintar. Jawab pertanyaan HANYA berdasarkan konteks yang diberikan. Jika jawaban tidak ada di konteks, katakan kamu tidak tahu dengan sopan. Jawabannya jangan ada kata seperti 'berdasarkan informasi yang saya miliki' atau 'menurut data yang tersedia'. Tapi jangan bilang kamu tidak tahu jika tidak ada di konteks, jawab saja dengan informasi yang tersedia tapi beri sedikit ralat dan klarifikasinya.",
                temperature=0.3,
            )
            
            prompt = f"Konteks Database:\n{konteks_gabungan}\n\nPertanyaan User: {user_text}"
            
            respons_gemini = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=gemini_config
            )
            jawaban_akhir = respons_gemini.text
            print("Berhasil menggunakan Gemini!")

        except Exception as e_gemini:
            print(f"Gemini gagal: {e_gemini}. Beralih ke cadangan 1 (Llama 3.1)...")
            
            # --- PERCOBAAN 2: LLAMA 3.1 (via HF Router) ---
            try:
                if not hf_openai_client:
                    raise Exception("Client HF Router tidak tersedia")

                respons_llama = hf_openai_client.chat.completions.create(
                    model="meta-llama/Llama-3.1-8B-Instruct:novita",
                    messages=[system_message, user_message],
                    temperature=0.3
                )
                jawaban_akhir = respons_llama.choices[0].message.content
                print("Berhasil menggunakan Llama 3.1!")

            except Exception as e_llama:
                print(f"Llama gagal: {e_llama}. Beralih ke cadangan 2 (Qwen 2.5)...")
                
                # --- PERCOBAAN 3: QWEN 2.5 (via HF Router) ---
                try:
                    respons_qwen = hf_openai_client.chat.completions.create(
                        model="Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
                        messages=[system_message, user_message],
                        temperature=0.3
                    )
                    jawaban_akhir = respons_qwen.choices[0].message.content
                    print("Berhasil menggunakan Qwen 2.5!")
                    
                except Exception as e_qwen:
                    print(f"Qwen juga gagal: {e_qwen}.")
                    jawaban_akhir = "Maaf, semua sistem AI kami sedang mengalami gangguan. Mohon coba beberapa saat lagi."

        return {"reply": jawaban_akhir, "context_used": dokumen_ditemukan}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))