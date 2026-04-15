import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from huggingface_hub import InferenceClient

from typing import Optional, Any, Dict
import json
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
    device_id: str

    weather: Optional[Dict[str, Any]] = None
    system_instructions: Optional[str] = None
    assistant_instructions: Optional[str] = None

@app.post("/api/chat")
async def chat_rag(request: ChatRequest):
    import json # Import di dalam fungsi agar aman tanpa mengubah bagian atas file
    
    try:
        user_text = request.message
        device_id = getattr(request, 'device_id', 'unknown_device') # Pastikan aman jika kosong
        
        # 1. EMBEDDING
        query_text = f"query: {user_text}"
        hasil_vektor = hf_client.feature_extraction(
            query_text,
            model="intfloat/multilingual-e5-large"
        )
        # Menggunakan caramu yang benar untuk e5-large
        embedding_vektor = hasil_vektor.tolist()

        # 2. RETRIEVAL
        response = supabase.rpc("match_documents", {
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
        history_response = supabase.table("chat_history") \
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
            supabase.table("chat_history").insert([
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