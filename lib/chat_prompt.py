import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for environments without zoneinfo
    ZoneInfo = None


def format_notifications_for_llm(notifications: List[Dict[str, Any]]) -> str:
    if not notifications:
        return "Tidak ada riwayat anomali cuaca yang relevan."

    lines: List[str] = ["Riwayat anomali cuaca Terbaru:"]
    for item in notifications[:10]:
        title = item.get("title") or "Tanpa judul"
        message = item.get("message") or item.get("content") or item.get("data") or ""
        category = item.get("category") or "umum"
        created_at = item.get("created_at") or item.get("time") or "unknown time"
        lines.append(f"- [{category}] {title}: {message} ({created_at})")

    return "\n".join(lines)


def _resolve_local_datetime(
    timezone_name: Optional[str],
    utc_offset_minutes: Optional[int],
) -> Tuple[datetime, str]:
    now_utc = datetime.now(timezone.utc)

    def _format_offset(total_minutes: int) -> str:
        sign = "+" if total_minutes >= 0 else "-"
        abs_minutes = abs(total_minutes)
        hours, minutes = divmod(abs_minutes, 60)
        return f"UTC{sign}{hours:02d}:{minutes:02d}"

    if timezone_name and ZoneInfo is not None:
        try:
            local_dt = now_utc.astimezone(ZoneInfo(timezone_name))
            offset = local_dt.utcoffset() or timedelta(0)
            total_minutes = int(offset.total_seconds() // 60)
            return local_dt, f"{timezone_name} ({_format_offset(total_minutes)})"
        except Exception:
            pass

    if utc_offset_minutes is not None:
        local_dt = now_utc + timedelta(minutes=utc_offset_minutes)
        return local_dt, _format_offset(utc_offset_minutes)

    return now_utc, "UTC"


def _build_time_context(
    timezone_name: Optional[str],
    utc_offset_minutes: Optional[int],
) -> Tuple[str, datetime]:
    local_dt, timezone_label = _resolve_local_datetime(timezone_name, utc_offset_minutes)

    return (
        f"Timezone user: {timezone_label}\n"
        f"Current local time: {local_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Current date: {local_dt.strftime('%A, %d %B %Y')}",
        local_dt,
    )


def _build_time_style_instruction(local_dt: datetime) -> str:
    hour = local_dt.hour

    if 5 <= hour < 11:
        periode = "pagi"
        gaya = (
            "Gunakan tone segar, ringan, memotivasi, dan actionable untuk memulai aktivitas. "
            "Jika memberi saran, prioritaskan persiapan pagi hari dan rencana hingga siang."
        )
    elif 11 <= hour < 15:
        periode = "siang"
        gaya = (
            "Gunakan tone praktis dan efisien. "
            "Prioritaskan saran yang relevan untuk aktivitas puncak siang serta manajemen panas/cuaca aktif."
        )
    elif 15 <= hour < 19:
        periode = "sore"
        gaya = (
            "Gunakan tone transisi yang suportif untuk akhir hari. "
            "Berikan saran tindak lanjut hingga malam dengan fokus keamanan perjalanan/pulang."
        )
    else:
        periode = "malam"
        gaya = (
            "Gunakan tone lebih tenang dan ringkas. "
            "Fokus pada keamanan malam, kenyamanan istirahat, dan rencana esok hari bila relevan."
        )

    return (
        "[Context Assembly: Time-of-Day Response Style]\n"
        f"Periode lokal user saat ini: {periode}.\n"
        f"Instruksi gaya jawaban: {gaya}"
    )


def build_llm_context(
    *,
    user_text: str,
    weather: Optional[Dict[str, Any]],
    system_instructions: Optional[str],
    assistant_instructions: Optional[str],
    notifications_for_llm: List[Dict[str, Any]],
    history_data: List[Dict[str, Any]],
    dokumen_ditemukan: Optional[List[Dict[str, Any]]],
    timezone_name: Optional[str],
    utc_offset_minutes: Optional[int],
) -> Dict[str, Any]:
    konteks_list: List[str] = []
    kategori_utama = "umum"

    if dokumen_ditemukan:
        kategori_utama = dokumen_ditemukan[0].get("kategori", "umum")
        for doc in dokumen_ditemukan:
            kategori_doc = doc.get("kategori", "umum")
            isi_doc = doc.get("content", "")
            konteks_list.append(f"[Kategori: {kategori_doc}]\n{isi_doc}")
        konteks_gabungan = "\n\n".join(konteks_list)
    else:
        konteks_gabungan = "[TIDAK ADA KONTEKS SPESIFIK DARI DATABASE UNTUK PERTANYAAN INI]"

    persona_mapping = {
        "mitigasi": "seorang ahli mitigasi iklim yang tegas, tenang, dan sangat peduli pada keselamatan. Nada bicaramu serius namun menenangkan, memberikan instruksi protektif yang jelas dan mudah diikuti.",
        "kesehatan": "seorang konsultan kesehatan cuaca yang penuh empati. Nada bicaramu sangat hangat dan peduli, fokus memberikan saran perawatan diri terbaik agar pengguna tetap sehat menghadapi cuaca.",
        "aktivitas": "seorang pemandu aktivitas dan agrikultur yang energik dan bersemangat. Nada bicaramu antusias dan memotivasi, memberikan saran yang praktis untuk kegiatan di luar ruangan.",
        "edukasi": "seorang ilmuwan cuaca yang cerdas dan sangat antusias. Nada bicaramu menginspirasi rasa ingin tahu, menjelaskan fenomena alam dengan bahasa yang menarik, layaknya sedang bercerita.",
        "umum": "asisten ahli meteorologi yang ramah, santai, dan suportif.",
    }

    peran_spesifik = persona_mapping.get(kategori_utama, persona_mapping["edukasi"])

    instruksi_persona = f"""Kamu adalah WAMchat, kamu berada dalam aplikasi yang disebut dengan WAMApp data cuaca user berasal dari situ. Kamu adalah {peran_spesifik}

ATURAN PENTING:
1. JANGAN PERNAH menggunakan frasa kaku seperti 'berdasarkan teks', 'menurut database', 'berdasarkan informasi yang saya miliki', atau yang sejenisnya. Jawablah mengalir seolah pengetahuan itu murni dari pikiranmu.
2. Gunakan referensi [Konteks Database] yang diberikan sebagai acuan fakta untuk menjawab.
3. Jika [Konteks Database] menunjukkan TIDAK ADA KONTEKS atau pertanyaan melenceng jauh dari topik cuaca/alam, tetaplah jawab dengan sopan memakai pengetahuan umummu, TAPI berikan sedikit klarifikasi ramah dengan gaya personamu bahwa kamu asisten cuaca."""

    additional_system = ""
    if system_instructions:
        additional_system += "\n\n" + system_instructions
    if assistant_instructions:
        additional_system += "\n\n" + assistant_instructions

    # Instruksi penting: gunakan keterangan/label cuaca, bukan hanya kode numerik
    label_instruction = (
        "[PENTING - FORMAT CUACA]\n"
        "Saat menyebut kondisi cuaca dalam jawaban, gunakan keterangan yang dapat dibaca manusia (mis. 'Cerah', 'Hujan', 'Badai petir') bukannya hanya kode numerik. "
        "Jika payload menyediakan keduanya, prioritaskan label/keterangan saat menyimpulkan kondisi atau memberi saran."
    )
    additional_system = label_instruction + "\n\n" + additional_system

    time_context, local_dt = _build_time_context(timezone_name, utc_offset_minutes)
    time_style_instruction = _build_time_style_instruction(local_dt)
    notification_context = format_notifications_for_llm(notifications_for_llm)

    context_assembly_instruction = (
        "[Context Assembly: Notification History]\n"
        "Berikut ringkasan anomali cuaca cuaca terbaru untuk user. "
        "Gunakan sebagai konteks tambahan saat menyusun jawaban, terutama untuk mitigasi dan tindak lanjut yang aman:\n"
        f"{notification_context}"
    )

    time_assembly_instruction = (
        "[Context Assembly: User Timezone]\n"
        "Gunakan waktu lokal user ini sebagai referensi utama saat menyebut waktu, jadwal, kata 'sekarang', 'hari ini', 'malam ini', 'besok', dan estimasi waktu lainnya. "
        "Jika user tidak menyebut zona waktu secara eksplisit, tetap ikuti timezone yang dikirim dari client atau offset UTC.\n"
        f"{time_context}"
    )

    combined_system_instruction = (
        instruksi_persona
        + "\n\n"
        + time_assembly_instruction
        + "\n\n"
        + time_style_instruction
        + "\n\n"
        + context_assembly_instruction
        + additional_system
    )

    history_text = ""
    if history_data:
        history_text = "Riwayat Percakapan Sebelumnya:\n"
        for item in history_data:
            nama = "User" if item.get("role") == "user" else "WAMchat"
            history_text += f"{nama}: {item.get('content', '')}\n"
        history_text += "\n"

    weather_text = ""
    if weather:
        try:
            weather_text = "External Weather Snapshot:\n" + json.dumps(weather, ensure_ascii=False) + "\n\n"
        except Exception:
            weather_text = "External Weather Snapshot: (unserializable)\n\n"

    user_content_lengkap = (
        f"{history_text}{weather_text}Konteks Database:\n{konteks_gabungan}\n\nPertanyaan User: {user_text}"
    )

    system_message = {"role": "system", "content": combined_system_instruction}
    user_message = {"role": "user", "content": user_content_lengkap}


    print("=== Debug Chat Context Assembly ===")
    print(f"Combined System Instruction:\n{combined_system_instruction}")
    # print(f"User Content:\n{user_content_lengkap}")
    # print("user_message dict:", user_message)
    
    
    return {
        "kategori_utama": kategori_utama,
        "combined_system_instruction": combined_system_instruction,
        "user_content_lengkap": user_content_lengkap,
        "system_message": system_message,
        "user_message": user_message,
        "konteks_gabungan": konteks_gabungan,
    }
