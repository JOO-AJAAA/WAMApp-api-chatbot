import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - fallback for environments without zoneinfo
    ZoneInfo = None


def format_notifications_for_llm(notifications: List[Dict[str, Any]]) -> str:
    if not notifications:
        return "No relevant weather anomaly history is available."

    lines: List[str] = ["Latest weather anomaly history:"]
    for item in notifications[:10]:
        title = item.get("title") or "Untitled"
        message = item.get("message") or item.get("content") or item.get("data") or ""
        category = item.get("category") or "general"
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
        f"User timezone: {timezone_label}\n"
        f"Current local time: {local_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Current date: {local_dt.strftime('%A, %d %B %Y')}",
        local_dt,
    )


def _build_time_style_instruction(local_dt: datetime) -> str:
    hour = local_dt.hour

    if 5 <= hour < 11:
        period = "morning"
        gaya = (
            "Use a fresh, light, motivating, and actionable tone to help the user start the day. "
            "If giving advice, prioritize morning preparation and plans through midday."
        )
    elif 11 <= hour < 15:
        period = "afternoon"
        gaya = (
            "Use a practical and efficient tone. "
            "Prioritize advice relevant to peak midday activity and heat/weather management."
        )
    elif 15 <= hour < 19:
        period = "evening"
        gaya = (
            "Use a supportive transition tone for the end of the day. "
            "Provide follow-up advice into the night with a focus on travel/return safety."
        )
    else:
        period = "night"
        gaya = (
            "Use a calmer and more concise tone. "
            "Focus on nighttime safety, rest comfort, and plans for tomorrow if relevant."
        )

    return (
        "[Context Assembly: Time-of-Day Response Style]\n"
        f"Current local period: {period}.\n"
        f"Response style instruction: {gaya}"
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
    context_list: List[str] = []
    primary_category = "general"

    if dokumen_ditemukan:
        primary_category = dokumen_ditemukan[0].get("kategori", "general")
        for doc in dokumen_ditemukan:
            category_doc = doc.get("kategori", "general")
            doc_content = doc.get("content", "")
            context_list.append(f"[Category: {category_doc}]\n{doc_content}")
        combined_context = "\n\n".join(context_list)
    else:
        combined_context = "[NO SPECIFIC DATABASE CONTEXT IS AVAILABLE FOR THIS QUESTION]"

    persona_mapping = {
        "mitigasi": "seorang ahli mitigasi iklim yang tegas, tenang, dan sangat peduli pada keselamatan. Nada bicaramu serius namun menenangkan, memberikan instruksi protektif yang jelas dan mudah diikuti.",
        "kesehatan": "seorang konsultan kesehatan cuaca yang penuh empati. Nada bicaramu sangat hangat dan peduli, fokus memberikan saran perawatan diri terbaik agar pengguna tetap sehat menghadapi cuaca.",
        "aktivitas": "seorang pemandu aktivitas dan agrikultur yang energik dan bersemangat. Nada bicaramu antusias dan memotivasi, memberikan saran yang praktis untuk kegiatan di luar ruangan.",
        "edukasi": "seorang ilmuwan cuaca yang cerdas dan sangat antusias. Nada bicaramu menginspirasi rasa ingin tahu, menjelaskan fenomena alam dengan bahasa yang menarik, layaknya sedang bercerita.",
        "umum": "asisten ahli meteorologi yang ramah, santai, dan suportif.",
    }

    specific_role = persona_mapping.get(primary_category, persona_mapping["edukasi"])

    instruksi_persona = f"""You are WAMchat, and you are inside an application called WAMApp where the user's weather data comes from. You are {specific_role}

IMPORTANT RULES:
1. NEVER use stiff phrases such as 'based on the text', 'according to the database', 'based on the information I have', or anything similar. Respond naturally as if the knowledge comes directly from your own mind.
2. Use the provided [Database Context] as the factual basis for your answer.
3. If [Database Context] shows NO CONTEXT or the question drifts far away from weather/nature topics, still answer politely using your general knowledge, BUT add a friendly clarification in your persona style that you are a weather assistant.
4. Always reply in the same language as the user's latest message. If the user message is mixed-language, follow the dominant language of that message.
"""

    additional_system = ""
    if system_instructions:
        additional_system += "\n\n" + system_instructions
    if assistant_instructions:
        additional_system += "\n\n" + assistant_instructions

    # Important instruction: use weather descriptions/labels, not only numeric codes
    label_instruction = (
        "[IMPORTANT - WEATHER FORMAT]\n"
        "When describing weather conditions in your answer, use human-readable labels (e.g. 'Clear', 'Rain', 'Thunderstorm') instead of only numeric codes. "
        "If the payload provides both, prioritize the label/description when inferring the condition or giving advice."
    )
    additional_system = label_instruction + "\n\n" + additional_system

    time_context, local_dt = _build_time_context(timezone_name, utc_offset_minutes)
    time_style_instruction = _build_time_style_instruction(local_dt)
    notification_context = format_notifications_for_llm(notifications_for_llm)

    context_assembly_instruction = (
        "[Context Assembly: Notification History]\n"
        "Below is a summary of the user's latest weather anomalies. "
        "Use it as additional context when composing the answer, especially for mitigation and safe follow-up actions:\n"
        f"{notification_context}"
    )

    time_assembly_instruction = (
        "[Context Assembly: User Timezone]\n"
        "Use the user's local time as the primary reference when mentioning time, schedules, the words 'now', 'today', 'tonight', 'tomorrow', and any other time estimates. "
        "If the user does not explicitly mention a timezone, still follow the timezone sent by the client or the UTC offset.\n"
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
        history_text = "Previous conversation history:\n"
        for item in history_data:
            speaker = "User" if item.get("role") == "user" else "WAMchat"
            history_text += f"{speaker}: {item.get('content', '')}\n"
        history_text += "\n"

    weather_text = ""
    if weather:
        try:
            weather_text = "External weather snapshot:\n" + json.dumps(weather, ensure_ascii=False) + "\n\n"
        except Exception:
            weather_text = "External weather snapshot: (unserializable)\n\n"

    user_content_lengkap = (
        f"{history_text}{weather_text}Database Context:\n{combined_context}\n\nUser Question: {user_text}"
    )

    system_message = {"role": "system", "content": combined_system_instruction}
    user_message = {"role": "user", "content": user_content_lengkap}    
    
    return {
        "kategori_utama": primary_category,
        "combined_system_instruction": combined_system_instruction,
        "user_content_lengkap": user_content_lengkap,
        "system_message": system_message,
        "user_message": user_message,
        "konteks_gabungan": combined_context,
        "primary_category": primary_category,
        "combined_context": combined_context,
    }
