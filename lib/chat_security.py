import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import jwt
from fastapi import HTTPException, Request


_CHAT_RATE_BUCKET: Dict[str, List[float]] = {}


def get_client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if req.client and req.client.host:
        return req.client.host
    return "unknown_ip"


def enforce_rate_limit(device_id: str, ip: str, max_requests: int, window_seconds: int) -> None:
    now = time.time()
    key = f"{device_id}:{ip}"
    window_start = now - window_seconds

    timestamps = _CHAT_RATE_BUCKET.get(key, [])
    timestamps = [ts for ts in timestamps if ts > window_start]

    if len(timestamps) >= max_requests:
        raise HTTPException(status_code=429, detail="Terlalu banyak request. Coba lagi sebentar.")

    timestamps.append(now)
    _CHAT_RATE_BUCKET[key] = timestamps


def create_chat_access_token(
    device_id: str,
    jwt_secret: str,
    jwt_expire_seconds: int,
    jwt_issuer: str,
    jwt_audience: str,
    jwt_algorithm: str = "HS256",
) -> str:
    if not jwt_secret:
        raise HTTPException(status_code=500, detail="JWT_SECRET belum dikonfigurasi di server.")

    now = datetime.now(timezone.utc)
    exp = now + timedelta(seconds=jwt_expire_seconds)

    payload = {
        "sub": device_id,
        "iss": jwt_issuer,
        "aud": jwt_audience,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }

    return jwt.encode(payload, jwt_secret, algorithm=jwt_algorithm)


def verify_chat_access_token(
    authorization: Optional[str],
    jwt_secret: str,
    jwt_issuer: str,
    jwt_audience: str,
    jwt_algorithm: str = "HS256",
) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Token tidak ditemukan.")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Format Authorization harus Bearer token.")

    token = parts[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Token kosong.")

    if not jwt_secret:
        raise HTTPException(status_code=500, detail="JWT_SECRET belum dikonfigurasi di server.")

    try:
        decoded = jwt.decode(
            token,
            jwt_secret,
            algorithms=[jwt_algorithm],
            audience=jwt_audience,
            issuer=jwt_issuer,
        )
        return decoded
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Token tidak valid.") from exc
