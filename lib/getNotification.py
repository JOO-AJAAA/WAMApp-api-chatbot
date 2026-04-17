import json
import os
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import UUID

from dotenv import load_dotenv


load_dotenv()
SUPABASE_URL = os.getenv("EXPO_SUPABASE_URL")
SUPABASE_KEY = os.getenv("EXPO_SUPABASE_ANON_KEY")


class DeviceIdUnavailableError(RuntimeError):
	pass


class NotificationFetchError(RuntimeError):
	pass


def _normalize_device_id(device_id: Optional[str]) -> str:
	cleaned = (device_id or "").strip()
	if not cleaned:
		raise DeviceIdUnavailableError("device_id is required for notifications RLS policy")

	try:
		return str(UUID(cleaned))
	except ValueError as exc:
		raise DeviceIdUnavailableError("device_id is invalid for notifications RLS policy") from exc


def _build_notifications_url(normalized_device_id: str, limit: int, since_iso: Optional[str]) -> str:
	if not SUPABASE_URL or not SUPABASE_KEY:
		raise NotificationFetchError("EXPO Supabase URL or ANON KEY is not configured in environment")

	base_url = SUPABASE_URL.rstrip("/")
	query_params = [
		("select", "id,device_id,title,message,category,data,is_read,created_at"),
		("device_id", f"eq.{normalized_device_id}"),
		("order", "created_at.desc"),
		("limit", str(limit)),
	]

	if since_iso:
		query_params.append(("created_at", f"gte.{since_iso}"))

	return f"{base_url}/rest/v1/notifications?{urlencode(query_params)}"


def _fetch_notifications(normalized_device_id: str, limit: int, since_iso: Optional[str]) -> List[Dict[str, Any]]:
	url = _build_notifications_url(normalized_device_id, limit, since_iso)
	headers = {
		"apikey": SUPABASE_KEY or "",
		"Authorization": f"Bearer {SUPABASE_KEY or ''}",
		"Accept": "application/json",
		"Content-Type": "application/json",
		"x-device-id": normalized_device_id,
		"device_id": normalized_device_id,
	}

	request = Request(url, headers=headers, method="GET")

	try:
		with urlopen(request, timeout=20) as response:
			payload = response.read().decode("utf-8")
			data = json.loads(payload) if payload else []
			if not isinstance(data, list):
				raise NotificationFetchError("Supabase response is not a list")
			return data
	except HTTPError as exc:
		detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
		raise NotificationFetchError(f"Supabase HTTP error: {exc.code} {detail}") from exc
	except URLError as exc:
		raise NotificationFetchError(f"Supabase connection error: {exc.reason}") from exc
	except json.JSONDecodeError as exc:
		raise NotificationFetchError(f"Invalid JSON response from Supabase: {exc}") from exc


def get_notifications_for_device(device_id: str, limit: int = 100, since_iso: Optional[str] = None) -> List[Dict[str, Any]]:
	"""Fetch notification history for a given device id.

	Args:
		device_id: UUID string of the device to query notifications for.
		limit: maximum number of notifications to return (default 100).
		since_iso: optional ISO8601 timestamp string; if provided, only return notifications created_at >= since_iso.

	Returns:
		List of notification records (dictionaries) ordered by created_at descending.

	Raises:
		DeviceIdUnavailableError if device_id is missing/invalid.
		NotificationFetchError if the query fails or env not configured.
	"""
	normalized_device_id = _normalize_device_id(device_id)
	return _fetch_notifications(normalized_device_id, limit, since_iso)
