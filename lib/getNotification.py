import os
from typing import List, Optional, Dict, Any
from supabase import create_client

SUPABASE_URL = os.getenv("EXPO_SUPABASE_URL")
SUPABASE_KEY = os.getenv("EXPO_SUPABASE_ANON_KEY")


def _get_supabase_client():
	"""Create a Supabase client using the service key from env.

	Returns:
		supabase.Client: configured client
	"""
	if not SUPABASE_URL or not SUPABASE_KEY:
		raise RuntimeError("EXPO Supabase URL or ANON KEY is not configured in environment")

	return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_notifications_for_device(device_id: str, limit: int = 100, since_iso: Optional[str] = None) -> List[Dict[str, Any]]:
	"""Fetch notification history for a given device id.

	Args:
		device_id: UUID string of the device to query notifications for.
		limit: maximum number of notifications to return (default 100).
		since_iso: optional ISO8601 timestamp string; if provided, only return notifications created_at >= since_iso.

	Returns:
		List of notification records (dictionaries) ordered by created_at descending.

	Raises:
		RuntimeError if the query fails or env not configured.
	"""
	client = _get_supabase_client()

	try:
		query = client.table('notifications').select('id, device_id, title, message, category, data, is_read, created_at')
		query = query.eq('device_id', device_id)
		if since_iso:
			# Supabase/python client supports gte filter
			query = query.gte('created_at', since_iso)

		query = query.order('created_at', desc=True).limit(limit)

		response = query.execute()

		# The client returns a dict-like response with 'data' and 'error'
		if hasattr(response, 'error') and response.error:
			raise RuntimeError(f"Supabase query error: {response.error}")

		# Some client versions return a dict
		data = None
		if isinstance(response, dict):
			data = response.get('data')
			error = response.get('error')
			if error:
				raise RuntimeError(f"Supabase query error: {error}")
		else:
			# response may be a custom object with .data attribute
			data = getattr(response, 'data', None)

		return data or []

	except Exception as exc:
		raise RuntimeError(f"Failed to fetch notifications: {exc}") from exc
