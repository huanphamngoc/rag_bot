"""HTTP helpers shared by the model providers."""
from __future__ import annotations

import email.utils
from datetime import datetime, timezone


def parse_retry_after(value: str | None) -> float | None:
    """Seconds to wait from a Retry-After header (delay in seconds or an HTTP date)."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
