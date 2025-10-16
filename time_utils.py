"""
Shared time utilities for logging and debug output.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

_IST = timezone(timedelta(hours=5, minutes=30))


def now_ist_stamp() -> str:
    """
    Return a compact timestamp string in IST (HH:MM:SS).
    """
    return datetime.now(_IST).strftime("[%H:%M:%S]")


__all__ = ["now_ist_stamp"]

