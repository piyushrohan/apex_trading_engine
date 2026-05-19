"""Binance kline interval helpers."""

import re
from datetime import timedelta

_INTERVAL_RE = re.compile(r"^(\d+)([mhdw])$")


def interval_to_timedelta(interval: str) -> timedelta:
    match = _INTERVAL_RE.match(interval)
    if not match:
        raise ValueError(f"Unsupported interval: {interval}")
    value = int(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    if unit == "w":
        return timedelta(weeks=value)
    raise ValueError(f"Unsupported interval unit: {unit}")


def interval_to_milliseconds(interval: str) -> int:
    return int(interval_to_timedelta(interval).total_seconds() * 1000)
