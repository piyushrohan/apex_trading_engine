"""Separate kill-switch lanes for operator and runtime safety controls."""

from datetime import datetime, timezone
from typing import Any, Dict, Iterable

KILL_SWITCH_LANES = ("manual", "model", "data", "execution", "account_sync")


def default_kill_switch_lanes() -> Dict[str, Dict[str, Any]]:
    return {
        lane: {"active": False, "reason": None, "updated_at": None}
        for lane in KILL_SWITCH_LANES
    }


def normalize_kill_switch_lanes(raw: Any) -> Dict[str, Dict[str, Any]]:
    lanes = default_kill_switch_lanes()
    if isinstance(raw, dict):
        for lane, payload in raw.items():
            if lane not in lanes:
                continue
            if isinstance(payload, dict):
                lanes[lane] = {
                    "active": bool(payload.get("active", False)),
                    "reason": payload.get("reason"),
                    "updated_at": payload.get("updated_at"),
                }
            else:
                lanes[lane]["active"] = bool(payload)
    return lanes


def set_kill_switch_lane(
    lanes: Dict[str, Dict[str, Any]],
    lane: str,
    *,
    active: bool,
    reason: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    if lane not in KILL_SWITCH_LANES:
        raise ValueError(f"Unknown kill-switch lane: {lane}")
    normalized = normalize_kill_switch_lanes(lanes)
    normalized[lane] = {
        "active": active,
        "reason": reason,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    return normalized


def kill_switch_active(lanes: Dict[str, Dict[str, Any]]) -> bool:
    return any(bool(payload.get("active")) for payload in lanes.values())


def active_kill_switch_lanes(lanes: Dict[str, Dict[str, Any]]) -> Iterable[str]:
    return [lane for lane, payload in lanes.items() if payload.get("active")]
