"""Runtime read model for FastAPI status endpoints."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

RUNTIME_STATUS_PATH = Path(
    os.getenv("APEX_RUNTIME_STATUS_PATH", "data_lake/runtime_status.json")
)


@dataclass
class StatusStore:
    """Thread-safe snapshot of operator runtime state for read-only API."""

    operator_mode: str = "paper"
    symbol: str = "ETHUSDC"
    regime: Optional[str] = None
    mark_price: Optional[float] = None
    kill_switch_active: bool = False
    model_id: Optional[str] = None
    last_explanation: Optional[Dict[str, Any]] = None
    sizing_calibration: Optional[Dict[str, Any]] = None
    portfolio: Dict[str, Any] = field(default_factory=dict)
    ingestion_enabled: bool = True
    hedge_enabled: bool = False
    updated_at: Optional[str] = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    def _payload(self) -> Dict[str, Any]:
        return {
            "operator_mode": self.operator_mode,
            "symbol": self.symbol,
            "regime": self.regime,
            "mark_price": self.mark_price,
            "kill_switch_active": self.kill_switch_active,
            "model_id": self.model_id,
            "ingestion_enabled": self.ingestion_enabled,
            "hedge_enabled": self.hedge_enabled,
            "updated_at": self.updated_at,
            "portfolio": dict(self.portfolio),
            "last_explanation": self.last_explanation,
            "sizing_calibration": self.sizing_calibration,
        }

    def _persist(self, payload: Dict[str, Any]) -> None:
        try:
            RUNTIME_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = RUNTIME_STATUS_PATH.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            tmp_path.replace(RUNTIME_STATUS_PATH)
        except OSError:
            pass

    def _load_persisted(self) -> Optional[Dict[str, Any]]:
        try:
            if not RUNTIME_STATUS_PATH.exists():
                return None
            return json.loads(RUNTIME_STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def update(
        self,
        *,
        operator_mode: Optional[str] = None,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        mark_price: Optional[float] = None,
        kill_switch_active: Optional[bool] = None,
        model_id: Optional[str] = None,
        last_explanation: Optional[Dict[str, Any]] = None,
        sizing_calibration: Optional[Dict[str, Any]] = None,
        portfolio: Optional[Dict[str, Any]] = None,
        ingestion_enabled: Optional[bool] = None,
        hedge_enabled: Optional[bool] = None,
    ):
        with self._lock:
            if operator_mode is not None:
                self.operator_mode = operator_mode
            if symbol is not None:
                self.symbol = symbol
            if regime is not None:
                self.regime = regime
            if mark_price is not None:
                self.mark_price = mark_price
            if kill_switch_active is not None:
                self.kill_switch_active = kill_switch_active
            if model_id is not None:
                self.model_id = model_id
            if last_explanation is not None:
                self.last_explanation = last_explanation
            if sizing_calibration is not None:
                self.sizing_calibration = dict(sizing_calibration)
            if portfolio is not None:
                self.portfolio = dict(portfolio)
            if ingestion_enabled is not None:
                self.ingestion_enabled = ingestion_enabled
            if hedge_enabled is not None:
                self.hedge_enabled = hedge_enabled
            self.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist(self._payload())

    def snapshot(self) -> Dict[str, Any]:
        persisted = self._load_persisted()
        persisted_at = (persisted or {}).get("updated_at")
        if (
            persisted
            and persisted_at
            and (self.updated_at is None or persisted_at > self.updated_at)
        ):
            return persisted
        with self._lock:
            return self._payload()


# Module-level singleton used by pipeline + API
_runtime_status = StatusStore()


def get_status_store() -> StatusStore:
    return _runtime_status
