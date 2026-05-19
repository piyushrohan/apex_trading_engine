"""In-memory read model for FastAPI status endpoints."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional


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
    portfolio: Dict[str, Any] = field(default_factory=dict)
    ingestion_enabled: bool = True
    hedge_enabled: bool = False
    updated_at: Optional[str] = None
    _lock: Lock = field(default_factory=Lock, repr=False)

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
            if portfolio is not None:
                self.portfolio = dict(portfolio)
            if ingestion_enabled is not None:
                self.ingestion_enabled = ingestion_enabled
            if hedge_enabled is not None:
                self.hedge_enabled = hedge_enabled
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
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
            }


# Module-level singleton used by pipeline + API
_runtime_status = StatusStore()


def get_status_store() -> StatusStore:
    return _runtime_status
