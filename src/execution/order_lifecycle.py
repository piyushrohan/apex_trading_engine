"""Order lifecycle telemetry for paper and live execution paths."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class OrderLifecycleEvent:
    timestamp: str
    event: str
    order_id: str
    symbol: str
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    status: Optional[str] = None
    execution_mode: Optional[str] = None
    book_id: Optional[str] = None
    position_side: Optional[str] = None
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    reason: Optional[str] = None
    queue_age_ms: Optional[float] = None
    fill_price: Optional[float] = None
    mark_price_after: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OrderLifecycleRecorder:
    """Append-only lifecycle recorder with optional DuckDB persistence."""

    def __init__(
        self,
        path: str = "data_lake/order_lifecycle.jsonl",
        *,
        cache: Any = None,
        execution_mode: str = "paper",
        book_id: str = "primary",
    ):
        self.path = Path(path)
        self.cache = cache
        self.execution_mode = execution_mode
        self.book_id = book_id
        self._submitted_at: Dict[str, datetime] = {}

    def record(
        self,
        event: str,
        *,
        order_id: str,
        symbol: str,
        side: Optional[str] = None,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        status: Optional[str] = None,
        position_side: Optional[str] = None,
        exchange_order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        reason: Optional[str] = None,
        fill_price: Optional[float] = None,
        mark_price_after: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        ts = timestamp or datetime.now(timezone.utc)
        if event in {"submitted", "accepted", "open"}:
            self._submitted_at.setdefault(order_id, ts)
        queue_age_ms = self._queue_age_ms(order_id, ts)
        payload = OrderLifecycleEvent(
            timestamp=ts.isoformat(),
            event=event,
            order_id=str(order_id),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            status=status,
            execution_mode=self.execution_mode,
            book_id=self.book_id,
            position_side=position_side,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
            reason=reason,
            queue_age_ms=queue_age_ms,
            fill_price=fill_price,
            mark_price_after=mark_price_after,
            metadata=metadata or {},
        ).as_dict()
        self._append_jsonl(payload)
        if self.cache is not None and hasattr(
            self.cache, "insert_order_lifecycle_event"
        ):
            self.cache.insert_order_lifecycle_event(payload)
        return payload

    def _queue_age_ms(self, order_id: str, timestamp: datetime) -> Optional[float]:
        started = self._submitted_at.get(order_id)
        if started is None:
            return None
        return max((timestamp - started).total_seconds() * 1000.0, 0.0)

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def summarize_order_lifecycle(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return trader-facing execution quality metrics from lifecycle rows."""
    items = list(rows)
    submitted = [row for row in items if row.get("event") == "submitted"]
    fills = [
        row
        for row in items
        if row.get("event") in {"filled", "partially_filled", "fill"}
    ]
    cancels = [row for row in items if row.get("event") == "canceled"]
    rejects = [row for row in items if row.get("event") == "rejected"]
    queue_ages = [
        float(row["queue_age_ms"])
        for row in fills
        if row.get("queue_age_ms") is not None
    ]
    drift_values = []
    for row in fills:
        fill_price = row.get("fill_price") or row.get("price")
        mark_after = row.get("mark_price_after")
        if fill_price and mark_after:
            drift_values.append(
                (float(mark_after) - float(fill_price)) / float(fill_price)
            )
    return {
        "events": len(items),
        "submitted": len(submitted),
        "fills": len(fills),
        "cancels": len(cancels),
        "rejects": len(rejects),
        "fill_rate": len(fills) / len(submitted) if submitted else 0.0,
        "cancel_replace_ratio": len(cancels) / len(submitted) if submitted else 0.0,
        "avg_queue_age_ms": (sum(queue_ages) / len(queue_ages) if queue_ages else None),
        "avg_post_fill_drift_bps": (
            sum(drift_values) / len(drift_values) * 10000.0 if drift_values else None
        ),
    }
