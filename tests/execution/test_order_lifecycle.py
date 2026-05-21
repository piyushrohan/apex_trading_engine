import asyncio
import json

import pytest

from src.execution.adapters.base import OrderRequest
from src.execution.adapters.paper import PaperExecutionAdapter
from src.execution.kill_switch import (
    active_kill_switch_lanes,
    kill_switch_active,
    normalize_kill_switch_lanes,
    set_kill_switch_lane,
)
from src.execution.order_lifecycle import (
    OrderLifecycleRecorder,
    summarize_order_lifecycle,
)


@pytest.mark.unit
def test_order_lifecycle_recorder_writes_events_and_summary(tmp_path):
    path = tmp_path / "orders.jsonl"
    recorder = OrderLifecycleRecorder(str(path), execution_mode="paper", book_id="p")

    recorder.record(
        "submitted",
        order_id="o1",
        symbol="ETHUSDC",
        side="BUY",
        quantity=1.0,
        price=100.0,
        status="PENDING",
    )
    fill = recorder.record(
        "filled",
        order_id="o1",
        symbol="ETHUSDC",
        side="BUY",
        quantity=1.0,
        price=100.0,
        status="FILLED",
        fill_price=100.0,
        mark_price_after=101.0,
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    summary = summarize_order_lifecycle(rows)

    assert fill["queue_age_ms"] >= 0
    assert rows[0]["book_id"] == "p"
    assert summary["submitted"] == 1
    assert summary["fills"] == 1
    assert summary["fill_rate"] == 1.0
    assert summary["avg_post_fill_drift_bps"] == 100.0


@pytest.mark.unit
def test_paper_adapter_records_open_fill_and_cancel_lifecycle(tmp_path):
    path = tmp_path / "orders.jsonl"
    recorder = OrderLifecycleRecorder(str(path), execution_mode="paper", book_id="p")
    adapter = PaperExecutionAdapter(book_id="p", lifecycle_recorder=recorder)

    async def scenario():
        result = await adapter.place_order(
            OrderRequest(
                symbol="ETHUSDC",
                side="BUY",
                quantity=1.0,
                price=100.0,
                client_order_id="client-1",
            )
        )
        fills = adapter.try_fill_on_market("ETHUSDC", 99.0)
        await adapter.place_order(
            OrderRequest(
                symbol="ETHUSDC",
                side="SELL",
                quantity=1.0,
                price=102.0,
                client_order_id="client-2",
            )
        )
        await adapter.cancel_order("ETHUSDC", "client-2")
        return result, fills

    result, fills = asyncio.run(scenario())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    events = [row["event"] for row in rows]

    assert result.success is True
    assert fills[0]["status"] == "FILLED"
    assert "submitted" in events
    assert "open" in events
    assert "filled" in events
    assert "canceled" in events


@pytest.mark.unit
def test_order_lifecycle_cache_hook_and_unknown_queue_age(tmp_path):
    class Cache:
        def __init__(self):
            self.events = []

        def insert_order_lifecycle_event(self, event):
            self.events.append(event)

    cache = Cache()
    recorder = OrderLifecycleRecorder(
        str(tmp_path / "orders.jsonl"),
        cache=cache,
        execution_mode="paper",
        book_id="primary",
    )

    fill = recorder.record(
        "filled",
        order_id="unknown-order",
        symbol="ETHUSDC",
        side="BUY",
        quantity=1.0,
        price=100.0,
        status="FILLED",
    )

    assert fill["queue_age_ms"] is None
    assert cache.events == [fill]


@pytest.mark.unit
def test_kill_switch_lane_normalization_and_validation():
    lanes = normalize_kill_switch_lanes(
        {
            "data": True,
            "execution": {
                "active": True,
                "reason": "exchange_down",
                "updated_at": "2026-05-21T00:00:00+00:00",
            },
            "unknown": True,
        }
    )

    assert lanes["data"]["active"] is True
    assert lanes["execution"]["reason"] == "exchange_down"
    assert "unknown" not in lanes
    assert kill_switch_active(lanes) is True
    assert set(active_kill_switch_lanes(lanes)) == {"data", "execution"}

    cleared = set_kill_switch_lane(lanes, "data", active=False, reason="fresh")
    assert cleared["data"]["active"] is False
    with pytest.raises(ValueError, match="Unknown kill-switch lane"):
        set_kill_switch_lane(lanes, "bad-lane", active=True)
