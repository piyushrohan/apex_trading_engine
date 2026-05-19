from unittest.mock import AsyncMock

import pytest

from src.execution.adapters.base import ExecutionAdapter, OrderRequest
from src.execution.adapters.live import LiveExecutionAdapter
from src.execution.adapters.paper import PaperExecutionAdapter
from src.execution.factory import create_execution_adapter


@pytest.mark.unit
@pytest.mark.asyncio
async def test_paper_adapter_places_gtx_order():
    adapter = PaperExecutionAdapter(book_id="primary")
    result = await adapter.place_order(
        OrderRequest(
            symbol="ETHUSDC",
            side="BUY",
            quantity=1.0,
            price=3500.0,
            time_in_force="GTX",
        )
    )
    assert result.success is True
    assert result.order_id is not None
    open_orders = await adapter.get_open_orders("ETHUSDC")
    assert len(open_orders) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_paper_adapter_rejects_non_gtx():
    adapter = PaperExecutionAdapter()
    result = await adapter.place_order(
        OrderRequest(
            symbol="ETHUSDC",
            side="BUY",
            quantity=1.0,
            price=3500.0,
            time_in_force="GTC",
        )
    )
    assert result.success is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_paper_adapter_maker_fill_simulation():
    adapter = PaperExecutionAdapter(book_id="primary")
    await adapter.place_order(
        OrderRequest(symbol="ETHUSDC", side="BUY", quantity=2.0, price=3500.0)
    )
    fills = adapter.try_fill_on_market("ETHUSDC", 3499.0)
    assert len(fills) == 1
    assert fills[0]["executedQty"] == 2.0


class DummyExecutionAdapter(ExecutionAdapter):
    def __init__(self):
        self.cancelled = []

    async def place_order(self, request):
        return None

    async def cancel_order(self, symbol, order_id):
        self.cancelled.append((symbol, order_id))
        return order_id != "reject"

    async def get_open_orders(self, symbol):
        return [
            {"orderId": "exchange-1"},
            {"order_id": "local-2"},
            {"orderId": "reject"},
            {"clientOrderId": "missing-order-id"},
        ]

    async def sync_fills(self, symbol):
        return []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_base_adapter_cancel_all_orders_counts_successes():
    adapter = DummyExecutionAdapter()

    cancelled = await adapter.cancel_all_orders("ETHUSDC")

    assert cancelled == 2
    assert adapter.cancelled == [
        ("ETHUSDC", "exchange-1"),
        ("ETHUSDC", "local-2"),
        ("ETHUSDC", "reject"),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_paper_adapter_cancel_partial_fill_sync_and_flatten():
    adapter = PaperExecutionAdapter(book_id="shadow", maker_fee_pct=0.001)
    await adapter.place_order(
        OrderRequest(
            symbol="ETHUSDC",
            side="BUY",
            quantity=2.0,
            price=3500.0,
            client_order_id="buy-1",
        )
    )
    await adapter.place_order(
        OrderRequest(
            symbol="ETHUSDC",
            side="SELL",
            quantity=1.0,
            price=3510.0,
            client_order_id="sell-1",
        )
    )

    fills = adapter.try_fill_on_market(
        "ETHUSDC",
        market_price=3499.0,
        aggressor_side="SELL",
        available_quantity=0.5,
    )
    assert fills[0]["status"] == "PARTIALLY_FILLED"
    assert fills[0]["fee"] == pytest.approx(1.75)
    assert adapter.try_fill_on_market("ETHUSDC", 3499.0, available_quantity=0) == []
    assert await adapter.sync_fills("ETHUSDC") == fills

    assert await adapter.cancel_order("ETHUSDC", "buy-1") is True
    assert await adapter.cancel_order("ETHUSDC", "missing") is False

    sell_fills = adapter.try_fill_on_market("ETHUSDC", market_price=3511.0)
    assert sell_fills[0]["status"] == "FILLED"
    assert await adapter.get_open_orders("ETHUSDC") == []

    await adapter.place_order(
        OrderRequest(
            symbol="ETHUSDC",
            side="BUY",
            quantity=1.0,
            price=3490.0,
            client_order_id="flatten-me",
        )
    )
    assert adapter.flatten_all_virtual_orders("ETHUSDC") == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_adapter_order_paths_and_passthroughs(mock_config):
    rest = AsyncMock()
    rest.place_order.return_value = {
        "orderId": 123,
        "status": "PARTIALLY_FILLED",
        "executedQty": "0.25",
    }
    rest.cancel_order.return_value = True
    rest.get_open_orders.return_value = [{"orderId": 123}]
    rest.get_recent_fills.return_value = [{"orderId": 123, "qty": "0.25"}]
    adapter = LiveExecutionAdapter(mock_config, rest)

    result = await adapter.place_order(
        OrderRequest(
            symbol="ETHUSDC",
            side="buy",
            quantity=1.0,
            price=3500.0,
            position_side="LONG",
        )
    )

    assert result.success is True
    assert result.order_id == "123"
    assert result.filled_qty == 0.25
    assert await adapter.cancel_order("ETHUSDC", "123") is True
    assert await adapter.get_open_orders("ETHUSDC") == [{"orderId": 123}]
    assert await adapter.sync_fills("ETHUSDC") == [{"orderId": 123, "qty": "0.25"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_adapter_handles_place_order_none_and_errors(mock_config):
    rest = AsyncMock()
    rest.place_order.return_value = None
    adapter = LiveExecutionAdapter(mock_config, rest)

    none_result = await adapter.place_order(
        OrderRequest(symbol="ETHUSDC", side="BUY", quantity=1.0, price=3500.0)
    )
    assert none_result.success is False
    assert "None" in none_result.error

    rest.place_order.side_effect = RuntimeError("exchange rejected")
    error_result = await adapter.place_order(
        OrderRequest(symbol="ETHUSDC", side="SELL", quantity=1.0, price=3500.0)
    )
    assert error_result.success is False
    assert error_result.error == "exchange rejected"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_live_adapter_flatten_reports_failed_legs(mock_config):
    rest = AsyncMock()
    rest.cancel_all_open_orders = AsyncMock(return_value=0)
    rest.close_position_market = AsyncMock(return_value=None)
    mock_config["execution"]["position_mode"] = "hedge"
    adapter = LiveExecutionAdapter(mock_config, rest)

    summary = await adapter.flatten_all_positions("ETHUSDC", 0.2, 0.1)

    assert summary == {"cancelled": 0, "closed_long": False, "closed_short": False}
    assert rest.close_position_market.await_count == 2


@pytest.mark.unit
def test_execution_factory_selects_live_and_paper_adapters(mock_config):
    rest = AsyncMock()

    paper = create_execution_adapter(mock_config, rest, book_id="shadow")
    assert isinstance(paper, PaperExecutionAdapter)
    assert paper.book_id == "shadow"

    mock_config["execution"]["operator_mode"] = "live"
    live = create_execution_adapter(mock_config, rest)
    assert isinstance(live, LiveExecutionAdapter)
