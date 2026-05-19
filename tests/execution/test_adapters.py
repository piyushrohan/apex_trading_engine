import pytest

from src.execution.adapters.base import OrderRequest
from src.execution.adapters.paper import PaperExecutionAdapter


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
