from unittest.mock import AsyncMock

import pytest

from src.execution.adapters.live import LiveExecutionAdapter


@pytest.mark.asyncio
@pytest.mark.unit
async def test_live_adapter_flatten_both_legs(mock_config):
    rest = AsyncMock()
    rest.cancel_all_open_orders = AsyncMock(return_value=1)
    rest.close_position_market = AsyncMock(return_value={"orderId": 1})
    mock_config["execution"]["position_mode"] = "hedge"

    adapter = LiveExecutionAdapter(mock_config, rest)
    summary = await adapter.flatten_all_positions(
        "ETHUSDC", long_qty=0.2, short_qty=0.1
    )

    assert summary["cancelled"] == 1
    assert summary["closed_long"] is True
    assert summary["closed_short"] is True
    assert rest.close_position_market.await_count == 2
