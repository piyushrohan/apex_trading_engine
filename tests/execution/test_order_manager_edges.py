from unittest.mock import AsyncMock

import pytest

from src.execution.order_manager import OrderManager


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cancel_order_returns_true(mock_config):
    """Verify cancel_order exposes a successful mocked cancellation."""
    rest_client = AsyncMock()
    rest_client.cancel_order = AsyncMock(return_value=True)
    manager = OrderManager(mock_config, rest_client)

    assert await manager.cancel_order("order-123") is True
    rest_client.cancel_order.assert_awaited_once_with("ETHUSDC", "order-123")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cancel_and_replace_abandons_when_chase_is_rejected(mock_config):
    """Verify stale orders are cancelled and not replaced when alpha decays."""
    rest_client = AsyncMock()
    manager = OrderManager(mock_config, rest_client)
    manager.slippage_manager.should_chase_order = lambda *args: False
    manager.cancel_order = AsyncMock(return_value=True)

    result = await manager.cancel_and_replace(
        "order-123", "BUY", 1.0, 3000.0, 0.5, 3001.0
    )

    assert result is None
    manager.cancel_order.assert_awaited_once_with("order-123")
    rest_client.place_order.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cancel_and_replace_places_new_order_when_chase_is_approved(mock_config):
    """Verify approved chase flow cancels and submits a replacement maker order."""
    rest_client = AsyncMock()
    rest_client.place_order.return_value = {"orderId": 123}
    manager = OrderManager(mock_config, rest_client)
    manager.slippage_manager.should_chase_order = lambda *args: True
    manager.cancel_order = AsyncMock(return_value=True)

    result = await manager.cancel_and_replace(
        "order-123", "SELL", 2.0, 3100.0, 0.95, 3099.0
    )

    assert result == {"orderId": 123}
    manager.cancel_order.assert_awaited_once_with("order-123")
    rest_client.place_order.assert_awaited_once()
