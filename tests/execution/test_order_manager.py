import pytest
from unittest.mock import AsyncMock, patch
from src.execution.order_manager import OrderManager

@pytest.fixture
def mock_rest_client():
    client = AsyncMock()
    # Mocking a successful post-only limit order response
    client.place_order.return_value = {
        "orderId": 123456,
        "status": "NEW",
        "symbol": "ETHUSDC"
    }
    return client

@pytest.mark.asyncio
async def test_order_manager_enforces_maker_only(mock_config, mock_rest_client):
    """
    Verify that OrderManager absolutely enforces timeInForce='GTX' 
    (Post-Only/Maker-Only) when placing orders.
    """
    manager = OrderManager(mock_config, mock_rest_client)
    
    # Place a simulated buy order
    await manager.place_maker_order("BUY", 1.5, 3000.0)
    
    # Verify the mock was called correctly
    mock_rest_client.place_order.assert_called_once_with(
        symbol="ETHUSDC",
        side="BUY",
        quantity=1.5,
        price=3000.0,
        timeInForce="GTX", # CRITICAL check
        orderType="LIMIT"
    )

@pytest.mark.asyncio
async def test_order_manager_handles_api_failure(mock_config, mock_rest_client):
    """
    Chaos Test: Verify that OrderManager gracefully handles Binance API exceptions 
    without crashing the main loop.
    """
    # Force an exception
    mock_rest_client.place_order.side_effect = Exception("API Timeout")
    
    manager = OrderManager(mock_config, mock_rest_client)
    
    # Should not raise an unhandled exception
    result = await manager.place_maker_order("SELL", 2.0, 3100.0)
    
    assert result is None
