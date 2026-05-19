import pytest

from src.data.binance_rest import BinanceRESTClient


@pytest.mark.asyncio
@pytest.mark.unit
async def test_signed_place_order_includes_position_side(monkeypatch):
    client = BinanceRESTClient(api_key="k", api_secret="s")
    captured = {}

    async def fake_signed(method, endpoint, params=None):
        captured["method"] = method
        captured["endpoint"] = endpoint
        captured["params"] = params
        return {"orderId": 99, "status": "NEW"}

    monkeypatch.setattr(client, "_signed_request", fake_signed)

    result = await client.place_order(
        "ETHUSDC",
        "BUY",
        0.1,
        3500.0,
        positionSide="LONG",
    )

    assert result["orderId"] == 99
    assert captured["params"]["positionSide"] == "LONG"
    assert captured["method"] == "POST"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cancel_all_and_close_position(monkeypatch):
    client = BinanceRESTClient(api_key="k", api_secret="s")
    calls = []

    async def fake_signed(method, endpoint, params=None):
        calls.append((method, endpoint, params))
        return {"orderId": 1}

    monkeypatch.setattr(client, "_signed_request", fake_signed)

    await client.cancel_all_open_orders("ETHUSDC")
    await client.close_position_market("ETHUSDC", "SELL", 0.5, position_side="LONG")

    assert calls[0][0] == "DELETE"
    assert calls[1][1] == "/fapi/v1/order"
    assert calls[1][2]["reduceOnly"] == "true"
