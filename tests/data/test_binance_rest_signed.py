import hashlib
import hmac
from unittest.mock import AsyncMock
from urllib.parse import urlencode

import pytest

from src.data import binance_rest
from src.data.binance_rest import BinanceRESTClient


class FakeResponse:
    def __init__(self, status, payload=None, text="error", headers=None):
        self.status = status
        self.payload = payload
        self._text = text
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return self._text


class ParamSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def _next_response(self):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url, params=None):
        self.calls.append(("GET", url, params))
        return self._next_response()

    def post(self, url, params=None):
        self.calls.append(("POST", url, params))
        return self._next_response()

    def delete(self, url, params=None):
        self.calls.append(("DELETE", url, params))
        return self._next_response()

    def put(self, url, params=None):
        self.calls.append(("PUT", url, params))
        return self._next_response()

    async def close(self):
        self.closed = True


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


@pytest.mark.unit
def test_sign_params_requires_secret_and_generates_signature(monkeypatch):
    client = BinanceRESTClient(api_key="k", api_secret="secret")
    monkeypatch.setattr(binance_rest.time, "time", lambda: 1234.567)

    signed = client._sign_params({"symbol": "ETHUSDC"})
    expected_payload = {"symbol": "ETHUSDC", "timestamp": 1234567}
    expected = hmac.new(
        b"secret", urlencode(expected_payload).encode(), hashlib.sha256
    ).hexdigest()

    assert signed["timestamp"] == 1234567
    assert signed["signature"] == expected

    with pytest.raises(ValueError, match="API secret required"):
        BinanceRESTClient()._sign_params({})


@pytest.mark.asyncio
@pytest.mark.unit
async def test_signed_request_retries_errors_and_rejects_unknown_methods(monkeypatch):
    session = ParamSession(
        [
            FakeResponse(500, text="server down"),
            FakeResponse(201, {"ok": True}),
            RuntimeError("socket reset"),
            FakeResponse(200, {"deleted": True}),
        ]
    )
    monkeypatch.setattr(binance_rest.asyncio, "sleep", AsyncMock())
    client = BinanceRESTClient(api_key="k", api_secret="secret")
    client.session = session

    created = await client._signed_request("POST", "/fapi/v1/order", {"a": 1})
    deleted = await client._signed_request("DELETE", "/fapi/v1/order", {"b": 2})
    unsupported = await client._signed_request("PATCH", "/fapi/v1/order", {})

    assert created == {"ok": True}
    assert deleted == {"deleted": True}
    assert unsupported is None
    assert session.calls[0][0] == "POST"
    assert "signature" in session.calls[0][2]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_signed_account_helpers_parse_success_and_fallbacks(monkeypatch):
    client = BinanceRESTClient(api_key="k", api_secret="s")
    results = iter(
        [
            {"orderId": 1},
            None,
            {"msg": "ok"},
            None,
            {"code": -2011},
            {"dualSidePosition": True},
            [],
            {"symbol": "ETHUSDC", "leverage": 3},
            [{"asset": "BTC"}, {"asset": "USDC", "balance": "100"}],
            [{"asset": "BTC"}],
            {"bad": "shape"},
            [
                {"symbol": "ETHUSDC", "positionAmt": "1"},
                {"symbol": "BTCUSDC", "positionAmt": "2"},
            ],
            {"bad": "shape"},
            [{"orderId": 1}],
            {"bad": "shape"},
            [{"id": 1}],
            None,
        ]
    )
    calls = []

    async def fake_signed(method, endpoint, params=None):
        calls.append((method, endpoint, params or {}))
        return next(results)

    monkeypatch.setattr(client, "_signed_request", fake_signed)

    assert await client.cancel_order("ETHUSDC", "1") is True
    assert await client.cancel_order("ETHUSDC", "2") is False
    assert await client.cancel_all_open_orders("ETHUSDC") == 1
    assert await client.cancel_all_open_orders("ETHUSDC") == 0
    assert await client.cancel_all_open_orders("ETHUSDC") == 0
    assert await client.set_hedge_mode(enabled=False) is True
    assert await client.get_position_mode() is None
    assert await client.set_leverage("ETHUSDC", 3) == {
        "symbol": "ETHUSDC",
        "leverage": 3,
    }
    assert await client.get_usdc_balance() == {"asset": "USDC", "balance": "100"}
    assert await client.get_usdc_balance() is None
    assert await client.get_usdc_balance() is None
    assert await client.get_positions("ETHUSDC") == [
        {"symbol": "ETHUSDC", "positionAmt": "1"}
    ]
    assert await client.get_positions() == []
    assert await client.get_open_orders("ETHUSDC") == [{"orderId": 1}]
    assert await client.get_open_orders("ETHUSDC") == []
    assert await client.get_recent_fills("ETHUSDC") == [{"id": 1}]
    assert await client.get_recent_fills("ETHUSDC") == []
    assert calls[5][2]["dualSidePosition"] == "false"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_public_market_helpers_handle_shapes_and_failures():
    session = ParamSession(
        [
            FakeResponse(200, {"lastFundingRate": "0.0001", "markPrice": "3500"}),
            FakeResponse(200, {"openInterest": "123.45"}),
            FakeResponse(200, []),
            FakeResponse(500, text="bad request"),
            RuntimeError("offline"),
        ]
    )
    client = BinanceRESTClient()
    client.session = session

    assert await client.fetch_premium_index("ETHUSDC") == {
        "lastFundingRate": "0.0001",
        "markPrice": "3500",
    }
    assert await client.fetch_open_interest("ETHUSDC") == 123.45
    assert await client.fetch_premium_index("ETHUSDC") is None
    assert await client._public_get("/bad") is None
    assert await client._public_get("/boom") is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_listen_key_success_paths_clear_cached_key(monkeypatch):
    session = ParamSession(
        [
            FakeResponse(200, {"listenKey": "listen-123"}),
            FakeResponse(200, {}),
            FakeResponse(200, {}),
        ]
    )
    monkeypatch.setattr(
        BinanceRESTClient, "_get_session", AsyncMock(return_value=session)
    )
    client = BinanceRESTClient(api_key="k")

    assert await client.get_listen_key() == "listen-123"
    await client.keepalive_listen_key()
    await client.close_listen_key()

    assert client.listen_key is None
    assert session.calls[1][2] == {"listenKey": "listen-123"}
