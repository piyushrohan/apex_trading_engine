from unittest.mock import AsyncMock

import pytest

from src.data import binance_rest
from src.data.binance_rest import BinanceRESTClient


class FakeResponse:
    def __init__(self, status, payload=None, text="error", headers=None):
        self.status = status
        self.payload = payload or {}
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


class FakeSession:
    def __init__(self, *, responses=None):
        self.closed = False
        self.responses = list(responses or [])
        self.calls = []

    def _next_response(self):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, endpoint, params=None):
        self.calls.append(("GET", endpoint, params))
        return self._next_response()

    def post(self, endpoint):
        self.calls.append(("POST", endpoint, None))
        return self._next_response()

    def put(self, endpoint):
        self.calls.append(("PUT", endpoint, None))
        return self._next_response()

    def delete(self, endpoint):
        self.calls.append(("DELETE", endpoint, None))
        return self._next_response()

    async def close(self):
        self.closed = True


def sample_kline(timestamp_ms):
    return [
        timestamp_ms,
        "3000.0",
        "3010.0",
        "2990.0",
        "3005.0",
        "100.0",
        timestamp_ms + 1,
        "0",
        10,
        "50",
        "0",
        "0",
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_rest_client_session_uses_api_key_header(monkeypatch):
    """Verify REST sessions include Binance API headers when configured."""
    captured = {}

    def fake_client_session(headers=None):
        captured["headers"] = headers
        return FakeSession()

    monkeypatch.setattr(binance_rest.aiohttp, "ClientSession", fake_client_session)
    client = BinanceRESTClient(api_key="api-key", api_secret="secret")

    session = await client._get_session()
    await client.close()

    assert captured["headers"] == {"X-MBX-APIKEY": "api-key"}
    assert session.closed is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fetch_klines_returns_json_on_success(monkeypatch):
    """Verify kline fetch returns decoded Binance payloads."""
    session = FakeSession(responses=[FakeResponse(200, payload=[[1, "open"]])])
    monkeypatch.setattr(
        BinanceRESTClient, "_get_session", AsyncMock(return_value=session)
    )

    data = await BinanceRESTClient().fetch_klines("ETHUSDC", "3m", 1, 2)

    assert data == [[1, "open"]]
    assert session.calls[0][2]["symbol"] == "ETHUSDC"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fetch_klines_retries_rate_limit_and_errors(monkeypatch):
    """Verify transient API failures back off and eventually return an empty list."""
    session = FakeSession(
        responses=[
            FakeResponse(429, headers={"Retry-After": "0"}),
            FakeResponse(500, text="server error"),
            RuntimeError("network down"),
        ]
    )
    monkeypatch.setattr(
        BinanceRESTClient, "_get_session", AsyncMock(return_value=session)
    )
    monkeypatch.setattr(binance_rest.asyncio, "sleep", AsyncMock())

    data = await BinanceRESTClient().fetch_klines("ETHUSDC", "3m", 1, 2)

    assert data == []
    assert len(session.calls) == 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_backfill_historical_data_formats_klines(monkeypatch):
    """Verify historical backfill converts raw klines into cache-ready OHLCV rows."""
    client = BinanceRESTClient()
    client.fetch_klines = AsyncMock(
        side_effect=[
            [sample_kline(0), sample_kline(60_000)],
            [],
        ]
    )
    monkeypatch.setattr(binance_rest.asyncio, "sleep", AsyncMock())

    df = await client.backfill_historical_data("ETHUSDC", "3m", 0, 120_000)

    assert list(df.columns) == [
        "timestamp",
        "symbol",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert df.loc[0, "symbol"] == "ETHUSDC"
    assert df.loc[0, "close"] == 3005.0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_backfill_historical_data_handles_empty_and_stuck_pages(monkeypatch):
    """Verify backfill exits cleanly on empty responses and stuck timestamps."""
    empty_client = BinanceRESTClient()
    empty_client.fetch_klines = AsyncMock(return_value=[])

    empty_df = await empty_client.backfill_historical_data("ETHUSDC", "3m", 0, 1)

    stuck_client = BinanceRESTClient()
    stuck_client.fetch_klines = AsyncMock(return_value=[sample_kline(0)])
    monkeypatch.setattr(binance_rest.asyncio, "sleep", AsyncMock())

    stuck_df = await stuck_client.backfill_historical_data("ETHUSDC", "3m", 0, 10)

    assert empty_df.empty
    assert len(stuck_df) == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_user_stream_endpoints_handle_success_and_failure(monkeypatch):
    """Verify listen-key REST helpers parse success and tolerate failure responses."""
    session = FakeSession(
        responses=[
            FakeResponse(200, payload={"listenKey": "listen-123"}),
            FakeResponse(400, text="bad request"),
            FakeResponse(500, text="put failed"),
            FakeResponse(500, text="delete failed"),
        ]
    )
    monkeypatch.setattr(
        BinanceRESTClient, "_get_session", AsyncMock(return_value=session)
    )

    client = BinanceRESTClient()

    assert await client.get_listen_key() == "listen-123"
    assert await client.get_listen_key() is None
    await client.keepalive_listen_key()
    await client.close_listen_key()
    assert [call[0] for call in session.calls] == ["POST", "POST", "PUT", "DELETE"]
