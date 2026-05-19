import asyncio
import json

import pytest

from src.data import binance_ws
from src.data.binance_ws import BinanceWebSocket


class FakeWebSocket:
    def __init__(self):
        self.reads = 0

    async def recv(self):
        self.reads += 1
        if self.reads == 1:
            return json.dumps({"stream": "ethusdc@aggTrade", "data": {"p": "3000"}})
        raise RuntimeError("stop stream")


class FakeWebSocketContext:
    def __init__(self, url, captured):
        self.url = url
        self.captured = captured

    async def __aenter__(self):
        self.captured["url"] = self.url
        return FakeWebSocket()

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def ws_config():
    return {
        "data": {
            "urls": {"ws_stream": "wss://stream.test/stream"},
            "target_symbol": "ETHUSDC",
            "macro_symbol": "BTCUSDC",
        }
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_binance_websocket_builds_stream_url_and_handles_messages(
    ws_config, monkeypatch
):
    """Verify WebSocket wires combined streams including markPrice."""
    captured = {"messages": []}

    def fake_connect(url):
        return FakeWebSocketContext(url, captured)

    monkeypatch.setattr(binance_ws.websockets, "connect", fake_connect)

    stop = asyncio.Event()
    ws = BinanceWebSocket(
        ws_config, on_message=lambda data: captured["messages"].append(data)
    )

    async def stop_soon():
        await asyncio.sleep(0.01)
        stop.set()

    asyncio.create_task(stop_soon())
    await ws.run_until_stopped(stop)

    url = captured["url"]
    assert "ethusdc@aggTrade" in url
    assert "ethusdc@depth5@100ms" in url
    assert "ethusdc@markPrice@1s" in url
    assert "btcusdc@aggTrade" in url
    assert captured["messages"][0]["data"]["p"] == "3000"


@pytest.mark.unit
def test_binance_websocket_init_normalizes_symbols(ws_config):
    ws = BinanceWebSocket(ws_config)
    assert ws.target_symbol == "ethusdc"
    assert ws.macro_symbol == "btcusdc"
    names = ws.stream_names()
    assert any("markPrice" in n for n in names)
    assert ws._handle_message({"stream": "noop"}) is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_binance_websocket_timeout_and_cancel_paths(ws_config, monkeypatch):
    class IdleWebSocket:
        async def recv(self):
            return "{}"

    class IdleContext:
        async def __aenter__(self):
            return IdleWebSocket()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    stop = asyncio.Event()
    waits = {"count": 0}

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        waits["count"] += 1
        stop.set()
        raise asyncio.TimeoutError

    monkeypatch.setattr(binance_ws.websockets, "connect", lambda url: IdleContext())
    monkeypatch.setattr(binance_ws.asyncio, "wait_for", fake_wait_for)

    await BinanceWebSocket(ws_config).run_until_stopped(stop)

    assert waits["count"] == 1

    async def cancelled_run(stop_event):
        raise asyncio.CancelledError

    class CancelContext:
        async def __aenter__(self):
            raise asyncio.CancelledError

        async def __aexit__(self, exc_type, exc, tb):
            return False

    stop.clear()
    monkeypatch.setattr(binance_ws.websockets, "connect", lambda url: CancelContext())
    await BinanceWebSocket(ws_config, on_message=cancelled_run).run_until_stopped(stop)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_binance_websocket_legacy_connect_sets_stop_on_cancel(
    ws_config, monkeypatch
):
    events = []

    async def fake_run_until_stopped(self, stop_event):
        events.append(("run", stop_event.is_set()))
        await stop_event.wait()
        events.append(("stopped", stop_event.is_set()))

    async def fake_sleep(_):
        raise asyncio.CancelledError

    monkeypatch.setattr(BinanceWebSocket, "run_until_stopped", fake_run_until_stopped)
    monkeypatch.setattr(binance_ws.asyncio, "sleep", fake_sleep)

    await BinanceWebSocket(ws_config).connect()

    assert events[-1] == ("stopped", True)
