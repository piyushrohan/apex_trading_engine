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
    """Verify WebSocket connect wires the combined streams and parses messages."""
    captured = {"messages": []}

    def fake_connect(url):
        return FakeWebSocketContext(url, captured)

    monkeypatch.setattr(binance_ws.websockets, "connect", fake_connect)

    ws = BinanceWebSocket(ws_config)
    ws._handle_message = lambda data: captured["messages"].append(data)

    await ws.connect()

    assert "ethusdc@aggTrade" in captured["url"]
    assert "ethusdc@depth5@100ms" in captured["url"]
    assert "btcusdc@aggTrade" in captured["url"]
    assert captured["messages"][0]["data"]["p"] == "3000"


@pytest.mark.unit
def test_binance_websocket_init_normalizes_symbols(ws_config):
    """Verify configured symbols are normalized for Binance stream names."""
    ws = BinanceWebSocket(ws_config)

    assert ws.target_symbol == "ethusdc"
    assert ws.macro_symbol == "btcusdc"
    assert ws._handle_message({"stream": "noop"}) is None
