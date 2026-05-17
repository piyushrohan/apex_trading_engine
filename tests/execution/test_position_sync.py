import json
from unittest.mock import AsyncMock

import pytest

from src.execution import position_sync
from src.execution.position_sync import AccountSynchronizer


@pytest.mark.asyncio
@pytest.mark.integration
async def test_account_synchronizer_does_not_start_without_listen_key():
    """Verify account sync stays stopped when Binance cannot issue a listen key."""
    rest_client = AsyncMock()
    rest_client.get_listen_key.return_value = None
    sync = AccountSynchronizer(rest_client)

    await sync.start()

    assert sync._running is False
    assert sync.listen_key is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_account_synchronizer_start_creates_background_tasks(monkeypatch):
    """Verify successful start stores the listen key and creates worker tasks."""
    rest_client = AsyncMock()
    rest_client.get_listen_key.return_value = "listen-123"
    created = []

    class FakeTask:
        def cancel(self):
            pass

    def fake_create_task(coro):
        coro.close()
        created.append(coro)
        return FakeTask()

    monkeypatch.setattr(position_sync.asyncio, "create_task", fake_create_task)
    sync = AccountSynchronizer(rest_client)

    await sync.start()

    assert sync._running is True
    assert sync.listen_key == "listen-123"
    assert len(created) == 2


@pytest.mark.asyncio
@pytest.mark.integration
async def test_account_synchronizer_keepalive_loop_pings_until_cancelled(monkeypatch):
    """Verify keepalive loop pings Binance while the synchronizer is running."""
    rest_client = AsyncMock()
    sync = AccountSynchronizer(rest_client)
    sync.listen_key = "listen-123"
    sync._running = True
    sleeps = {"count": 0}

    async def fake_sleep(_):
        sleeps["count"] += 1
        if sleeps["count"] > 1:
            raise position_sync.asyncio.CancelledError

    monkeypatch.setattr(position_sync.asyncio, "sleep", fake_sleep)

    with pytest.raises(position_sync.asyncio.CancelledError):
        await sync._keepalive_loop()

    rest_client.keepalive_listen_key.assert_awaited_once()


class FakeWSMessage:
    def __init__(self, message_type, data="{}"):
        self.type = message_type
        self.data = data


class FakeUserDataWebSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)


class FakeAioHTTPSession:
    def __init__(self, messages):
        self.messages = messages

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def ws_connect(self, endpoint):
        return FakeUserDataWebSocket(self.messages)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_account_synchronizer_websocket_loop_routes_and_refreshes(monkeypatch):
    """Verify websocket loop routes messages and refreshes the listen key."""
    rest_client = AsyncMock()
    sync = AccountSynchronizer(rest_client)
    sync.listen_key = "listen-123"
    sync._running = True

    async def refresh_listen_key():
        sync._running = False
        return "listen-456"

    rest_client.get_listen_key.side_effect = refresh_listen_key
    messages = [
        FakeWSMessage(
            position_sync.aiohttp.WSMsgType.TEXT,
            json.dumps({"e": "ACCOUNT_UPDATE", "a": {"B": [], "P": []}}),
        ),
        FakeWSMessage(position_sync.aiohttp.WSMsgType.CLOSED),
    ]
    monkeypatch.setattr(
        position_sync.aiohttp,
        "ClientSession",
        lambda: FakeAioHTTPSession(messages),
    )

    await sync._websocket_loop()

    assert sync.listen_key == "listen-456"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_account_synchronizer_websocket_loop_backs_off_on_errors(monkeypatch):
    """Verify websocket loop catches connection errors and backs off."""
    rest_client = AsyncMock()
    sync = AccountSynchronizer(rest_client)
    sync.listen_key = "listen-123"
    sync._running = True

    def broken_session():
        raise RuntimeError("connection failed")

    async def fake_sleep(_):
        sync._running = False

    monkeypatch.setattr(position_sync.aiohttp, "ClientSession", broken_session)
    monkeypatch.setattr(position_sync.asyncio, "sleep", fake_sleep)

    await sync._websocket_loop()

    rest_client.get_listen_key.assert_not_called()


@pytest.mark.integration
def test_account_synchronizer_handles_account_and_order_events():
    """Verify user-data stream events update balances, positions, and orders."""
    sync = AccountSynchronizer(AsyncMock())
    position_events = []
    order_events = []
    sync.on_position_change = lambda symbol, position: position_events.append(
        (symbol, position)
    )
    sync.on_order_update = lambda order_id, order: order_events.append(
        (order_id, order)
    )

    sync._handle_account_update(
        {
            "a": {
                "B": [{"a": "USDC", "wb": "1000", "cw": "900", "bc": "5"}],
                "P": [
                    {
                        "s": "ETHUSDC",
                        "pa": "1.5",
                        "ep": "3000",
                        "up": "25",
                        "mt": "cross",
                    }
                ],
            }
        }
    )
    sync._handle_order_update(
        {
            "o": {
                "s": "ETHUSDC",
                "c": "client-1",
                "X": "FILLED",
                "z": "1.5",
                "p": "3005",
                "S": "BUY",
                "o": "LIMIT",
            }
        }
    )

    assert sync.balances["USDC"]["wallet_balance"] == 1000.0
    assert sync.positions["ETHUSDC"]["amount"] == 1.5
    assert sync.open_orders["client-1"]["status"] == "FILLED"
    assert position_events[0][0] == "ETHUSDC"
    assert order_events[0][0] == "client-1"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_account_synchronizer_routes_events_and_stops():
    """Verify event routing and graceful stop cleanup."""
    rest_client = AsyncMock()
    sync = AccountSynchronizer(rest_client)
    sync.listen_key = "listen-123"
    sync._running = True

    class FakeTask:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    sync._ws_task = FakeTask()
    sync._keepalive_task = FakeTask()

    await sync._handle_event({"e": "listenKeyExpired"})
    await sync._handle_event({"e": "ACCOUNT_UPDATE", "a": {"B": [], "P": []}})
    await sync._handle_event(
        {
            "e": "ORDER_TRADE_UPDATE",
            "o": {
                "s": "ETHUSDC",
                "c": "client-2",
                "X": "NEW",
                "z": "0",
                "p": "3000",
                "S": "SELL",
                "o": "LIMIT",
            },
        }
    )
    await sync._handle_event({"e": "IGNORED"})
    await sync.stop()

    assert sync._running is False
    assert sync.listen_key is None
    assert sync._ws_task.cancelled is True
    assert sync._keepalive_task.cancelled is True
    rest_client.close_listen_key.assert_awaited_once()
