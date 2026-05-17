import asyncio
from unittest.mock import AsyncMock

import pytest

from src.pipelines import live_trade
from src.pipelines.live_trade import LiveTradePipeline


class FakeRESTClient:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeAccountSynchronizer:
    def __init__(self, rest_client):
        self.rest_client = rest_client
        self.started = False
        self.stopped = False
        self.on_position_change = None

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


class FakeRiskEngine:
    def __init__(self, config):
        self.config = config

    def calculate_kelly_size(self, win_rate, win_loss_ratio, conviction):
        return 0.25

    def approve_order(self, side, proposed_fraction, current_exposure):
        return proposed_fraction


class FakeOrderManager:
    def __init__(self, config, rest_client):
        self.place_maker_order = AsyncMock(return_value={"orderId": 123})


class FakeRegistry:
    def get_prod_model_path(self):
        return None


class FakeMetaController:
    def __init__(self, config):
        self.config = config

    def get_action(self, state_vector, regime):
        return 2, 0.9, {"feature_contributions": [0.1] * 10}


class FakeExplainability:
    def __init__(self, config):
        self.config = config

    def decode_decision(self, action, conviction, context):
        return {"primary_reasons": ["test reason"]}


def patch_live_dependencies(monkeypatch):
    monkeypatch.setattr(live_trade, "BinanceRESTClient", FakeRESTClient)
    monkeypatch.setattr(live_trade, "AccountSynchronizer", FakeAccountSynchronizer)
    monkeypatch.setattr(live_trade, "RiskEngine", FakeRiskEngine)
    monkeypatch.setattr(live_trade, "OrderManager", FakeOrderManager)
    monkeypatch.setattr(live_trade, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(live_trade, "MetaController", FakeMetaController)
    monkeypatch.setattr(live_trade, "ExplainabilityEngine", FakeExplainability)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_trade_start_runs_one_trading_iteration(mock_config, monkeypatch):
    """Verify live pipeline starts sync, processes one approved trade, and stops."""
    patch_live_dependencies(monkeypatch)
    sleeps = {"count": 0}

    async def fake_sleep(_):
        sleeps["count"] += 1
        if sleeps["count"] > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(live_trade.asyncio, "sleep", fake_sleep)
    pipeline = LiveTradePipeline(mock_config)

    await pipeline.start()
    await pipeline.stop()

    pipeline.order_manager.place_maker_order.assert_awaited_once_with(
        "BUY", 1.0, 3500.49
    )
    assert pipeline.account_sync.started is True
    assert pipeline.account_sync.stopped is True
    assert pipeline.rest_client.closed is True


@pytest.mark.integration
def test_live_trade_manual_position_callback_initializes(mock_config, monkeypatch):
    """Verify live pipeline construction and position callback are safe."""
    patch_live_dependencies(monkeypatch)
    pipeline = LiveTradePipeline(mock_config)

    pipeline._on_manual_position_change("ETHUSDC", {"amount": 1.0})

    assert pipeline._running is False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_trade_loop_recovers_from_iteration_errors(mock_config, monkeypatch):
    """Verify live loop catches model/runtime errors and backs off."""

    class BrokenMetaController:
        def __init__(self, config):
            self.config = config

        def get_action(self, state_vector, regime):
            raise RuntimeError("model unavailable")

    patch_live_dependencies(monkeypatch)
    monkeypatch.setattr(live_trade, "MetaController", BrokenMetaController)
    sleeps = {"count": 0}

    async def fake_sleep(_):
        sleeps["count"] += 1
        if sleeps["count"] > 1:
            pipeline._running = False

    monkeypatch.setattr(live_trade.asyncio, "sleep", fake_sleep)
    pipeline = LiveTradePipeline(mock_config)
    pipeline._running = True

    await pipeline._trading_loop()

    assert sleeps["count"] == 2
