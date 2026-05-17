import asyncio

import pytest

from src.pipelines import shadow_trade
from src.pipelines.shadow_trade import ShadowTradePipeline


class FakeRegistry:
    registry_data = {"active_shadow": "shadow-v1"}


class FakeMetaController:
    def __init__(self, config):
        self.config = config

    def get_action(self, state_vector, regime):
        return 0, 0.91, {}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_shadow_trade_loop_processes_one_virtual_signal(mock_config, monkeypatch):
    """Verify shadow loop can process a non-flat virtual signal and stop."""
    sleeps = {"count": 0}

    async def fake_sleep(_):
        sleeps["count"] += 1
        if sleeps["count"] > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(shadow_trade, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(shadow_trade, "MetaController", FakeMetaController)
    monkeypatch.setattr(shadow_trade.asyncio, "sleep", fake_sleep)

    pipeline = ShadowTradePipeline(mock_config)
    pipeline._running = True
    await pipeline._shadow_loop()
    await pipeline.stop()

    assert pipeline._running is False
    assert pipeline.virtual_equity == mock_config["environment"]["initial_capital"]


@pytest.mark.integration
def test_shadow_trade_initializes_without_active_shadow(mock_config, monkeypatch):
    """Verify shadow pipeline still initializes when no shadow model is active."""

    class EmptyRegistry:
        registry_data = {"active_shadow": None}

    monkeypatch.setattr(shadow_trade, "ModelRegistry", EmptyRegistry)
    monkeypatch.setattr(shadow_trade, "MetaController", FakeMetaController)

    pipeline = ShadowTradePipeline(mock_config)

    assert pipeline._running is False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_shadow_trade_start_sets_running_and_exits_on_cancel(
    mock_config, monkeypatch
):
    """Verify start delegates into the shadow loop."""

    async def fake_shadow_loop(self):
        self._running = False

    monkeypatch.setattr(shadow_trade, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(shadow_trade, "MetaController", FakeMetaController)
    monkeypatch.setattr(ShadowTradePipeline, "_shadow_loop", fake_shadow_loop)

    pipeline = ShadowTradePipeline(mock_config)
    await pipeline.start()

    assert pipeline._running is False


@pytest.mark.asyncio
@pytest.mark.integration
async def test_shadow_trade_loop_recovers_from_model_errors(mock_config, monkeypatch):
    """Verify shadow loop logs model errors and continues until stopped."""

    class BrokenMetaController:
        def __init__(self, config):
            self.config = config

        def get_action(self, state_vector, regime):
            raise RuntimeError("model unavailable")

    sleeps = {"count": 0}

    async def fake_sleep(_):
        sleeps["count"] += 1
        if sleeps["count"] > 1:
            pipeline._running = False

    monkeypatch.setattr(shadow_trade, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(shadow_trade, "MetaController", BrokenMetaController)
    monkeypatch.setattr(shadow_trade.asyncio, "sleep", fake_sleep)
    pipeline = ShadowTradePipeline(mock_config)
    pipeline._running = True

    await pipeline._shadow_loop()

    assert sleeps["count"] == 2
