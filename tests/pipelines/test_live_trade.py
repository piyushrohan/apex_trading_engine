import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.adapters.base import OrderRequest, OrderResult
from src.pipelines import trading_pipeline
from src.pipelines.live_trade import LiveTradePipeline


class FakeRESTClient:
    def __init__(self, api_key=None, api_secret=None):
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
        self.is_kill_switch_active = False

    def calculate_kelly_size(self, win_rate, win_loss_ratio, conviction):
        return 0.25

    def approve_order(self, side, proposed_fraction, current_exposure, **kwargs):
        return proposed_fraction

    def update_equity(self, equity):
        pass


class FakeExecutionAdapter:
    def __init__(self):
        self.place_order = AsyncMock(
            return_value=OrderResult(success=True, order_id="live_1")
        )

    async def cancel_order(self, symbol, order_id):
        return True

    async def get_open_orders(self, symbol):
        return []

    async def sync_fills(self, symbol):
        return []

    async def cancel_all_orders(self, symbol):
        return 0


class FakeRegistry:
    def get_prod_model_path(self):
        return None


class FakeMetaController:
    def __init__(self, config):
        self.config = config

    def get_dual_inference(self, state_vector, regime):
        ctx = {"action_probs": [0.1, 0.2, 0.7], "feature_contributions": [0.1] * 10}
        probs = [0.1, 0.2, 0.7]
        return 2, 0.9, ctx, probs, probs


class FakeExplainability:
    def __init__(self, config):
        self.config = config

    def decode_decision(
        self, action, conviction, context, write_journal=False, **kwargs
    ):
        return {
            "primary_reasons": ["test reason"],
            "risk_factors": [],
            "schema_version": 2,
        }

    def decode_portfolio_state(self, **kwargs):
        return {"event": "portfolio_sync", "schema_version": 2}

    def _log_to_journal(self, explanation):
        pass


class FakeHedgeOrchestrator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, ctx):
        return None, {"enabled": False}


class FakeIngestion:
    def __init__(self, *args, **kwargs):
        self.cache = MagicMock()

    async def bootstrap_historical(self):
        return {}

    async def start_live(self):
        pass

    async def stop(self):
        pass

    async def flush_ticks(self):
        return 0

    def get_last_mark_price(self, symbol):
        return 3500.49

    def close(self):
        pass


class FakeAccountSync:
    def __init__(self, rest_client):
        self.rest_client = rest_client
        self.balances = {}
        self.on_position_change = None
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def fetch_snapshot(self, symbol):
        return {
            "long_qty": 0.0,
            "short_qty": 0.0,
            "amount": 0.0,
            "unrealized_pnl": 0.0,
        }


class FakeMarketState:
    def __init__(self, config, cache=None):
        pass

    def build_latest(self):
        return {
            "state_vector": [0.1] * 10,
            "regime": "STRONG_TREND_UP",
            "mark_price": 3500.0,
            "eth_btc_zscore": 0.0,
            "volatility_zscore": 0.0,
            "trend_slope": 0.0,
            "is_buy_liquidity_sweep": False,
            "is_sell_liquidity_sweep": False,
        }

    def close(self):
        pass


def patch_pipeline_dependencies(monkeypatch):
    monkeypatch.setattr(trading_pipeline, "validate_live_startup", lambda config: None)
    monkeypatch.setattr(
        trading_pipeline, "check_api_credentials", lambda config: (True, None)
    )
    monkeypatch.setattr(trading_pipeline, "BinanceRESTClient", FakeRESTClient)
    monkeypatch.setattr(trading_pipeline, "DataIngestionService", FakeIngestion)
    monkeypatch.setattr(trading_pipeline, "MarketStateService", FakeMarketState)
    monkeypatch.setattr(trading_pipeline, "AccountSynchronizer", FakeAccountSync)
    monkeypatch.setattr(trading_pipeline, "RiskEngine", FakeRiskEngine)
    monkeypatch.setattr(
        trading_pipeline,
        "create_execution_adapter",
        lambda *a, **k: FakeExecutionAdapter(),
    )
    monkeypatch.setattr(trading_pipeline, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(trading_pipeline, "MetaController", FakeMetaController)
    monkeypatch.setattr(trading_pipeline, "ExplainabilityEngine", FakeExplainability)
    monkeypatch.setattr(
        trading_pipeline,
        "build_hedge_orchestrator",
        lambda config: FakeHedgeOrchestrator(config),
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_trade_start_runs_one_trading_iteration(mock_config, monkeypatch):
    """Verify live pipeline starts sync, processes one approved trade, and stops."""
    patch_pipeline_dependencies(monkeypatch)
    config = dict(mock_config)
    config["execution"] = {
        **config.get("execution", {}),
        "operator_mode": "live",
    }
    config["live"] = {"enabled": True}

    sleeps = {"count": 0}

    async def fake_sleep(_):
        sleeps["count"] += 1
        if sleeps["count"] > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(trading_pipeline.asyncio, "sleep", fake_sleep)
    pipeline = LiveTradePipeline(config)

    await pipeline.start()
    await pipeline.stop()

    pipeline.execution_adapter.place_order.assert_awaited_once()
    call_args = pipeline.execution_adapter.place_order.await_args[0][0]
    assert isinstance(call_args, OrderRequest)
    assert call_args.side == "BUY"
    assert call_args.price == pytest.approx(3500.48, rel=1e-3)
    assert pipeline.account_sync.started is True
    assert pipeline.account_sync.stopped is True
    assert pipeline.rest_client.closed is True


@pytest.mark.integration
def test_live_trade_manual_position_callback_updates_book(mock_config, monkeypatch):
    patch_pipeline_dependencies(monkeypatch)
    config = dict(mock_config)
    config["execution"] = {"operator_mode": "live"}
    config["live"] = {"enabled": True}
    pipeline = LiveTradePipeline(config)

    pipeline._on_account_position_update(
        "ETHUSDC",
        {
            "long_qty": 1.5,
            "short_qty": 0.0,
            "amount": 1.5,
            "unrealized_pnl": 10.0,
            "entry_price_long": 3000.0,
            "entry_price_short": 0.0,
        },
    )

    assert pipeline.primary_book.long_qty == 1.5


@pytest.mark.asyncio
@pytest.mark.integration
async def test_live_trade_loop_recovers_from_iteration_errors(mock_config, monkeypatch):
    class BrokenMetaController:
        def __init__(self, config):
            self.config = config

        def get_dual_inference(self, state_vector, regime):
            raise RuntimeError("model unavailable")

    patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr(trading_pipeline, "MetaController", BrokenMetaController)
    config = dict(mock_config)
    config["execution"] = {"operator_mode": "live"}
    config["live"] = {"enabled": True}

    sleeps = {"count": 0}

    async def fake_sleep(_):
        sleeps["count"] += 1
        if sleeps["count"] > 1:
            pipeline._running = False

    monkeypatch.setattr(trading_pipeline.asyncio, "sleep", fake_sleep)
    pipeline = LiveTradePipeline(config)
    pipeline._running = True

    await pipeline._trading_loop()

    assert sleeps["count"] == 2
