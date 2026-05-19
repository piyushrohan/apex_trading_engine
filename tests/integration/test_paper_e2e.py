"""Paper path: market state → inference → virtual fill → journal."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.adapters.paper import PaperExecutionAdapter
from src.pipelines import trading_pipeline
from src.pipelines.trading_pipeline import TradingPipeline
from tests.fixtures.market_data import generate_liquidity_sweep_klines


class FakeIngestion:
    def __init__(self, config, rest_client=None, cache=None):
        self.config = config
        self.cache = cache or MagicMock()
        self._mark = 3500.0

    async def bootstrap_historical(self):
        return {}

    async def start_live(self):
        pass

    async def stop(self):
        pass

    async def flush_ticks(self):
        return 0

    def get_last_mark_price(self, symbol):
        return self._mark

    def close(self):
        pass


@pytest.mark.asyncio
@pytest.mark.integration
async def test_paper_loop_fill_and_journal(tmp_path, mock_config, monkeypatch):
    eth = generate_liquidity_sweep_klines()
    snapshot = {
        "state_vector": [0.1] * 10,
        "regime": "STRONG_TREND_UP",
        "mark_price": float(eth.iloc[-1]["close"]),
        "eth_btc_zscore": 0.0,
        "volatility_zscore": 0.0,
        "trend_slope": 0.01,
        "is_buy_liquidity_sweep": False,
        "is_sell_liquidity_sweep": False,
    }

    class FakeMarketState:
        def __init__(self, config, cache=None):
            pass

        def build_latest(self):
            return snapshot

        def close(self):
            pass

    class FakeMeta:
        def __init__(self, config):
            pass

        def get_dual_inference(self, state_vector, regime):
            ctx = {
                "action_probs": [0.05, 0.05, 0.9],
                "feature_contributions": [0.1] * 10,
            }
            probs = [0.05, 0.05, 0.9]
            return 2, 0.85, ctx, probs, [0.2, 0.2, 0.6]

    journal_path = tmp_path / "trade_journal.jsonl"
    logged = []

    class FakeExplain:
        def __init__(self, config):
            self.trade_journal_path = str(journal_path)

        def decode_decision(
            self, action, conviction, context, write_journal=False, **kwargs
        ):
            return {
                "decision": "LONG",
                "primary_reasons": ["test"],
                "risk_factors": [],
                "schema_version": 2,
            }

        def _log_to_journal(self, explanation):
            logged.append(explanation)

    monkeypatch.setattr(trading_pipeline, "DataIngestionService", FakeIngestion)
    monkeypatch.setattr(trading_pipeline, "MarketStateService", FakeMarketState)
    monkeypatch.setattr(trading_pipeline, "MetaController", FakeMeta)
    monkeypatch.setattr(trading_pipeline, "ExplainabilityEngine", FakeExplain)
    monkeypatch.setattr(
        trading_pipeline,
        "build_hedge_orchestrator",
        lambda config: MagicMock(evaluate=lambda ctx: (None, {"enabled": False})),
    )
    monkeypatch.setattr(
        trading_pipeline,
        "BinanceRESTClient",
        lambda *a, **k: MagicMock(close=AsyncMock()),
    )
    monkeypatch.setattr(
        trading_pipeline,
        "ModelRegistry",
        lambda: MagicMock(get_prod_model_path=lambda: None),
    )

    config = dict(mock_config)
    config["data"]["ingestion"] = {"enabled": True}
    config["data"]["loop_interval_sec"] = 0.01
    config["execution"]["operator_mode"] = "paper"

    pipeline = TradingPipeline(config)
    paper_adapter = PaperExecutionAdapter(book_id="primary")
    pipeline.execution_adapter = paper_adapter

    sleeps = {"n": 0}

    async def fake_sleep(_):
        sleeps["n"] += 1
        mark = snapshot["mark_price"]
        if sleeps["n"] == 1:
            await pipeline._execute_signal(2, 0.85, {"primary_reasons": []}, mark, 0.1)
        elif sleeps["n"] == 2:
            pipeline._simulate_paper_fills(mark - 0.05)
            pipeline._running = False
        elif sleeps["n"] > 2:
            pipeline._running = False

    monkeypatch.setattr(trading_pipeline.asyncio, "sleep", fake_sleep)
    pipeline._running = True
    await pipeline._trading_loop()

    assert paper_adapter._fills, "expected at least one virtual fill"
    assert logged, "expected journal entries"
    assert logged[-1].get("execution", {}).get("mode") == "paper"
    assert "hedge" in logged[-1]
