import json
from types import SimpleNamespace

import pytest

from src.execution.portfolio import PortfolioService
from src.mlops.registry import ModelRegistry
from src.mlops.shadow_lane import ShadowLaneRunner
from src.models.gbm_agent import GBMAgent
from src.strategies.hedge.base import HedgeProposal


@pytest.mark.asyncio
@pytest.mark.mlops
async def test_shadow_lane_runs_candidate_on_paper_adapter(tmp_path, mock_config):
    """Verify shadow lanes use virtual books and journal candidate decisions."""
    config = dict(mock_config)
    config["shadow"] = {
        "enabled": True,
        "auto_register": True,
        "max_parallel_candidates": 1,
        "decision_log_path": str(tmp_path / "decisions.jsonl"),
    }
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    model_path = registry.register_model("candidate-v1", "GBM", {"sharpe": 0.5})
    agent = GBMAgent(config)
    agent.save(model_path)
    portfolio = PortfolioService(position_mode="one_way")
    runner = ShadowLaneRunner(
        config=config,
        registry=registry,
        portfolio=portfolio,
        symbol="ETHUSDC",
        operator_mode="live",
    )

    decisions = await runner.run_tick(
        snapshot={
            "state_vector": [0.1] * 10,
            "regime": "MEAN_REVERSION",
        },
        mark_price=3000.0,
        hedge_payload={"enabled": True, "candidates": {"protective_hedge": 0.4}},
    )

    line = json.loads((tmp_path / "decisions.jsonl").read_text().strip())
    assert decisions[0]["book"]["role"] == "shadow"
    assert decisions[0]["execution"]["mode"] == "live"
    assert line["model_id"] == "candidate-v1"
    assert "shadow_candidate-v1" in portfolio.books


@pytest.mark.asyncio
@pytest.mark.mlops
async def test_shadow_lane_prioritizes_active_shadow_and_executes_hedges(
    tmp_path, mock_config, monkeypatch
):
    import src.mlops.shadow_lane as shadow_module

    class FakeRegistry:
        registry_data = {
            "active_shadow": "active-shadow",
            "models": {
                "candidate-v1": {"type": "GBM", "status": "CANDIDATE"},
                "active-shadow": {"type": "GBM", "status": "SHADOW"},
                "evaluating-v1": {"type": "GBM", "status": "EVALUATING"},
            },
        }

        def get_model_path(self, model_id):
            return f"/missing/{model_id}"

    class FakeController:
        def load_model_artifact(self, model_type, model_path):
            raise FileNotFoundError(model_path)

        def get_dual_inference(self, state_vector, regime):
            return (
                2,
                0.9,
                {
                    "action_probs": [0.1, 0.1, 0.8],
                    "ppo_action_probs": [0.1, 0.1, 0.8],
                    "gbm_action_probs": [0.2, 0.2, 0.6],
                },
                [0.1, 0.1, 0.8],
                [0.2, 0.2, 0.6],
            )

    class FakeRisk:
        def __init__(self):
            self.approvals = [0.2, 0.1, 0.05]

        def calculate_kelly_size(self, win_rate, win_loss_ratio, conviction):
            return 0.25

        def approve_order(self, *args, **kwargs):
            return self.approvals.pop(0)

    class RecordingAdapter:
        def __init__(self):
            self.orders = []

        async def place_order(self, request):
            self.orders.append(request)
            return SimpleNamespace(success=True)

        def try_fill_on_market(self, symbol, mark_price):
            return [
                {
                    "side": "BUY",
                    "executedQty": 0.01,
                    "avgPrice": mark_price,
                    "positionSide": "LONG",
                }
            ]

    class FakeHedge:
        def evaluate(self, context):
            assert context.symbol == "ETHUSDC"
            assert context.primary_action == 2
            return (
                shadow_module.HedgeProposal(
                    "protective_hedge", long_delta_qty=0.1, short_delta_qty=0.05
                ),
                {"enabled": True, "selected": "protective_hedge"},
            )

    config = dict(mock_config)
    config["execution"] = {**mock_config["execution"], "position_mode": "hedge"}
    config["hedge"] = {"enabled": True}
    config["shadow"] = {
        "enabled": True,
        "auto_register": True,
        "max_parallel_candidates": 2,
        "decision_log_path": str(tmp_path / "decisions.jsonl"),
    }
    monkeypatch.setattr(
        shadow_module, "MetaController", lambda config: FakeController()
    )
    monkeypatch.setattr(
        shadow_module, "build_hedge_orchestrator", lambda config: FakeHedge()
    )
    portfolio = PortfolioService(position_mode="hedge")

    runner = ShadowLaneRunner(
        config=config,
        registry=FakeRegistry(),
        portfolio=portfolio,
        symbol="ETHUSDC",
        operator_mode="paper",
    )
    lane = runner.lanes["active-shadow"]
    lane["risk"] = FakeRisk()
    lane["adapter"] = RecordingAdapter()

    decisions = await runner.run_tick(
        snapshot={
            "state_vector": [0.1] * 10,
            "regime": "MEAN_REVERSION",
            "volatility_zscore": 1.4,
            "funding_rate": -0.0003,
            "eth_btc_zscore": -1.7,
            "trend_slope": 0.01,
            "is_buy_liquidity_sweep": True,
            "is_sell_liquidity_sweep": False,
            "cvd": 12.0,
            "spread_bps": 1.2,
        },
        mark_price=3000.0,
    )

    assert list(runner.lanes) == ["active-shadow", "candidate-v1"]
    assert decisions[0]["hedge"]["selected"] == "protective_hedge"
    assert lane["book"].long_qty == 0.01
    assert [order.position_side for order in lane["adapter"].orders] == [
        "LONG",
        "LONG",
        "SHORT",
    ]


@pytest.mark.asyncio
@pytest.mark.mlops
async def test_shadow_lane_quarantines_artifacts_that_fail_preflight(
    tmp_path, mock_config, monkeypatch
):
    import src.mlops.shadow_lane as shadow_module

    model_dir = tmp_path / "models" / "bad-shadow"
    model_dir.mkdir(parents=True)
    (model_dir / "gbm_model.pkl").write_bytes(b"native-crash-payload")

    class FakeRegistry:
        registry_data = {
            "active_shadow": "bad-shadow",
            "models": {"bad-shadow": {"type": "GBM", "status": "SHADOW"}},
        }

        def get_model_path(self, model_id):
            return str(model_dir)

    monkeypatch.setattr(
        shadow_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=-11,
            stderr="Segmentation fault",
            stdout="",
        ),
    )
    portfolio = PortfolioService(position_mode="one_way")
    config = dict(mock_config)
    config["shadow"] = {"enabled": True, "max_parallel_candidates": 1}

    runner = ShadowLaneRunner(
        config=config,
        registry=FakeRegistry(),
        portfolio=portfolio,
        symbol="ETHUSDC",
        operator_mode="paper",
    )

    assert runner.lanes == {}
    assert "bad-shadow" in runner.disabled_candidates
    assert (
        "artifact_preflight_failed"
        in runner.disabled_candidates["bad-shadow"]["reason"]
    )


@pytest.mark.asyncio
@pytest.mark.mlops
async def test_shadow_lane_skips_empty_and_rejected_hedge_orders(mock_config):
    class RejectingRisk:
        def approve_order(self, *args, **kwargs):
            return 0.0

    class RecordingAdapter:
        def __init__(self):
            self.orders = []

        async def place_order(self, request):
            self.orders.append(request)

    portfolio = PortfolioService(position_mode="hedge")
    book = portfolio.get_or_create_book(
        book_id="shadow-v1",
        role="shadow",
        model_id="candidate-v1",
        symbol="ETHUSDC",
        initial_equity=1000.0,
    )
    runner = ShadowLaneRunner.__new__(ShadowLaneRunner)
    runner.symbol = "ETHUSDC"
    runner.portfolio = portfolio
    lane = {"book": book, "risk": RejectingRisk(), "adapter": RecordingAdapter()}

    await runner._place_shadow_hedge(lane, HedgeProposal("none"), 3000.0)
    await runner._place_shadow_hedge(
        lane, HedgeProposal("protective_hedge", long_delta_qty=0.2), 3000.0
    )

    assert lane["adapter"].orders == []
