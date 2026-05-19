import json

import pytest

from src.execution.portfolio import PortfolioService
from src.mlops.registry import ModelRegistry
from src.mlops.shadow_lane import ShadowLaneRunner
from src.models.gbm_agent import GBMAgent


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
