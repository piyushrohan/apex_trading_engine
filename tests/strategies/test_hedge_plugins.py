import pytest

from src.strategies.hedge.base import HedgeContext
from src.strategies.hedge.protective_hedge import ProtectiveHedgeStrategy
from src.strategies.hedge.regime_straddle import RegimeStraddleStrategy
from src.strategies.hedge.registry import build_hedge_orchestrator
from src.strategies.hedge.signal_disagreement import SignalDisagreementStrategy


def _ctx(**kwargs) -> HedgeContext:
    defaults = dict(
        symbol="ETHUSDC",
        regime="CHOP_COMPRESSION",
        mark_price=3500.0,
        feature_vector=[0.0] * 10,
        ppo_action_probs=[0.1, 0.2, 0.7],
        gbm_action_probs=[0.7, 0.2, 0.1],
        risk_factors=[],
        primary_size_fraction=0.2,
    )
    defaults.update(kwargs)
    return HedgeContext(**defaults)


@pytest.mark.unit
def test_signal_disagreement_high_when_models_disagree(mock_config):
    strat = SignalDisagreementStrategy(mock_config)
    score = strat.score(_ctx())
    assert score >= 0.4
    proposal = strat.propose(_ctx())
    assert proposal.strategy_name == "signal_disagreement"


@pytest.mark.unit
def test_signal_disagreement_low_when_models_agree(mock_config):
    strat = SignalDisagreementStrategy(mock_config)
    score = strat.score(
        _ctx(ppo_action_probs=[0.1, 0.2, 0.7], gbm_action_probs=[0.1, 0.2, 0.7])
    )
    assert score <= 0.2


@pytest.mark.unit
def test_regime_straddle_scores_compression(mock_config):
    strat = RegimeStraddleStrategy(mock_config)
    assert strat.score(_ctx(regime="CHOP_COMPRESSION", volatility_zscore=-1.5)) >= 0.8
    proposal = strat.propose(_ctx())
    assert proposal.long_delta_qty > 0 and proposal.short_delta_qty > 0


@pytest.mark.unit
def test_protective_hedge_requires_position_and_risk_factors(mock_config):
    strat = ProtectiveHedgeStrategy(mock_config)
    assert strat.score(_ctx(primary_long_qty=0.0)) == 0.0
    high = strat.score(
        _ctx(
            primary_long_qty=0.5,
            risk_factors=["a", "b", "c"],
        )
    )
    assert high >= 0.65


@pytest.mark.unit
def test_hedge_orchestrator_journal_payload(mock_config):
    config = dict(mock_config)
    config["hedge"] = {
        "enabled": True,
        "min_score": 0.3,
        "strategies": {
            "signal_disagreement": {"enabled": True},
            "regime_straddle": {"enabled": True},
            "protective_hedge": {"enabled": True},
        },
    }
    orch = build_hedge_orchestrator(config)
    proposal, payload = orch.evaluate(_ctx())
    assert payload["enabled"] is True
    assert "candidates" in payload
    assert len(payload["candidates"]) == 3
    if proposal:
        assert payload["selected"] in payload["candidates"]
