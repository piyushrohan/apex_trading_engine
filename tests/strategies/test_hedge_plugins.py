import pytest

from src.strategies.hedge.base import HedgeContext
from src.strategies.hedge.eth_btc_rs_hedge import EthBtcRelativeStrengthHedgeStrategy
from src.strategies.hedge.funding_bias_hedge import FundingBiasHedgeStrategy
from src.strategies.hedge.maker_grid_hedge import MakerGridHedgeStrategy
from src.strategies.hedge.protective_hedge import ProtectiveHedgeStrategy
from src.strategies.hedge.regime_straddle import RegimeStraddleStrategy
from src.strategies.hedge.registry import build_hedge_orchestrator
from src.strategies.hedge.signal_disagreement import SignalDisagreementStrategy
from src.strategies.hedge.sweep_dual_leg import SweepDualLegStrategy


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


@pytest.mark.unit
def test_relative_strength_hedge_scores_high_zscore(mock_config):
    strat = EthBtcRelativeStrengthHedgeStrategy(mock_config)
    ctx = _ctx(
        eth_btc_zscore=2.1,
        primary_action=0,
        extra={"cvd": -100.0},
    )
    assert strat.score(ctx) >= 0.55
    proposal = strat.propose(ctx)
    assert proposal.strategy_name == "eth_btc_rs_hedge"
    assert proposal.long_delta_qty > 0


@pytest.mark.unit
def test_sweep_dual_leg_requires_sweep_and_flow_conflict(mock_config):
    strat = SweepDualLegStrategy(mock_config)
    assert strat.score(_ctx()) < 0.1
    ctx = _ctx(is_buy_liquidity_sweep=True, extra={"cvd": -10.0})
    assert strat.score(ctx) >= 0.8
    proposal = strat.propose(ctx)
    assert proposal.long_delta_qty > 0 and proposal.short_delta_qty > 0


@pytest.mark.unit
def test_funding_bias_hedge_scores_extreme_funding(mock_config):
    strat = FundingBiasHedgeStrategy(mock_config)
    assert strat.score(_ctx(funding_rate=0.0)) < 0.1
    ctx = _ctx(funding_rate=0.0006, primary_action=0, trend_slope=0.0)
    assert strat.score(ctx) >= 0.6
    proposal = strat.propose(ctx)
    assert proposal.short_delta_qty > 0


@pytest.mark.unit
def test_maker_grid_scores_chop_tight_market(mock_config):
    strat = MakerGridHedgeStrategy(mock_config)
    ctx = _ctx(
        regime="CHOP_COMPRESSION",
        volatility_zscore=-0.2,
        trend_slope=0.0,
        extra={"spread_bps": 1.0},
    )
    assert strat.score(ctx) >= 0.6
    proposal = strat.propose(ctx)
    assert proposal.long_delta_qty > 0 and proposal.short_delta_qty > 0


@pytest.mark.unit
def test_registry_registers_all_seven_hedge_plugins(mock_config):
    config = dict(mock_config)
    config["hedge"] = {
        "enabled": True,
        "min_score": 0.99,
        "strategies": {
            "signal_disagreement": {"enabled": True},
            "regime_straddle": {"enabled": True},
            "protective_hedge": {"enabled": True},
            "eth_btc_rs_hedge": {"enabled": True},
            "sweep_dual_leg": {"enabled": True},
            "maker_grid_hedge": {"enabled": True},
            "funding_bias_hedge": {"enabled": True},
        },
    }
    orch = build_hedge_orchestrator(config)
    _, payload = orch.evaluate(_ctx())
    assert set(payload["candidates"]) == {
        "signal_disagreement",
        "regime_straddle",
        "protective_hedge",
        "eth_btc_rs_hedge",
        "sweep_dual_leg",
        "maker_grid_hedge",
        "funding_bias_hedge",
    }


@pytest.mark.unit
def test_protective_hedge_low_risk_and_short_position_paths(mock_config):
    strat = ProtectiveHedgeStrategy(mock_config)
    low = _ctx(primary_long_qty=0.5, risk_factors=["spread"])
    short = _ctx(primary_short_qty=0.5, risk_factors=["spread", "sweep"])

    assert strat.score(low) == 0.2
    proposal = strat.propose(short)
    assert proposal.long_delta_qty > 0
    assert proposal.short_delta_qty == 0.0


@pytest.mark.unit
def test_maker_grid_negative_scoring_branches(mock_config):
    strat = MakerGridHedgeStrategy(mock_config)

    assert strat.score(_ctx(regime="STRONG_TREND_UP")) == 0.12
    assert strat.score(_ctx(trend_slope=0.01)) == 0.2
    assert strat.score(_ctx(extra={"spread_bps": 10.0})) == 0.25
    assert strat.score(_ctx(volatility_zscore=0.9)) == 0.45


@pytest.mark.unit
def test_signal_disagreement_missing_probs_and_directional_hedges(mock_config):
    strat = SignalDisagreementStrategy(mock_config)

    assert strat.score(_ctx(ppo_action_probs=[], gbm_action_probs=[])) == 0.15
    assert (
        strat.score(
            _ctx(
                regime="STRONG_TREND_UP",
                ppo_action_probs=[0.8, 0.1, 0.1],
                gbm_action_probs=[0.1, 0.1, 0.8],
            )
        )
        == 0.25
    )
    long_hedge = strat.propose(
        _ctx(ppo_action_probs=[0.8, 0.1, 0.1], gbm_action_probs=[0.1, 0.1, 0.8])
    )
    short_hedge = strat.propose(
        _ctx(ppo_action_probs=[0.1, 0.1, 0.8], gbm_action_probs=[0.8, 0.1, 0.1])
    )

    assert long_hedge.long_delta_qty > 0
    assert short_hedge.short_delta_qty > 0


@pytest.mark.unit
def test_funding_bias_long_side_and_strong_trend_penalty(mock_config):
    strat = FundingBiasHedgeStrategy(mock_config)
    ctx = _ctx(funding_rate=-0.0006, primary_action=2, trend_slope=0.02)

    assert strat.score(ctx) >= 0.58
    assert strat._mild_trend_agreement(ctx) is False
    proposal = strat.propose(ctx)
    assert proposal.long_delta_qty > 0
    assert proposal.short_delta_qty == 0.0
