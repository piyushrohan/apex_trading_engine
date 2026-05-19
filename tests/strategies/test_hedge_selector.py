import pytest

from src.strategies.hedge.base import HedgeContext, HedgeProposal
from src.strategies.hedge.selector import RuleBasedHedgeSelector


class StubHedgeStrategy:
    def __init__(self, name: str, score: float):
        self.name = name
        self._score = score

    def score(self, ctx: HedgeContext) -> float:
        return self._score

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        return HedgeProposal(
            strategy_name=self.name,
            short_delta_qty=0.1,
            intent="test",
        )


@pytest.mark.unit
def test_rule_based_selector_picks_highest_above_min():
    config = {"hedge": {"min_score": 0.5}}
    selector = RuleBasedHedgeSelector(config)
    selector.register(StubHedgeStrategy("low", 0.3))
    selector.register(StubHedgeStrategy("high", 0.8))

    ctx = HedgeContext(
        symbol="ETHUSDC",
        regime="CHOP_COMPRESSION",
        mark_price=3500.0,
        feature_vector=[0.0] * 10,
        ppo_action_probs=[0.1, 0.2, 0.7],
        gbm_action_probs=[0.7, 0.2, 0.1],
        risk_factors=[],
    )
    name, scores, proposal = selector.select(ctx)
    assert name == "high"
    assert scores["high"] == 0.8
    assert proposal is not None


@pytest.mark.unit
def test_rule_based_selector_returns_none_below_min():
    config = {"hedge": {"min_score": 0.9}}
    selector = RuleBasedHedgeSelector(config)
    selector.register(StubHedgeStrategy("mid", 0.5))
    ctx = HedgeContext(
        symbol="ETHUSDC",
        regime="MEAN_REVERSION",
        mark_price=3500.0,
        feature_vector=[],
        ppo_action_probs=[],
        gbm_action_probs=[],
        risk_factors=[],
    )
    name, _, proposal = selector.select(ctx)
    assert name is None
    assert proposal is None
