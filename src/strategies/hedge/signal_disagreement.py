from src.strategies.hedge.base import HedgeContext, HedgeProposal


class SignalDisagreementStrategy:
    name = "signal_disagreement"

    def __init__(self, config: dict):
        hedge = config.get("hedge", {})
        self.max_hedge_ratio = hedge.get("max_hedge_ratio", 0.35)

    def _actions(self, probs: list) -> int:
        if not probs or len(probs) < 3:
            return 1
        return int(max(range(len(probs)), key=lambda i: probs[i]))

    def score(self, ctx: HedgeContext) -> float:
        ppo_a = self._actions(ctx.ppo_action_probs)
        gbm_a = self._actions(ctx.gbm_action_probs)
        if ppo_a == gbm_a:
            return 0.15
        if ctx.regime in ("STRONG_TREND_UP", "STRONG_TREND_DOWN"):
            return 0.25
        if ctx.regime in ("CHOP_COMPRESSION", "MEAN_REVERSION"):
            ppo_c = max(ctx.ppo_action_probs) if ctx.ppo_action_probs else 0
            gbm_c = max(ctx.gbm_action_probs) if ctx.gbm_action_probs else 0
            if ppo_c > 0.45 and gbm_c > 0.45:
                return min(0.55 + abs(ppo_a - gbm_a) * 0.15, 0.9)
        return 0.4

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        ppo_a = self._actions(ctx.ppo_action_probs)
        gbm_a = self._actions(ctx.gbm_action_probs)
        primary = (
            ppo_a
            if max(ctx.ppo_action_probs or [0]) >= max(ctx.gbm_action_probs or [0])
            else gbm_a
        )
        hedge_action = gbm_a if primary == ppo_a else ppo_a
        base = ctx.primary_size_fraction * self.max_hedge_ratio
        long_delta = short_delta = 0.0
        if hedge_action == 2:
            long_delta = base
        elif hedge_action == 0:
            short_delta = base
        return HedgeProposal(
            strategy_name=self.name,
            long_delta_qty=long_delta,
            short_delta_qty=short_delta,
            intent="hedge_model_disagreement",
        )
