from src.strategies.hedge.base import HedgeContext, HedgeProposal


class FundingBiasHedgeStrategy:
    name = "funding_bias_hedge"

    def __init__(self, config: dict):
        cfg = config.get("hedge", {}).get("strategies", {}).get(self.name, {})
        self.funding_extreme = float(cfg.get("funding_extreme", 0.0003))
        self.hedge_fraction = float(cfg.get("hedge_fraction", 0.2))

    def score(self, ctx: HedgeContext) -> float:
        funding_abs = abs(ctx.funding_rate)
        if funding_abs < self.funding_extreme:
            return 0.08

        score = 0.58 + min(
            (funding_abs - self.funding_extreme) / self.funding_extreme * 0.15,
            0.22,
        )
        if self._mild_trend_agreement(ctx):
            score += 0.08
        return min(score, 0.9)

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        size = ctx.primary_size_fraction * self.hedge_fraction
        long_delta = short_delta = 0.0
        if ctx.funding_rate > 0:
            short_delta = size
            intent = "funding_extreme_bias_short_with_tactical_hedge"
        else:
            long_delta = size
            intent = "funding_extreme_bias_long_with_tactical_hedge"
        return HedgeProposal(
            strategy_name=self.name,
            long_delta_qty=long_delta,
            short_delta_qty=short_delta,
            intent=intent,
        )

    def _mild_trend_agreement(self, ctx: HedgeContext) -> bool:
        if abs(ctx.trend_slope) > 0.01:
            return False
        return (ctx.funding_rate > 0 and ctx.primary_action == 0) or (
            ctx.funding_rate < 0 and ctx.primary_action == 2
        )
