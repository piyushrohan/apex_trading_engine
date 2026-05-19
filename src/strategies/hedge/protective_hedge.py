from src.strategies.hedge.base import HedgeContext, HedgeProposal


class ProtectiveHedgeStrategy:
    name = "protective_hedge"

    def __init__(self, config: dict):
        cfg = config.get("hedge", {}).get("strategies", {}).get("protective_hedge", {})
        self.min_risk_factors = cfg.get("min_risk_factors", 2)
        self.hedge_fraction = cfg.get("hedge_fraction", 0.25)

    def score(self, ctx: HedgeContext) -> float:
        has_position = ctx.primary_long_qty > 0 or ctx.primary_short_qty > 0
        if not has_position:
            return 0.0
        n = len(ctx.risk_factors)
        if n >= self.min_risk_factors:
            return min(0.65 + 0.1 * (n - self.min_risk_factors), 0.92)
        return 0.2

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        hedge_size = ctx.primary_size_fraction * self.hedge_fraction
        long_delta = short_delta = 0.0
        if ctx.primary_long_qty > 0:
            short_delta = hedge_size
        elif ctx.primary_short_qty > 0:
            long_delta = hedge_size
        return HedgeProposal(
            strategy_name=self.name,
            long_delta_qty=long_delta,
            short_delta_qty=short_delta,
            intent="protective_insurance_against_risk_factors",
        )
