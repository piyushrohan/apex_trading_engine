from src.strategies.hedge.base import HedgeContext, HedgeProposal


class RegimeStraddleStrategy:
    name = "regime_straddle"

    def __init__(self, config: dict):
        cfg = config.get("hedge", {}).get("strategies", {}).get("regime_straddle", {})
        self.straddle_fraction = cfg.get("straddle_size_fraction", 0.25)

    def score(self, ctx: HedgeContext) -> float:
        if ctx.regime != "CHOP_COMPRESSION":
            return 0.1
        if ctx.volatility_zscore < -1.0:
            return 0.8
        if ctx.volatility_zscore < 0:
            return 0.55
        return 0.25

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        size = ctx.primary_size_fraction * self.straddle_fraction
        return HedgeProposal(
            strategy_name=self.name,
            long_delta_qty=size,
            short_delta_qty=size,
            intent="compression_straddle_both_sides",
        )
