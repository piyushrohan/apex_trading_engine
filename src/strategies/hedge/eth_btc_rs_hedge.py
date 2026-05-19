from src.strategies.hedge.base import HedgeContext, HedgeProposal


class EthBtcRelativeStrengthHedgeStrategy:
    name = "eth_btc_rs_hedge"

    def __init__(self, config: dict):
        cfg = config.get("hedge", {}).get("strategies", {}).get(self.name, {})
        self.zscore_entry = float(cfg.get("zscore_entry", 1.5))
        self.hedge_fraction = float(cfg.get("hedge_fraction", 0.25))

    def score(self, ctx: HedgeContext) -> float:
        z = abs(ctx.eth_btc_zscore)
        if z < self.zscore_entry:
            return 0.1

        rs_action = 2 if ctx.eth_btc_zscore > 0 else 0
        conflicts_primary = (
            ctx.primary_action in (0, 2) and ctx.primary_action != rs_action
        )
        conflicts_flow = self._flow_conflicts_relative_strength(ctx)

        score = 0.55 + min((z - self.zscore_entry) * 0.12, 0.25)
        if conflicts_primary:
            score += 0.12
        if conflicts_flow:
            score += 0.08
        return min(score, 0.92)

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        size = ctx.primary_size_fraction * self.hedge_fraction
        long_delta = short_delta = 0.0

        if ctx.eth_btc_zscore > 0:
            long_delta = size
            intent = "relative_strength_bias_long_with_tactical_hedge"
        else:
            short_delta = size
            intent = "relative_strength_bias_short_with_tactical_hedge"

        if ctx.primary_action == 2:
            short_delta = max(short_delta, size)
        elif ctx.primary_action == 0:
            long_delta = max(long_delta, size)

        return HedgeProposal(
            strategy_name=self.name,
            long_delta_qty=long_delta,
            short_delta_qty=short_delta,
            intent=intent,
        )

    @staticmethod
    def _flow_conflicts_relative_strength(ctx: HedgeContext) -> bool:
        cvd = float(ctx.extra.get("cvd", 0.0))
        return (ctx.eth_btc_zscore > 0 and cvd < 0) or (
            ctx.eth_btc_zscore < 0 and cvd > 0
        )
