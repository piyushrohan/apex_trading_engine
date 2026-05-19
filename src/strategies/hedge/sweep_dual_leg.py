from src.strategies.hedge.base import HedgeContext, HedgeProposal


class SweepDualLegStrategy:
    name = "sweep_dual_leg"

    def __init__(self, config: dict):
        cfg = config.get("hedge", {}).get("strategies", {}).get(self.name, {})
        self.leg_fraction = float(cfg.get("leg_fraction", 0.2))

    def score(self, ctx: HedgeContext) -> float:
        if not (ctx.is_buy_liquidity_sweep or ctx.is_sell_liquidity_sweep):
            return 0.05

        cvd = float(ctx.extra.get("cvd", 0.0))
        buy_sweep_conflict = ctx.is_buy_liquidity_sweep and cvd < 0
        sell_sweep_conflict = ctx.is_sell_liquidity_sweep and cvd > 0
        if buy_sweep_conflict or sell_sweep_conflict:
            return 0.82
        return 0.48

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        size = ctx.primary_size_fraction * self.leg_fraction
        return HedgeProposal(
            strategy_name=self.name,
            long_delta_qty=size,
            short_delta_qty=size,
            intent="dual_leg_after_liquidity_sweep_until_flow_confirms",
        )
