from src.strategies.hedge.base import HedgeContext, HedgeProposal


class MakerGridHedgeStrategy:
    name = "maker_grid_hedge"

    def __init__(self, config: dict):
        hedge_cfg = config.get("hedge", {})
        cfg = hedge_cfg.get("strategies", {}).get(self.name, {})
        self.grid_levels = int(cfg.get("grid_levels", 3))
        self.grid_fraction = float(cfg.get("grid_fraction", 0.15))
        self.max_spread_bps = float(cfg.get("max_spread_bps", 4.0))

    def score(self, ctx: HedgeContext) -> float:
        spread_bps = float(ctx.extra.get("spread_bps", 0.0))
        if ctx.regime != "CHOP_COMPRESSION":
            return 0.12
        if abs(ctx.trend_slope) > 0.002:
            return 0.2
        if spread_bps and spread_bps > self.max_spread_bps:
            return 0.25
        if ctx.volatility_zscore <= 0.25:
            return 0.68
        return 0.45

    def propose(self, ctx: HedgeContext) -> HedgeProposal:
        size = ctx.primary_size_fraction * self.grid_fraction
        return HedgeProposal(
            strategy_name=self.name,
            long_delta_qty=size,
            short_delta_qty=size,
            intent=f"maker_grid_inventory_hedge_{self.grid_levels}_levels",
        )
