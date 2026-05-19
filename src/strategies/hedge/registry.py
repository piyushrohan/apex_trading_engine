"""Register enabled hedge strategy plugins from config."""

from src.strategies.hedge.eth_btc_rs_hedge import EthBtcRelativeStrengthHedgeStrategy
from src.strategies.hedge.funding_bias_hedge import FundingBiasHedgeStrategy
from src.strategies.hedge.maker_grid_hedge import MakerGridHedgeStrategy
from src.strategies.hedge.orchestrator import HedgeOrchestrator
from src.strategies.hedge.protective_hedge import ProtectiveHedgeStrategy
from src.strategies.hedge.regime_straddle import RegimeStraddleStrategy
from src.strategies.hedge.signal_disagreement import SignalDisagreementStrategy
from src.strategies.hedge.sweep_dual_leg import SweepDualLegStrategy


def build_hedge_orchestrator(config: dict) -> HedgeOrchestrator:
    orchestrator = HedgeOrchestrator(config)
    strategies_cfg = config.get("hedge", {}).get("strategies", {})
    default_enabled = not bool(strategies_cfg)

    def enabled(name: str) -> bool:
        return strategies_cfg.get(name, {}).get("enabled", default_enabled)

    if enabled("signal_disagreement"):
        orchestrator.register(SignalDisagreementStrategy(config))
    if enabled("regime_straddle"):
        orchestrator.register(RegimeStraddleStrategy(config))
    if enabled("protective_hedge"):
        orchestrator.register(ProtectiveHedgeStrategy(config))
    if enabled("eth_btc_rs_hedge"):
        orchestrator.register(EthBtcRelativeStrengthHedgeStrategy(config))
    if enabled("sweep_dual_leg"):
        orchestrator.register(SweepDualLegStrategy(config))
    if enabled("maker_grid_hedge"):
        orchestrator.register(MakerGridHedgeStrategy(config))
    if enabled("funding_bias_hedge"):
        orchestrator.register(FundingBiasHedgeStrategy(config))

    return orchestrator
