"""Register enabled hedge strategy plugins from config."""

from src.strategies.hedge.orchestrator import HedgeOrchestrator
from src.strategies.hedge.protective_hedge import ProtectiveHedgeStrategy
from src.strategies.hedge.regime_straddle import RegimeStraddleStrategy
from src.strategies.hedge.signal_disagreement import SignalDisagreementStrategy


def build_hedge_orchestrator(config: dict) -> HedgeOrchestrator:
    orchestrator = HedgeOrchestrator(config)
    strategies_cfg = config.get("hedge", {}).get("strategies", {})

    if strategies_cfg.get("signal_disagreement", {}).get("enabled", True):
        orchestrator.register(SignalDisagreementStrategy(config))
    if strategies_cfg.get("regime_straddle", {}).get("enabled", True):
        orchestrator.register(RegimeStraddleStrategy(config))
    if strategies_cfg.get("protective_hedge", {}).get("enabled", True):
        orchestrator.register(ProtectiveHedgeStrategy(config))

    return orchestrator
