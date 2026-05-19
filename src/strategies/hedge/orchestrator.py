import logging
from typing import Any, Dict, Optional, Tuple

from src.strategies.hedge.base import HedgeContext, HedgeProposal
from src.strategies.hedge.selector import RuleBasedHedgeSelector

logger = logging.getLogger(__name__)


class HedgeOrchestrator:
    """Single writer for hedge proposals and journal payloads."""

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("hedge", {}).get("enabled", False)
        selection = config.get("hedge", {}).get("selection", "rule_based")
        if selection != "rule_based":
            logger.warning(
                "Only rule_based hedge selection is implemented; "
                f"falling back from {selection}"
            )
        self.selector = RuleBasedHedgeSelector(config)

    def register_strategy(self, strategy) -> None:
        self.selector.register(strategy)

    def register(self, strategy) -> None:
        """Alias used by hedge plugin registry."""
        self.register_strategy(strategy)

    def evaluate(
        self, ctx: HedgeContext
    ) -> Tuple[Optional[HedgeProposal], Dict[str, Any]]:
        if not self.enabled:
            return None, {"enabled": False}

        selected, scores, proposal = self.selector.select(ctx)
        payload: Dict[str, Any] = {
            "enabled": True,
            "selection_mode": "rule_based",
            "selected": selected,
            "candidates": scores,
        }
        if proposal:
            payload["proposal"] = {
                "long_delta_qty": proposal.long_delta_qty,
                "short_delta_qty": proposal.short_delta_qty,
                "intent": proposal.intent,
            }
            payload["selected_score"] = proposal.score
        return proposal, payload
