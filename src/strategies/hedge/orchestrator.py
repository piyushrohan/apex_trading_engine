import logging
from typing import Any, Dict, Optional, Tuple

from src.strategies.hedge.bandit_selector import ContextualBanditSelector
from src.strategies.hedge.base import HedgeContext, HedgeProposal
from src.strategies.hedge.selector import RuleBasedHedgeSelector

logger = logging.getLogger(__name__)


class HedgeOrchestrator:
    """Single writer for hedge proposals and journal payloads."""

    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("hedge", {}).get("enabled", False)
        self.selection_mode = config.get("hedge", {}).get("selection", "rule_based")
        self.selector = RuleBasedHedgeSelector(config)
        self.bandit_selector: Optional[ContextualBanditSelector] = None

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

        scores = self.selector.score_all(ctx)
        selected: Optional[str] = None
        proposal: Optional[HedgeProposal] = None
        bandit_active = False
        payload: Dict[str, Any] = {
            "enabled": True,
            "selection_mode": "rule_based",
            "selected": None,
            "candidates": scores,
        }

        if self.selection_mode == "contextual_bandit":
            if self.bandit_selector is None:
                self.bandit_selector = ContextualBanditSelector(
                    self.config, self.selector.strategy_names()
                )

            if self.bandit_selector.is_eligible():
                bandit_active = True
                selected, bandit_scores, exploration = self.bandit_selector.select_arm(
                    ctx, scores
                )
                payload["selection_mode"] = "contextual_bandit"
                payload["bandit_arm"] = selected
                payload["exploration"] = exploration
                payload["candidates_rule_shadow"] = scores
                payload["candidates"] = bandit_scores
                if selected and scores.get(selected, 0.0) >= self.selector.min_score:
                    strategy = self.selector.get_strategy(selected)
                    if strategy is not None:
                        proposal = strategy.propose(ctx)
                        proposal.score = scores[selected]
            else:
                logger.info(
                    "Contextual bandit requested but activation gate is not met; "
                    "falling back to rule_based selection."
                )

        if proposal is None and not bandit_active:
            selected, _, proposal = self.selector.select(ctx)
            payload["selected"] = selected
        else:
            payload["selected"] = selected

        if selected is None:
            payload["selected"] = None

        if proposal:
            payload["proposal"] = {
                "long_delta_qty": proposal.long_delta_qty,
                "short_delta_qty": proposal.short_delta_qty,
                "intent": proposal.intent,
            }
            payload["selected_score"] = proposal.score
        return proposal, payload
