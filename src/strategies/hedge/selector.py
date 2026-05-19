import logging
from typing import Dict, Optional, Tuple

from src.strategies.hedge.base import HedgeContext, HedgeProposal, HedgeStrategy

logger = logging.getLogger(__name__)


class RuleBasedHedgeSelector:
    """
    Scores registered hedge strategies and selects argmax if above min_score.
    Contextual bandit selector will replace this in Milestone 9.
    """

    def __init__(self, config: dict):
        hedge_cfg = config.get("hedge", {})
        self.min_score = hedge_cfg.get("min_score", 0.5)
        self._strategies: Dict[str, HedgeStrategy] = {}

    def register(self, strategy: HedgeStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def score_all(self, ctx: HedgeContext) -> Dict[str, float]:
        scores = {}
        for name, strategy in self._strategies.items():
            try:
                scores[name] = float(strategy.score(ctx))
            except Exception as exc:
                logger.warning(f"Hedge strategy {name} score failed: {exc}")
                scores[name] = 0.0
        return scores

    def select(
        self, ctx: HedgeContext
    ) -> Tuple[Optional[str], Dict[str, float], Optional[HedgeProposal]]:
        scores = self.score_all(ctx)
        if not scores:
            return None, scores, None

        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]
        if best_score < self.min_score:
            return None, scores, None

        proposal = self._strategies[best_name].propose(ctx)
        proposal.score = best_score
        return best_name, scores, proposal
