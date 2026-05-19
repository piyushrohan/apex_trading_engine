"""Position lifecycle state for explainability (why open, why flat, invalidation)."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PositionLifecycleTracker:
    """
    Tracks primary-book position state and emits human-readable lifecycle events.
    """

    state: str = "FLAT"
    side: Optional[str] = None
    opened_at: Optional[str] = None
    last_event: Optional[str] = None
    invalidation_rules: List[str] = field(default_factory=list)

    def update(
        self,
        *,
        action: int,
        long_qty: float,
        short_qty: float,
        regime: str,
        risk_factors: List[str],
        conviction: float,
        kill_switch: bool,
        timestamp: str,
        min_conviction_flat: float = 0.35,
        max_risk_factors_open: int = 3,
    ) -> Dict[str, Any]:
        prev_state = self.state
        net = long_qty - short_qty

        if kill_switch:
            self.invalidation_rules = ["kill_switch_engaged"]
            self.state = "FLAT"
            self.side = None
            self.last_event = "invalidated_kill_switch"
        elif long_qty > 0 and short_qty > 0:
            self.state = "HEDGED"
            self.side = "BOTH"
            self.last_event = "hedged_dual_leg"
        elif long_qty > 0:
            self.state = "LONG_OPEN"
            self.side = "LONG"
            if prev_state != "LONG_OPEN":
                self.opened_at = timestamp
                self.last_event = "opened_long"
        elif short_qty > 0:
            self.state = "SHORT_OPEN"
            self.side = "SHORT"
            if prev_state != "SHORT_OPEN":
                self.opened_at = timestamp
                self.last_event = "opened_short"
        else:
            if prev_state in ("LONG_OPEN", "SHORT_OPEN", "HEDGED"):
                self.last_event = "closed_flat"
            self.state = "FLAT"
            self.side = None
            self.opened_at = None

        why_flat = self._why_flat(
            action, conviction, risk_factors, regime, min_conviction_flat
        )
        why_open = self._why_open(self.state, action, regime, conviction)
        invalidation = self._invalidation(
            risk_factors, kill_switch, max_risk_factors_open
        )

        return {
            "state": self.state,
            "side": self.side,
            "opened_at": self.opened_at,
            "last_event": self.last_event,
            "why_flat": why_flat,
            "why_open": why_open,
            "invalidation": invalidation,
            "net_qty": net,
        }

    @staticmethod
    def _why_flat(
        action: int,
        conviction: float,
        risk_factors: List[str],
        regime: str,
        min_conviction: float,
    ) -> Optional[str]:
        if action != 1:
            return None
        if conviction < min_conviction:
            return (
                f"Model chose FLAT — conviction {conviction:.2f} below "
                f"threshold {min_conviction:.2f}"
            )
        if len(risk_factors) >= 2:
            return (
                f"Model chose FLAT — {len(risk_factors)} opposing risk factors "
                f"({risk_factors[0]})"
            )
        if regime in ("CHOP_COMPRESSION", "MEAN_REVERSION"):
            return f"Model chose FLAT — {regime} regime favors reduced exposure"
        return "Model chose FLAT — no directional edge above internal threshold"

    @staticmethod
    def _why_open(
        state: str, action: int, regime: str, conviction: float
    ) -> Optional[str]:
        if state == "FLAT":
            return None
        action_str = {0: "SHORT", 2: "LONG"}.get(action, "HOLD")
        return (
            f"Position {state.replace('_OPEN', '')} — signal {action_str} "
            f"in {regime} (conviction {conviction:.2f})"
        )

    @staticmethod
    def _invalidation(
        risk_factors: List[str],
        kill_switch: bool,
        max_risk: int,
    ) -> List[str]:
        rules = []
        if kill_switch:
            rules.append("kill_switch: flatten all legs immediately")
        if len(risk_factors) >= max_risk:
            rules.append(
                f"risk_factor_count: {len(risk_factors)} >= {max_risk} "
                "— consider reducing exposure"
            )
        for rf in risk_factors[:2]:
            rules.append(f"opposing_feature: {rf}")
        return rules
