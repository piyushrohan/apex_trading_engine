import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskEngine:
    """
      Institutional Risk Engine.
      Operates strictly independently of the AI models.
      Enforces drawdown limits, computes dynamic Kelly position sizes,
      and handles the emergency kill switch.
    In hedge mode enforces gross/net leverage and hedge-ratio caps.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        exec_config = config.get("execution", {})
        risk_config = config.get("risk", {})
        self.max_leverage = exec_config.get(
            "max_leverage", risk_config.get("max_leverage", 3.0)
        )
        self.kelly_fraction_cap = exec_config.get(
            "kelly_fraction_cap", risk_config.get("kelly_fraction_cap", 0.3)
        )
        self.max_daily_drawdown = exec_config.get(
            "max_daily_drawdown", risk_config.get("max_daily_drawdown", 0.05)
        )
        self.max_gross_leverage = exec_config.get(
            "max_gross_leverage", risk_config.get("max_gross_leverage", 2.0)
        )
        self.max_net_leverage = exec_config.get(
            "max_net_leverage", risk_config.get("max_net_leverage", 1.0)
        )
        self.max_hedge_ratio = exec_config.get(
            "max_hedge_ratio", risk_config.get("max_hedge_ratio", 0.35)
        )
        self.position_mode = exec_config.get("position_mode", "one_way")
        self.risk_profile = risk_config.get("profile", "balanced")

        self.initial_equity = config.get("environment", {}).get(
            "initial_capital", 1000.0
        )
        self.current_equity = self.initial_equity
        self.high_water_mark = self.initial_equity

        self.is_kill_switch_active = False

    def update_equity(self, current_equity: float):
        """Updates internal equity tracking and checks for kill switch conditions."""
        self.current_equity = current_equity

        if self.current_equity > self.high_water_mark:
            self.high_water_mark = self.current_equity

        if self.high_water_mark <= 0:
            return

        drawdown = (self.high_water_mark - self.current_equity) / self.high_water_mark

        if drawdown >= self.max_daily_drawdown and not self.is_kill_switch_active:
            logger.critical(
                f"KILL SWITCH ENGAGED! Drawdown {drawdown:.2%} exceeded "
                f"max {self.max_daily_drawdown:.2%}"
            )
            self.is_kill_switch_active = True

    def calculate_kelly_size(
        self, win_rate: float, win_loss_ratio: float, confidence: float
    ) -> float:
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0.0

        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)

        if kelly_pct <= 0:
            return 0.0

        adjusted_kelly = kelly_pct * confidence
        return min(adjusted_kelly, self.kelly_fraction_cap)

    def project_hedge_leverages(
        self,
        long_qty: float,
        short_qty: float,
        equity: float,
        mark_price: float,
        add_long: float = 0.0,
        add_short: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Returns (gross_lev, net_lev, hedge_ratio) after proposed deltas."""
        if equity <= 0 or mark_price <= 0:
            return 0.0, 0.0, 0.0
        lq = max(long_qty + add_long, 0.0)
        sq = max(short_qty + add_short, 0.0)
        gross = (lq + sq) * mark_price / equity
        net = abs(lq - sq) * mark_price / equity
        hedge_ratio = min(lq, sq) / max(lq, sq) if lq > 0 and sq > 0 else 0.0
        return gross, net, hedge_ratio

    def check_hedge_limits(
        self,
        long_qty: float,
        short_qty: float,
        equity: float,
        mark_price: float,
        add_long: float = 0.0,
        add_short: float = 0.0,
        is_hedge_leg: bool = False,
    ) -> Tuple[bool, str]:
        gross, net, hedge_ratio = self.project_hedge_leverages(
            long_qty, short_qty, equity, mark_price, add_long, add_short
        )
        if gross > self.max_gross_leverage + 1e-9:
            return False, (
                f"gross leverage {gross:.2f}x would exceed "
                f"max {self.max_gross_leverage:.2f}x"
            )
        if net > self.max_net_leverage + 1e-9:
            return False, (
                f"net leverage {net:.2f}x would exceed "
                f"max {self.max_net_leverage:.2f}x"
            )
        if is_hedge_leg and hedge_ratio > self.max_hedge_ratio + 1e-9:
            return False, (
                f"hedge ratio {hedge_ratio:.2f} would exceed "
                f"max {self.max_hedge_ratio:.2f}"
            )
        return True, ""

    def approve_order(
        self,
        proposed_side: str,
        proposed_fraction: float,
        current_exposure: float,
        *,
        long_qty: float = 0.0,
        short_qty: float = 0.0,
        equity: Optional[float] = None,
        mark_price: float = 0.0,
        is_hedge_leg: bool = False,
    ) -> float:
        """
        Takes a proposed order size fraction and applies risk limits.
        Returns the approved fraction of equity to deploy.
        """
        if self.is_kill_switch_active:
            logger.warning("Order rejected. Kill switch is active.")
            return 0.0

        available_fraction = self.max_leverage - current_exposure
        if available_fraction <= 0:
            logger.warning(
                f"Order rejected. Max leverage ({self.max_leverage}x) reached."
            )
            return 0.0

        approved_fraction = min(proposed_fraction, available_fraction)

        if (
            self.position_mode == "hedge"
            and equity
            and equity > 0
            and mark_price > 0
            and approved_fraction > 0
        ):
            notional = approved_fraction * equity
            add_long = add_short = 0.0
            if proposed_side.upper() == "BUY":
                add_long = notional / mark_price
            else:
                add_short = notional / mark_price
            ok, reason = self.check_hedge_limits(
                long_qty,
                short_qty,
                equity,
                mark_price,
                add_long=add_long,
                add_short=add_short,
                is_hedge_leg=is_hedge_leg,
            )
            if not ok:
                logger.warning(f"Order rejected (hedge limits): {reason}")
                return 0.0

        return approved_fraction
