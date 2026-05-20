import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Offline Backtest Evaluator.
    Determines if a newly trained model meets the stringent safety and performance
    criteria required to be promoted to SHADOW or PROD.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_free_rate = 0.0  # Assuming 0% for intraday crypto

        eval_cfg = config.get("mlops", {}).get("evaluation", {})
        self.min_sharpe = eval_cfg.get("min_sharpe", 1.5)
        self.max_drawdown = eval_cfg.get("max_drawdown", 0.10)
        self.min_trades = eval_cfg.get("min_trades", 50)
        self.min_profit_factor = eval_cfg.get("min_profit_factor", 1.0)
        self.min_win_rate = eval_cfg.get("min_win_rate", 0.0)
        stress_cfg = config.get("mlops", {}).get("stress", {})
        self.stress_cost_bps = stress_cfg.get("cost_bps", 4.0)
        self.stress_max_drawdown = stress_cfg.get("max_drawdown", self.max_drawdown)
        self.stress_min_return = stress_cfg.get("min_return", 0.0)

    def evaluate_oos(
        self, pnl_series: pd.Series, trade_history: List[dict]
    ) -> Dict[str, Any]:
        """
        Evaluates an Out-Of-Sample (OOS) equity curve.
        pnl_series: Pandas Series of sequential account balances or cumulative PnL.
        """
        metrics = {}

        if len(trade_history) < self.min_trades:
            logger.warning(
                f"Not enough trades for robust evaluation: {len(trade_history)}"
            )
            metrics["sharpe"] = 0.0
            metrics["max_drawdown"] = 1.0
            metrics["total_trades"] = len(trade_history)
            metrics["passed_safety"] = False
            return metrics

        # Calculate returns
        returns = pnl_series.pct_change().dropna()

        # Annualized Sharpe (Assuming series is 3m intervals)
        intervals_per_year = (365 * 24 * 60) / 3
        if returns.std() != 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(intervals_per_year)
        else:
            sharpe = 0.0

        # Max Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())

        # Win Rate
        winning_trades = [t for t in trade_history if t.get("pnl", 0) > 0]
        losing_trades = [t for t in trade_history if t.get("pnl", 0) < 0]
        win_rate = len(winning_trades) / len(trade_history)
        gains = sum(float(t.get("pnl", 0)) for t in winning_trades)
        losses = abs(sum(float(t.get("pnl", 0)) for t in losing_trades))
        profit_factor = gains / losses if losses > 0 else 999.0
        downside = returns[returns < 0]
        sortino = 0.0
        if len(downside) and downside.std() != 0:
            sortino = (returns.mean() / downside.std()) * np.sqrt(intervals_per_year)
        calmar = 0.0
        if max_dd > 0:
            calmar = float(returns.mean() * intervals_per_year / max_dd)

        metrics = {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "total_trades": len(trade_history),
        }

        # Safety Gate Logic
        passed = (
            metrics["sharpe"] >= self.min_sharpe
            and metrics["max_drawdown"] <= self.max_drawdown
            and metrics["profit_factor"] >= self.min_profit_factor
            and metrics["win_rate"] >= self.min_win_rate
        )

        metrics["passed_safety"] = passed

        if passed:
            logger.info(
                f"Model PASSED evaluation: Sharpe {sharpe:.2f}, MaxDD {max_dd:.2%}"
            )
        else:
            logger.warning(
                f"Model FAILED evaluation: Sharpe {sharpe:.2f}, MaxDD {max_dd:.2%}"
            )

        return metrics

    def evaluate_stress(
        self, pnl_series: pd.Series, trade_history: List[dict]
    ) -> Dict[str, Any]:
        """Apply a simple transaction-cost stress to the OOS equity curve."""
        if pnl_series.empty or len(pnl_series) < 2:
            return {
                "stress_passed": False,
                "reason": "insufficient_equity_series",
                "cost_bps": self.stress_cost_bps,
            }

        values = pnl_series.astype(float).copy()
        initial_equity = max(float(values.iloc[0]), 1e-9)
        total_cost = (
            initial_equity * (self.stress_cost_bps / 10000.0) * len(trade_history)
        )
        stressed_values = values.copy()
        if len(stressed_values) > 1:
            cost_step = total_cost / (len(stressed_values) - 1)
            stressed_values.iloc[1:] = (
                stressed_values.iloc[1:]
                - np.arange(1, len(stressed_values)) * cost_step
            )

        returns = stressed_values.pct_change().dropna()
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = abs(float(drawdown.min())) if not drawdown.empty else 1.0
        stressed_return = float(
            (stressed_values.iloc[-1] - stressed_values.iloc[0])
            / max(abs(stressed_values.iloc[0]), 1e-9)
        )
        passed = (
            max_dd <= self.stress_max_drawdown
            and stressed_return >= self.stress_min_return
        )
        return {
            "stress_passed": passed,
            "stressed_return": stressed_return,
            "stressed_max_drawdown": max_dd,
            "cost_bps": self.stress_cost_bps,
        }
