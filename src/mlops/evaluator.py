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

        # Minimum requirements for auto-promotion
        self.min_sharpe = 1.5
        self.max_drawdown = 0.10  # 10%
        self.min_trades = 50

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
        win_rate = len(winning_trades) / len(trade_history)

        metrics = {
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "total_trades": len(trade_history),
        }

        # Safety Gate Logic
        passed = (
            metrics["sharpe"] >= self.min_sharpe
            and metrics["max_drawdown"] <= self.max_drawdown
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
