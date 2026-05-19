import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.strategies.hedge.base import HedgeContext


class ContextualBanditSelector:
    """
    LinUCB selector for hedge strategy arms.

    Activation is gated by per-strategy decision counts from historical logs.
    """

    def __init__(self, config: dict, strategy_names: List[str]):
        hedge_cfg = config.get("hedge", {})
        bandit_cfg = hedge_cfg.get("bandit", {})

        self.strategy_names = list(strategy_names)
        self.alpha = float(bandit_cfg.get("exploration_factor", 0.1))
        self.min_decisions = int(bandit_cfg.get("min_decisions", 500))
        self.state_path = Path(
            bandit_cfg.get("state_path", "data_lake/hedge_bandit/state.json")
        )
        self.decision_log_path = Path(
            config.get("shadow", {}).get(
                "decision_log_path",
                "data_lake/hedge_bandit/training/decisions.jsonl",
            )
        )

        self.dim = 6
        self.counts: Dict[str, int] = {name: 0 for name in self.strategy_names}
        self.total_observations = 0
        self._a: Dict[str, np.ndarray] = {
            name: np.identity(self.dim, dtype=float) for name in self.strategy_names
        }
        self._b: Dict[str, np.ndarray] = {
            name: np.zeros(self.dim, dtype=float) for name in self.strategy_names
        }

        self._load_state()
        self._bootstrap_counts_from_decisions()

    def is_eligible(self) -> bool:
        if not self.strategy_names:
            return False
        return all(
            self.counts.get(name, 0) >= self.min_decisions
            for name in self.strategy_names
        )

    def select_arm(
        self, ctx: HedgeContext, rule_scores: Dict[str, float]
    ) -> Tuple[Optional[str], Dict[str, float], bool]:
        if not rule_scores:
            return None, {}, False

        x = self._context_vector(ctx)
        ucb_scores: Dict[str, float] = {}
        exploration = False

        for name in self.strategy_names:
            if name not in rule_scores:
                continue
            a_inv = np.linalg.inv(self._a[name])
            theta = a_inv.dot(self._b[name])
            exploit = float(theta.dot(x))
            bonus = self.alpha * math.sqrt(float(x.T.dot(a_inv).dot(x)))
            ucb_scores[name] = exploit + bonus

        if not ucb_scores:
            return None, {}, False

        best_name = max(ucb_scores, key=ucb_scores.get)
        top = sorted(ucb_scores.items(), key=lambda kv: kv[1], reverse=True)
        if len(top) > 1 and abs(top[0][1] - top[1][1]) < 1e-6:
            exploration = True
        return best_name, ucb_scores, exploration

    def record_reward(self, arm: str, ctx: HedgeContext, reward: float) -> None:
        if arm not in self._a:
            return
        x = self._context_vector(ctx)
        self._a[arm] = self._a[arm] + np.outer(x, x)
        self._b[arm] = self._b[arm] + (reward * x)
        self.counts[arm] = self.counts.get(arm, 0) + 1
        self._save_state()

    def update_from_reward_log(self) -> int:
        """
        Train bandit state from decision rows containing hedge.selected and
        hedge_reward. This keeps the selector off by default until history exists.
        """
        if not self.decision_log_path.exists():
            return 0
        updates = 0
        for line in self.decision_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            hedge = row.get("hedge") or {}
            arm = hedge.get("selected") or hedge.get("bandit_arm")
            if arm not in self._a or row.get("hedge_reward") is None:
                continue
            ctx = self._context_from_row(row)
            self.record_reward(arm, ctx, float(row["hedge_reward"]))
            updates += 1
        return updates

    def _context_vector(self, ctx: HedgeContext) -> np.ndarray:
        disagreement = 0.0
        if ctx.ppo_action_probs and ctx.gbm_action_probs:
            ppo = int(
                max(
                    range(len(ctx.ppo_action_probs)),
                    key=lambda i: ctx.ppo_action_probs[i],
                )
            )
            gbm = int(
                max(
                    range(len(ctx.gbm_action_probs)),
                    key=lambda i: ctx.gbm_action_probs[i],
                )
            )
            disagreement = 1.0 if ppo != gbm else 0.0

        regime_hash = (abs(hash(ctx.regime)) % 1000) / 1000.0
        vol_bucket = max(-2.0, min(2.0, float(ctx.volatility_zscore))) / 2.0
        funding_bucket = max(-3.0, min(3.0, float(ctx.funding_rate) * 10000.0)) / 3.0
        primary_side = (
            1.0 if ctx.primary_action == 2 else -1.0 if ctx.primary_action == 0 else 0.0
        )
        return np.array(
            [1.0, regime_hash, vol_bucket, funding_bucket, disagreement, primary_side],
            dtype=float,
        )

    def _context_from_row(self, row: Dict[str, Any]) -> HedgeContext:
        ctx = row.get("bandit_context") or {}
        return HedgeContext(
            symbol=row.get("symbol", "ETHUSDC"),
            regime=row.get("regime", "MEAN_REVERSION"),
            mark_price=float(row.get("mark_price", 0.0) or 0.0),
            feature_vector=[],
            ppo_action_probs=list(ctx.get("ppo_action_probs") or []),
            gbm_action_probs=list(ctx.get("gbm_action_probs") or []),
            primary_action=int(ctx.get("primary_action", row.get("action", 1))),
            primary_size_fraction=float(ctx.get("primary_size_fraction", 0.0)),
            volatility_zscore=float(ctx.get("volatility_zscore", 0.0)),
            funding_rate=float(ctx.get("funding_rate", 0.0)),
        )

    def _bootstrap_counts_from_decisions(self) -> None:
        if not self.decision_log_path.exists():
            return
        for line in self.decision_log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            hedge = row.get("hedge") or {}
            for name in (hedge.get("candidates") or {}).keys():
                if name in self.counts:
                    self.counts[name] += 1
            self.total_observations += 1

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        counts = state.get("counts", {})
        for name in self.strategy_names:
            self.counts[name] = int(counts.get(name, self.counts.get(name, 0)))

        a_map = state.get("a", {})
        b_map = state.get("b", {})
        for name in self.strategy_names:
            if name in a_map:
                self._a[name] = np.array(a_map[name], dtype=float)
            if name in b_map:
                self._b[name] = np.array(b_map[name], dtype=float)

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "counts": self.counts,
            "a": {name: self._a[name].tolist() for name in self.strategy_names},
            "b": {name: self._b[name].tolist() for name in self.strategy_names},
        }
        self.state_path.write_text(json.dumps(payload), encoding="utf-8")
