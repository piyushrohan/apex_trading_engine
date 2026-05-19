import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.mlops.position_lifecycle import PositionLifecycleTracker

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Translates model inferences into human-readable JSON payloads (v2).
    Adds confidence buckets, market-structure narrative, and position lifecycle.
    """

    FEATURE_MAP = [
        "Price Momentum",
        "Volume Accumulation",
        "Cumulative Volume Delta (CVD)",
        "Liquidity Sweep (Buy-Side)",
        "Liquidity Sweep (Sell-Side)",
        "ETH/BTC Rolling Beta",
        "ETH/BTC Spread Z-Score",
        "ATR Volatility",
        "Volatility Expansion Z-Score",
        "Trend Slope",
    ]

    BUCKET_FEATURES = {
        "momentum": [0, 2],
        "liquidity": [1, 3, 4],
        "trend": [7, 9],
        "regime": [5, 6, 8],
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        explain_cfg = config.get("explainability", {})
        self.trade_journal_path = explain_cfg.get(
            "journal_path", "data_lake/trade_journal.jsonl"
        )
        self.min_conviction_flat = explain_cfg.get("min_conviction_flat", 0.35)
        self.max_risk_factors_open = explain_cfg.get("max_risk_factors_open", 3)
        self.lifecycle = PositionLifecycleTracker()

    def compute_confidence_buckets(
        self,
        feature_contributions: List[float],
        action: int,
        regime: str,
        conviction: float,
    ) -> Dict[str, Any]:
        """Decompose conviction into trend / momentum / liquidity / regime buckets."""
        contributions = list(feature_contributions or [])
        buckets: Dict[str, Any] = {}

        for bucket_name, indices in self.BUCKET_FEATURES.items():
            vals = [contributions[i] for i in indices if i < len(contributions)]
            if not vals:
                buckets[bucket_name] = {
                    "score": 0.0,
                    "alignment": "neutral",
                    "feature_count": 0,
                }
                continue
            score = float(np.mean(vals))
            buckets[bucket_name] = {
                "score": round(score, 4),
                "alignment": self._alignment_label(score, action),
                "feature_count": len(vals),
            }

        buckets["regime"]["active_regime"] = regime
        aligned = sum(
            1
            for b in ("momentum", "liquidity", "trend", "regime")
            if buckets[b].get("alignment") == "aligned"
        )
        buckets["summary"] = {
            "conviction": round(conviction, 4),
            "aligned_buckets": aligned,
            "total_buckets": 4,
            "confidence_tier": self._confidence_tier(conviction, aligned),
        }
        return buckets

    @staticmethod
    def _alignment_label(score: float, action: int) -> str:
        if action == 1:
            return "neutral"
        if action == 2:
            if score > 0.05:
                return "aligned"
            if score < -0.05:
                return "opposing"
        else:
            if score < -0.05:
                return "aligned"
            if score > 0.05:
                return "opposing"
        return "neutral"

    @staticmethod
    def _confidence_tier(conviction: float, aligned_buckets: int) -> str:
        if conviction >= 0.75 and aligned_buckets >= 3:
            return "high"
        if conviction >= 0.5 and aligned_buckets >= 2:
            return "medium"
        if conviction >= 0.35:
            return "low"
        return "insufficient"

    def build_market_narrative(
        self,
        regime: str,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Short regime / liquidity / structure lines for the payload."""
        lines = [f"Regime classified as {regime}."]
        if not snapshot:
            return lines

        if snapshot.get("is_buy_liquidity_sweep"):
            lines.append("Buy-side liquidity sweep detected on latest bar.")
        if snapshot.get("is_sell_liquidity_sweep"):
            lines.append("Sell-side liquidity sweep detected on latest bar.")

        z = snapshot.get("eth_btc_zscore", 0.0)
        if abs(z) > 1.5:
            direction = "rich" if z > 0 else "cheap"
            lines.append(f"ETH/BTC spread z-score {z:.2f} — ETH {direction} vs BTC.")

        vol_z = snapshot.get("volatility_zscore", 0.0)
        if vol_z > 1.0:
            lines.append("Volatility expansion — wider stops / smaller size advised.")
        elif vol_z < -1.0:
            lines.append("Volatility compression — mean-reversion favored.")

        funding = snapshot.get("funding_rate", 0.0)
        if funding and abs(funding) > 0.0002:
            lines.append(f"Funding rate elevated ({funding:.6f}).")

        return lines

    def decode_decision(
        self,
        action: int,
        conviction: float,
        context: Dict[str, Any],
        write_journal: bool = True,
        *,
        portfolio: Optional[Dict[str, Any]] = None,
        market_snapshot: Optional[Dict[str, Any]] = None,
        kill_switch: bool = False,
    ) -> Dict[str, Any]:
        action_str = {0: "SHORT", 1: "FLAT", 2: "LONG"}.get(action, "UNKNOWN")
        regime = context.get("active_regime", "UNKNOWN")
        model = context.get("selected_by_meta", "UNKNOWN")
        feature_contributions = context.get("feature_contributions", [])

        sorted_indices = np.argsort(np.abs(feature_contributions))[::-1]
        reasons = []
        risk_factors = []

        for idx in sorted_indices[:3]:
            if idx < len(self.FEATURE_MAP):
                feature_name = self.FEATURE_MAP[idx]
                val = feature_contributions[idx]
                if val > 0:
                    if action == 2:
                        reasons.append(f"{feature_name} aligned bullishly (+{val:.2f})")
                    elif action == 0:
                        risk_factors.append(
                            f"{feature_name} opposing bearish conviction (+{val:.2f})"
                        )
                elif val < 0:
                    if action == 0:
                        reasons.append(f"{feature_name} aligned bearishly ({val:.2f})")
                    elif action == 2:
                        risk_factors.append(
                            f"{feature_name} opposing bullish conviction ({val:.2f})"
                        )

        confidence_buckets = self.compute_confidence_buckets(
            feature_contributions, action, regime, conviction
        )
        narrative = self.build_market_narrative(regime, market_snapshot)

        lifecycle = self.lifecycle.update(
            action=action,
            long_qty=float((portfolio or {}).get("long_qty", 0.0)),
            short_qty=float((portfolio or {}).get("short_qty", 0.0)),
            regime=regime,
            risk_factors=risk_factors,
            conviction=conviction,
            kill_switch=kill_switch,
            timestamp=datetime.now(timezone.utc).isoformat(),
            min_conviction_flat=self.min_conviction_flat,
            max_risk_factors_open=self.max_risk_factors_open,
        )

        explanation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 2,
            "decision": action_str,
            "conviction_score": round(conviction, 4),
            "confidence_buckets": confidence_buckets,
            "active_regime": regime,
            "executing_model": model,
            "primary_reasons": reasons,
            "risk_factors": risk_factors,
            "market_narrative": narrative,
            "position_lifecycle": lifecycle,
            "raw_action_probs": context.get("action_probs", []),
        }

        if write_journal:
            self._log_to_journal(explanation)
        return explanation

    def decode_portfolio_state(
        self,
        *,
        symbol: str,
        operator_mode: str,
        book: Dict[str, Any],
        regime: str = "UNKNOWN",
        mark_price: float = 0.0,
        source: str = "account_sync",
    ) -> Dict[str, Any]:
        """Explain portfolio after manual intervention or account sync."""
        long_qty = float(book.get("long_qty", 0.0))
        short_qty = float(book.get("short_qty", 0.0))
        lifecycle = self.lifecycle.update(
            action=1,
            long_qty=long_qty,
            short_qty=short_qty,
            regime=regime,
            risk_factors=[],
            conviction=0.0,
            kill_switch=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": 2,
            "event": "portfolio_sync",
            "source": source,
            "symbol": symbol,
            "execution": {"mode": operator_mode},
            "portfolio": {
                "long_qty": long_qty,
                "short_qty": short_qty,
                "equity": book.get("equity"),
                "mark_price": mark_price,
            },
            "position_lifecycle": lifecycle,
            "summary": self._portfolio_summary(long_qty, short_qty),
        }

    @staticmethod
    def _portfolio_summary(long_qty: float, short_qty: float) -> str:
        if long_qty <= 0 and short_qty <= 0:
            return "Flat — no open legs on primary book."
        if long_qty > 0 and short_qty > 0:
            return f"Hedged — LONG {long_qty:.4f} and SHORT {short_qty:.4f}."
        if long_qty > 0:
            return f"Net long {long_qty:.4f} ETH."
        return f"Net short {short_qty:.4f} ETH."

    def read_latest_journal_entry(
        self, journal_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        path = journal_path or self.trade_journal_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                return None
            return json.loads(lines[-1])
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.error(f"Failed to read journal: {exc}")
            return None

    def _log_to_journal(self, explanation: Dict[str, Any]):
        try:
            with open(self.trade_journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(explanation) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to trade journal: {e}")
