import json
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """
    Translates mathematical model inferences (PPO gradients/attention or GBM SHAP values)
    into human-readable JSON payloads. This provides 'X-ray vision' into the AI's 
    decision-making process.
    """
    
    # Map index in the state vector to a human-readable feature name
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
        "Trend Slope"
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trade_journal_path = "data_lake/trade_journal.jsonl"

    def decode_decision(self, action: int, conviction: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw context from the Meta-Controller and converts it into a 
        structured explanation payload.
        """
        action_str = {0: "SHORT", 1: "FLAT", 2: "LONG"}.get(action, "UNKNOWN")
        regime = context.get("active_regime", "UNKNOWN")
        model = context.get("selected_by_meta", "UNKNOWN")
        
        feature_contributions = context.get("feature_contributions", [])
        
        # Sort features by absolute contribution magnitude to find the top drivers
        sorted_indices = np.argsort(np.abs(feature_contributions))[::-1]
        
        reasons = []
        risk_factors = []
        
        for idx in sorted_indices[:3]: # Top 3 driving features
            if idx < len(self.FEATURE_MAP):
                feature_name = self.FEATURE_MAP[idx]
                val = feature_contributions[idx]
                
                if val > 0:
                    if action == 2:
                        reasons.append(f"{feature_name} aligned bullishly (+{val:.2f})")
                    elif action == 0:
                        risk_factors.append(f"{feature_name} opposing bearish conviction (+{val:.2f})")
                elif val < 0:
                    if action == 0:
                        reasons.append(f"{feature_name} aligned bearishly ({val:.2f})")
                    elif action == 2:
                        risk_factors.append(f"{feature_name} opposing bullish conviction ({val:.2f})")
        
        explanation = {
            "timestamp": pd.Timestamp.utcnow().isoformat() if 'pd' in globals() else "NOW",
            "decision": action_str,
            "conviction_score": round(conviction, 4),
            "active_regime": regime,
            "executing_model": model,
            "primary_reasons": reasons,
            "risk_factors": risk_factors,
            "raw_action_probs": context.get("action_probs", [])
        }
        
        self._log_to_journal(explanation)
        return explanation

    def _log_to_journal(self, explanation: Dict[str, Any]):
        """Appends the explanation to the immutable trade journal."""
        try:
            with open(self.trade_journal_path, "a") as f:
                f.write(json.dumps(explanation) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to trade journal: {e}")
