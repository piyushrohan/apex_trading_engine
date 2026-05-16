import pytest
import os
from src.mlops.explainability import ExplainabilityEngine

def test_explainability_no_silent_trades(mock_config, tmp_path):
    """
    Verify that every trade decision includes reasoning, confidence decomposition,
    and never executes a 'silent' trade without logs.
    """
    engine = ExplainabilityEngine(mock_config)
    
    # Override trade journal path to use a temporary pytest directory
    test_journal = tmp_path / "test_journal.jsonl"
    engine.trade_journal_path = str(test_journal)
    
    # Mock context from MetaController
    # Feature 3 is Liquidity Sweep (Buy-Side) -> high positive weight
    # Feature 8 is Volatility Expansion -> negative weight opposing conviction
    mock_context = {
        "active_regime": "VOLATILITY_EXPANSION",
        "selected_by_meta": "PPO",
        "action_probs": [0.1, 0.2, 0.7],
        "feature_contributions": [0.0] * 10 
    }
    mock_context["feature_contributions"][3] = 1.25  # High bullish conviction
    mock_context["feature_contributions"][8] = -0.40 # Moderate bearish conviction
    
    action = 2 # LONG
    conviction = 0.7
    
    explanation = engine.decode_decision(action, conviction, mock_context)
    
    assert explanation["decision"] == "LONG"
    assert explanation["conviction_score"] == 0.7
    assert explanation["active_regime"] == "VOLATILITY_EXPANSION"
    
    # Verify Reasons
    assert any("Liquidity Sweep (Buy-Side)" in r for r in explanation["primary_reasons"])
    assert any("bullishly" in r for r in explanation["primary_reasons"])
    
    # Verify Risk Factors
    assert any("Volatility Expansion Z-Score" in r for r in explanation["risk_factors"])
    
    # Verify File I/O
    assert os.path.exists(str(test_journal))
    with open(str(test_journal), "r") as f:
        content = f.read()
        assert "Liquidity Sweep (Buy-Side)" in content
