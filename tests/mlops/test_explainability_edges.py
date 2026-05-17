import json

import pytest

from src.mlops.explainability import ExplainabilityEngine


@pytest.mark.mlops
def test_explainability_decodes_bearish_reasons_and_risks(mock_config, tmp_path):
    """Verify SHORT explanations separate bearish support from bullish opposition."""
    engine = ExplainabilityEngine(mock_config)
    engine.trade_journal_path = str(tmp_path / "journal.jsonl")
    context = {
        "active_regime": "MEAN_REVERSION",
        "selected_by_meta": "GBM",
        "feature_contributions": [-0.9, 0.7, 0.2],
        "action_probs": [0.8, 0.1, 0.1],
    }

    explanation = engine.decode_decision(0, 0.81234, context)

    journal_line = (tmp_path / "journal.jsonl").read_text().strip()
    assert explanation["decision"] == "SHORT"
    assert explanation["conviction_score"] == 0.8123
    assert "aligned bearishly" in explanation["primary_reasons"][0]
    assert "opposing bearish conviction" in explanation["risk_factors"][0]
    assert json.loads(journal_line)["decision"] == "SHORT"


@pytest.mark.mlops
def test_explainability_handles_unknown_action_and_journal_write_failure(
    mock_config, tmp_path
):
    """Verify unknown actions are represented and journal failures are swallowed."""
    engine = ExplainabilityEngine(mock_config)
    engine.trade_journal_path = str(tmp_path / "missing" / "journal.jsonl")

    explanation = engine.decode_decision(
        99,
        0.1,
        {"feature_contributions": [0.5], "action_probs": [0.2, 0.3, 0.5]},
    )

    assert explanation["decision"] == "UNKNOWN"
    assert explanation["primary_reasons"] == []
    assert explanation["risk_factors"] == []
