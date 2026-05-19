import pytest

from src.mlops.explainability import ExplainabilityEngine


@pytest.mark.mlops
def test_confidence_buckets_decomposition(mock_config):
    engine = ExplainabilityEngine(mock_config)
    contributions = [0.0] * 10
    contributions[0] = 0.8
    contributions[3] = 1.2
    contributions[9] = 0.5
    contributions[8] = -0.3

    buckets = engine.compute_confidence_buckets(
        contributions, action=2, regime="STRONG_TREND_UP", conviction=0.72
    )

    assert "momentum" in buckets
    assert "liquidity" in buckets
    assert "trend" in buckets
    assert "regime" in buckets
    assert buckets["regime"]["active_regime"] == "STRONG_TREND_UP"
    assert buckets["summary"]["confidence_tier"] in (
        "high",
        "medium",
        "low",
        "insufficient",
    )


@pytest.mark.mlops
def test_decode_decision_includes_lifecycle_and_buckets(mock_config, tmp_path):
    engine = ExplainabilityEngine(mock_config)
    engine.trade_journal_path = str(tmp_path / "journal.jsonl")
    context = {
        "active_regime": "CHOP_COMPRESSION",
        "selected_by_meta": "GBM",
        "action_probs": [0.1, 0.8, 0.1],
        "feature_contributions": [0.0] * 10,
    }

    flat = engine.decode_decision(
        1,
        0.25,
        context,
        portfolio={"long_qty": 0.0, "short_qty": 0.0},
        market_snapshot={"is_buy_liquidity_sweep": True},
    )

    assert flat["schema_version"] == 2
    assert "confidence_buckets" in flat
    assert flat["position_lifecycle"]["why_flat"] is not None
    assert any("sweep" in n.lower() for n in flat["market_narrative"])


@pytest.mark.mlops
def test_decode_portfolio_state_after_sync(mock_config):
    engine = ExplainabilityEngine(mock_config)
    payload = engine.decode_portfolio_state(
        symbol="ETHUSDC",
        operator_mode="live",
        book={"long_qty": 0.5, "short_qty": 0.1, "equity": 1200.0},
        regime="MEAN_REVERSION",
        mark_price=3500.0,
    )
    assert payload["event"] == "portfolio_sync"
    assert "Hedged" in payload["summary"]
