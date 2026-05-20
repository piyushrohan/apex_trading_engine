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


@pytest.mark.mlops
def test_market_narrative_confidence_and_portfolio_summary_edges(mock_config):
    engine = ExplainabilityEngine(mock_config)

    high = engine.compute_confidence_buckets(
        [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
        action=2,
        regime="STRONG_TREND_UP",
        conviction=0.8,
    )
    rich_narrative = engine.build_market_narrative(
        "STRONG_TREND_UP",
        {
            "is_sell_liquidity_sweep": True,
            "eth_btc_zscore": 1.8,
            "volatility_zscore": 1.2,
            "funding_rate": 0.0003,
        },
    )
    cheap_narrative = engine.build_market_narrative(
        "MEAN_REVERSION",
        {"eth_btc_zscore": -1.9, "volatility_zscore": -1.2},
    )

    assert high["summary"]["confidence_tier"] == "high"
    assert any("Sell-side" in line for line in rich_narrative)
    assert any("ETH rich" in line for line in rich_narrative)
    assert any("Volatility expansion" in line for line in rich_narrative)
    assert any("Funding rate elevated" in line for line in rich_narrative)
    assert any("ETH cheap" in line for line in cheap_narrative)
    assert any("Volatility compression" in line for line in cheap_narrative)
    assert engine._portfolio_summary(0.0, 0.0).startswith("Flat")
    assert engine._portfolio_summary(0.25, 0.0).startswith("Net long")
    assert engine._portfolio_summary(0.0, 0.5).startswith("Net short")


@pytest.mark.mlops
def test_read_latest_journal_entry_empty_missing_and_malformed(mock_config, tmp_path):
    engine = ExplainabilityEngine(mock_config)
    empty = tmp_path / "empty.jsonl"
    malformed = tmp_path / "malformed.jsonl"
    empty.write_text("\n", encoding="utf-8")
    malformed.write_text('{"decision": "LONG"}\n{bad-json', encoding="utf-8")

    assert engine.read_latest_journal_entry(str(tmp_path / "missing.jsonl")) is None
    assert engine.read_latest_journal_entry(str(empty)) is None
    assert engine.read_latest_journal_entry(str(malformed)) is None
