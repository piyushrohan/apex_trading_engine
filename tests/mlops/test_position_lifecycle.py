import pytest

from src.mlops.position_lifecycle import PositionLifecycleTracker


@pytest.mark.unit
def test_lifecycle_open_long():
    tracker = PositionLifecycleTracker()
    out = tracker.update(
        action=2,
        long_qty=0.5,
        short_qty=0.0,
        regime="STRONG_TREND_UP",
        risk_factors=[],
        conviction=0.8,
        kill_switch=False,
        timestamp="2026-05-18T12:00:00Z",
    )
    assert out["state"] == "LONG_OPEN"
    assert out["why_open"] is not None
    assert out["why_flat"] is None


@pytest.mark.unit
def test_lifecycle_why_flat_low_conviction():
    tracker = PositionLifecycleTracker()
    out = tracker.update(
        action=1,
        long_qty=0.0,
        short_qty=0.0,
        regime="CHOP_COMPRESSION",
        risk_factors=["feat opposing"],
        conviction=0.2,
        kill_switch=False,
        timestamp="2026-05-18T12:00:00Z",
        min_conviction_flat=0.35,
    )
    assert out["state"] == "FLAT"
    assert "below" in out["why_flat"].lower()


@pytest.mark.unit
def test_lifecycle_kill_switch_invalidation():
    tracker = PositionLifecycleTracker()
    out = tracker.update(
        action=2,
        long_qty=0.2,
        short_qty=0.0,
        regime="VOLATILITY_EXPANSION",
        risk_factors=[],
        conviction=0.9,
        kill_switch=True,
        timestamp="2026-05-18T12:00:00Z",
    )
    assert out["state"] == "FLAT"
    assert any("kill_switch" in r for r in out["invalidation"])


@pytest.mark.unit
def test_lifecycle_short_hedged_and_closed_flat_transitions():
    tracker = PositionLifecycleTracker()

    short = tracker.update(
        action=0,
        long_qty=0.0,
        short_qty=0.4,
        regime="STRONG_TREND_DOWN",
        risk_factors=[],
        conviction=0.76,
        kill_switch=False,
        timestamp="2026-05-18T12:00:00Z",
    )
    hedged = tracker.update(
        action=1,
        long_qty=0.2,
        short_qty=0.4,
        regime="VOLATILITY_EXPANSION",
        risk_factors=["basis_widening"],
        conviction=0.5,
        kill_switch=False,
        timestamp="2026-05-18T12:03:00Z",
    )
    closed = tracker.update(
        action=1,
        long_qty=0.0,
        short_qty=0.0,
        regime="MEAN_REVERSION",
        risk_factors=[],
        conviction=0.6,
        kill_switch=False,
        timestamp="2026-05-18T12:06:00Z",
    )

    assert short["state"] == "SHORT_OPEN"
    assert short["side"] == "SHORT"
    assert "SHORT" in short["why_open"]
    assert hedged["state"] == "HEDGED"
    assert hedged["side"] == "BOTH"
    assert hedged["last_event"] == "hedged_dual_leg"
    assert closed["state"] == "FLAT"
    assert closed["last_event"] == "closed_flat"


@pytest.mark.unit
def test_lifecycle_flat_reasons_and_risk_count_invalidation():
    tracker = PositionLifecycleTracker()

    risk_flat = tracker.update(
        action=1,
        long_qty=0.0,
        short_qty=0.0,
        regime="VOLATILITY_EXPANSION",
        risk_factors=["spread", "sweep", "funding"],
        conviction=0.8,
        kill_switch=False,
        timestamp="2026-05-18T12:00:00Z",
        max_risk_factors_open=3,
    )
    regime_flat = tracker.update(
        action=1,
        long_qty=0.0,
        short_qty=0.0,
        regime="MEAN_REVERSION",
        risk_factors=[],
        conviction=0.8,
        kill_switch=False,
        timestamp="2026-05-18T12:03:00Z",
    )
    edge_flat = tracker.update(
        action=1,
        long_qty=0.0,
        short_qty=0.0,
        regime="STRONG_TREND_UP",
        risk_factors=[],
        conviction=0.8,
        kill_switch=False,
        timestamp="2026-05-18T12:06:00Z",
    )

    assert "opposing risk factors" in risk_flat["why_flat"]
    assert any("risk_factor_count" in rule for rule in risk_flat["invalidation"])
    assert "MEAN_REVERSION" in regime_flat["why_flat"]
    assert "no directional edge" in edge_flat["why_flat"]
