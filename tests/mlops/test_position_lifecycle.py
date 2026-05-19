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
