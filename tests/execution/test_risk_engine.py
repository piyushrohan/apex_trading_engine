import pytest

from src.execution.risk_engine import RiskEngine


@pytest.mark.risk
def test_risk_engine_kelly_sizing(mock_config):
    """Verify Kelly Criterion calculates optimal fraction accurately."""
    engine = RiskEngine(mock_config)

    # 55% win rate, 1.2 W/L ratio
    # Kelly = W - ((1-W)/R) = 0.55 - (0.45 / 1.2) = 0.55 - 0.375 = 0.175
    # Half-Kelly (config fraction 0.5) = 0.0875

    fraction = engine.calculate_kelly_size(
        win_rate=0.55, win_loss_ratio=1.2, confidence=0.8
    )

    assert round(fraction, 4) == 0.14


@pytest.mark.risk
def test_risk_engine_kill_switch(mock_config):
    """
    Risk Catastrophe Test: Verify that exceeding the maximum drawdown
    forces position sizing to 0, stopping the bleeding immediately.
    """
    engine = RiskEngine(mock_config)

    # Simulate a 15% drawdown (Config max is 10%)
    engine.update_equity(current_equity=8500.0)

    # The Kill Switch must override everything at order approval
    fraction = engine.calculate_kelly_size(
        win_rate=0.80, win_loss_ratio=2.0, confidence=0.99
    )
    approved = engine.approve_order(
        proposed_side="BUY", proposed_fraction=fraction, current_exposure=0.0
    )

    assert approved == 0.0


@pytest.mark.risk
def test_risk_engine_leverage_cap(mock_config):
    """Verify approve_order respects the hard max_leverage constraint."""
    engine = RiskEngine(mock_config)

    # Attempt to take a position representing 4x leverage
    # Config max_leverage is 3
    approved = engine.approve_order(
        proposed_side="BUY", proposed_fraction=4.0, current_exposure=0.0
    )

    assert approved == 3.0  # Capped at max_leverage

    # Attempt to add to a position that is already maxed
    approved_additional = engine.approve_order(
        proposed_side="BUY", proposed_fraction=1.0, current_exposure=3.0
    )

    assert approved_additional == 0.0  # No more leverage allowed


@pytest.mark.risk
def test_risk_engine_equity_and_kelly_guard_edges(mock_config):
    mock_config["environment"]["initial_capital"] = 0.0
    engine = RiskEngine(mock_config)

    engine.update_equity(current_equity=-1.0)
    assert engine.is_kill_switch_active is False
    assert engine.calculate_kelly_size(0.0, 1.0, 1.0) == 0.0
    assert engine.calculate_kelly_size(0.4, 1.0, 1.0) == 0.0
    assert engine.project_hedge_leverages(1.0, 1.0, 0.0, 3500.0) == (
        0.0,
        0.0,
        0.0,
    )


@pytest.mark.risk
def test_risk_engine_hedge_limit_rejections(mock_config):
    mock_config["execution"].update(
        {
            "position_mode": "hedge",
            "max_gross_leverage": 2.0,
            "max_net_leverage": 1.0,
            "max_hedge_ratio": 0.4,
        }
    )
    engine = RiskEngine(mock_config)

    ok, reason = engine.check_hedge_limits(
        long_qty=3.0,
        short_qty=0.0,
        equity=1000.0,
        mark_price=500.0,
    )
    assert ok is False
    assert "net leverage" in reason

    ok, reason = engine.check_hedge_limits(
        long_qty=1.0,
        short_qty=0.5,
        equity=1000.0,
        mark_price=500.0,
        is_hedge_leg=True,
    )
    assert ok is False
    assert "hedge ratio" in reason

    ok, reason = engine.check_hedge_limits(
        long_qty=1.0,
        short_qty=0.2,
        equity=1000.0,
        mark_price=500.0,
        is_hedge_leg=True,
    )
    assert ok is True
    assert reason == ""


@pytest.mark.risk
def test_risk_engine_approve_order_rejects_sell_when_hedge_limits_fail(mock_config):
    mock_config["execution"].update(
        {
            "position_mode": "hedge",
            "max_gross_leverage": 2.0,
            "max_net_leverage": 0.5,
        }
    )
    engine = RiskEngine(mock_config)

    approved = engine.approve_order(
        proposed_side="SELL",
        proposed_fraction=1.0,
        current_exposure=0.0,
        long_qty=0.0,
        short_qty=0.0,
        equity=1000.0,
        mark_price=1000.0,
    )

    assert approved == 0.0
