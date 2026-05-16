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
