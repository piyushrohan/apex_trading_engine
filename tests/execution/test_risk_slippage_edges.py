import pytest

from src.execution.risk_engine import RiskEngine
from src.execution.slippage import SlippageManager


@pytest.mark.risk
def test_risk_engine_updates_high_water_mark_and_rejects_no_edge(mock_config):
    """Verify equity recovery updates high-water mark and no-edge Kelly returns zero."""
    engine = RiskEngine(mock_config)

    engine.update_equity(11_000.0)
    no_edge = engine.calculate_kelly_size(
        win_rate=0.30, win_loss_ratio=1.0, confidence=0.9
    )
    invalid_edge = engine.calculate_kelly_size(
        win_rate=0.55, win_loss_ratio=0.0, confidence=0.9
    )

    assert engine.high_water_mark == 11_000.0
    assert engine.is_kill_switch_active is False
    assert no_edge == 0.0
    assert invalid_edge == 0.0


@pytest.mark.unit
def test_slippage_manager_sell_side_and_front_of_queue_paths():
    """Verify SELL-side distance math and no-chase front-of-queue guard."""
    manager = SlippageManager(chase_tolerance_ticks=3, tick_size=0.01)

    assert manager.should_chase_order(3000.00, 3000.01, "SELL", 0.5) is True
    assert manager.should_chase_order(3000.02, 3000.01, "SELL", 0.9) is False
    assert manager.should_chase_order(3000.00, 3000.10, "SELL", 0.5) is False
    assert manager.should_chase_order(3000.00, 3000.10, "SELL", 0.95) is True
