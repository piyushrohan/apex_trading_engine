import pytest
from src.execution.slippage import SlippageManager

def test_slippage_chases_small_distance():
    """Verify that SlippageManager chases orders within tolerance."""
    manager = SlippageManager(chase_tolerance_ticks=3, tick_size=0.01)
    
    # BBO is 3000.02, our bid is 3000.00 (distance = 2 ticks)
    should_chase = manager.should_chase_order(
        current_bbo=3000.02, 
        resting_price=3000.00, 
        side="BUY", 
        conviction_score=0.5
    )
    
    assert should_chase is True

def test_slippage_abandons_large_distance_low_conviction():
    """Verify that it abandons trades when market runs away and conviction is low."""
    manager = SlippageManager(chase_tolerance_ticks=3, tick_size=0.01)
    
    # BBO is 3000.10, our bid is 3000.00 (distance = 10 ticks)
    should_chase = manager.should_chase_order(
        current_bbo=3000.10, 
        resting_price=3000.00, 
        side="BUY", 
        conviction_score=0.7 # Low conviction for a chase
    )
    
    assert should_chase is False

def test_slippage_chases_large_distance_high_conviction():
    """Verify that it chases runaway markets ONLY if conviction is exceptionally high."""
    manager = SlippageManager(chase_tolerance_ticks=3, tick_size=0.01)
    
    # Distance = 10 ticks
    should_chase = manager.should_chase_order(
        current_bbo=3000.10, 
        resting_price=3000.00, 
        side="BUY", 
        conviction_score=0.95 # High conviction
    )
    
    assert should_chase is True
