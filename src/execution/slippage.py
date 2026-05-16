import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SlippageManager:
    """
    Manages execution slippage and order chasing for Maker-Only strategies.
    Since we only use POST-ONLY orders, if the market moves away, we must 
    evaluate whether to chase the price or abandon the trade based on alpha decay.
    """
    
    def __init__(self, chase_tolerance_ticks: int = 3, tick_size: float = 0.01):
        self.chase_tolerance_ticks = chase_tolerance_ticks
        self.tick_size = tick_size

    def should_chase_order(self, current_bbo: float, resting_price: float, side: str, conviction_score: float) -> bool:
        """
        Determines if a resting order should be cancelled and replaced closer to the BBO.
        current_bbo: Best Bid (if buying) or Best Offer (if selling).
        resting_price: The price of our current open limit order.
        side: "BUY" or "SELL".
        conviction_score: 0.0 to 1.0, AI's confidence in the trade.
        """
        if side.upper() == "BUY":
            distance_ticks = (current_bbo - resting_price) / self.tick_size
        else:
            distance_ticks = (resting_price - current_bbo) / self.tick_size
            
        if distance_ticks <= 0:
            return False # We are at the front of the queue or better
            
        # If the market ran away beyond our tolerance
        if distance_ticks > self.chase_tolerance_ticks:
            logger.debug(f"Order too far behind BBO ({distance_ticks:.1f} ticks).")
            # Only chase if AI conviction is extremely high, otherwise abandon
            if conviction_score > 0.85:
                logger.info(f"High conviction ({conviction_score:.2f}). Approving chase.")
                return True
            else:
                logger.info("Alpha likely decayed. Do not chase.")
                return False
                
        # If it's within tolerance, always chase to capture the spread
        return True
