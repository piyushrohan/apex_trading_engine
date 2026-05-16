import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RiskEngine:
    """
    Institutional Risk Engine.
    Operates strictly independently of the AI models. 
    Enforces drawdown limits, computes dynamic Kelly position sizes, 
    and handles the emergency kill switch.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk parameters
        exec_config = config.get('execution', {})
        self.max_leverage = exec_config.get('max_leverage', 3.0)
        self.kelly_fraction_cap = exec_config.get('kelly_fraction_cap', 0.3)
        self.max_daily_drawdown = exec_config.get('max_daily_drawdown', 0.05) # 5% default
        
        self.initial_equity = config.get('environment', {}).get('initial_capital', 1000.0)
        self.current_equity = self.initial_equity
        self.high_water_mark = self.initial_equity
        
        self.is_kill_switch_active = False

    def update_equity(self, current_equity: float):
        """Updates internal equity tracking and checks for kill switch conditions."""
        self.current_equity = current_equity
        
        if self.current_equity > self.high_water_mark:
            self.high_water_mark = self.current_equity
            
        drawdown = (self.high_water_mark - self.current_equity) / self.high_water_mark
        
        if drawdown >= self.max_daily_drawdown and not self.is_kill_switch_active:
            logger.critical(f"KILL SWITCH ENGAGED! Drawdown {drawdown:.2%} exceeded max {self.max_daily_drawdown:.2%}")
            self.is_kill_switch_active = True

    def calculate_kelly_size(self, win_rate: float, win_loss_ratio: float, confidence: float) -> float:
        """
        Calculates optimal position size fraction using the Kelly Criterion.
        win_rate: historical win rate of the current regime/model.
        win_loss_ratio: Average Win Size / Average Loss Size.
        confidence: AI's conviction score (0.0 to 1.0).
        """
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0.0
            
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        if kelly_pct <= 0:
            return 0.0 # No edge, no bet
            
        # Scale by AI confidence
        adjusted_kelly = kelly_pct * confidence
        
        # Cap the Kelly fraction (Half-Kelly or custom cap)
        final_fraction = min(adjusted_kelly, self.kelly_fraction_cap)
        
        return final_fraction

    def approve_order(self, proposed_side: str, proposed_fraction: float, current_exposure: float) -> float:
        """
        Takes a proposed order size fraction and applies risk limits.
        Returns the approved fraction of equity to deploy.
        """
        if self.is_kill_switch_active:
            logger.warning("Order rejected. Kill switch is active.")
            return 0.0
            
        # Ensure leverage limits are respected
        available_fraction = self.max_leverage - current_exposure
        if available_fraction <= 0:
            logger.warning(f"Order rejected. Max leverage ({self.max_leverage}x) reached.")
            return 0.0
            
        approved_fraction = min(proposed_fraction, available_fraction)
        
        return approved_fraction
