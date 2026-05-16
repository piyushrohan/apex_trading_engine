import logging
from typing import Dict, Any, Tuple
from src.models.ppo_agent import PPOAgent
from src.models.gbm_agent import GBMAgent

logger = logging.getLogger(__name__)

class MetaController:
    """
    The orchestrator of the Intelligence Layer.
    Uses the Regime classification to route inference to the model 
    that historically performs best in that specific market environment.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ppo_agent = PPOAgent(state_dim=10, config=config) # Assuming 10 features
        self.gbm_agent = GBMAgent(config=config)
        
        # Mapping of Regime ID to Preferred Model
        # e.g., GBM handles chop better, PPO handles trends better
        self.regime_model_map = {
            "STRONG_TREND_UP": "PPO",
            "STRONG_TREND_DOWN": "PPO",
            "CHOP_COMPRESSION": "GBM",
            "VOLATILITY_EXPANSION": "PPO",
            "MEAN_REVERSION": "GBM"
        }
        
        logger.info("Initialized Meta-Controller with PPO and GBM Agents.")

    def get_action(self, state_vector: list, regime_str: str) -> Tuple[int, float, Dict[str, Any]]:
        """
        Determines the active model based on regime, executes inference,
        and returns the action and explainability payload.
        """
        preferred_model_name = self.regime_model_map.get(regime_str, "PPO")
        
        if preferred_model_name == "PPO":
            action, conviction, context = self.ppo_agent.act(state_vector)
        else:
            action, conviction, context = self.gbm_agent.act(state_vector)
            
        logger.debug(f"MetaController routed to {preferred_model_name} for regime {regime_str}. Action: {action}, Conviction: {conviction:.2f}")
        
        # Enrich context for the Explainability Engine
        context["active_regime"] = regime_str
        context["selected_by_meta"] = preferred_model_name
        
        return action, conviction, context
