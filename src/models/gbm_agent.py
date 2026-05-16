import logging
import numpy as np
from typing import Dict, Any, Tuple

# In a real environment, this would be: import lightgbm as lgb
# Mocking for architectural blueprint purposes
class MockLGBMClassifier:
    def predict_proba(self, x):
        # Mock probabilities [Short, Flat, Long]
        return np.array([[0.1, 0.2, 0.7]])
        
    def feature_importance(self):
        return np.random.rand(10)

logger = logging.getLogger(__name__)

class GBMAgent:
    """
    Gradient Boosting Machine (LightGBM/XGBoost) Agent.
    Often performs better than Deep RL in noisy, tabular financial data.
    Used as a fallback or primary model in specific regimes (e.g., Mean Reversion).
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = MockLGBMClassifier()
        self.is_trained = False
        
        logger.info("Initialized GBM Agent.")

    def act(self, state_vector: list) -> Tuple[int, float, Dict[str, Any]]:
        """
        Infers the next action and returns conviction/explainability data.
        Returns: (action, conviction_score, explainability_context)
        """
        state_arr = np.array([state_vector])
        probs = self.model.predict_proba(state_arr)[0]
        
        action = int(probs.argmax())
        conviction = float(probs[action])
        
        # GBM Explainability usually leverages SHAP values or Native Feature Importance
        feature_importance = self.model.feature_importance()
        
        # Approximate feature contribution for this specific inference
        feature_contributions = feature_importance * state_vector
        
        explainability_context = {
            "model_type": "GBM",
            "action_probs": probs.tolist(),
            "feature_contributions": feature_contributions.tolist()
        }
        
        return action, conviction, explainability_context
