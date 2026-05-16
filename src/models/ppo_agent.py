import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class PPOActorCritic(nn.Module):
    """
    Neural Network underlying the PPO Agent.
    Designed with a bottleneck layer to extract feature importance (attention-like mechanism)
    for the Explainability Engine.
    """
    def __init__(self, input_dim: int, action_dim: int = 3): # Actions: Short(0), Flat(1), Long(2)
        super(PPOActorCritic, self).__init__()
        
        # Shared Feature Extractor
        self.feature_layer = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        
        # Actor Head (Policy)
        self.actor_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic Head (Value)
        self.critic_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns Action Probabilities, State Value, and the Extracted Features
        (which are passed to the Explainability Engine).
        """
        features = self.relu(self.feature_layer(state))
        
        action_probs = self.actor_net(features)
        state_value = self.critic_net(features)
        
        return action_probs, state_value, features

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Trading Agent.
    Used for continuously learning market micro-structure.
    """
    def __init__(self, state_dim: int, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOActorCritic(input_dim=state_dim).to(self.device)
        
        logger.info(f"Initialized PPO Agent on {self.device}")

    def act(self, state_vector: list) -> Tuple[int, float, Dict[str, Any]]:
        """
        Infers the next action and returns conviction/explainability data.
        Returns: (action, conviction_score, explainability_context)
        """
        self.model.eval()
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value, latent_features = self.model(state_tensor)
            
        probs = action_probs.squeeze().cpu().numpy()
        action = int(probs.argmax())
        conviction = float(probs[action])
        
        # Generate context for the Explainability Engine
        # We pass the input weights multiplied by the state vector to approximate feature contribution
        input_weights = self.model.feature_layer.weight.data.cpu().numpy()
        feature_contributions = input_weights.mean(axis=0) * state_vector
        
        explainability_context = {
            "model_type": "PPO",
            "action_probs": probs.tolist(),
            "state_value": float(state_value.item()),
            "feature_contributions": feature_contributions.tolist()
        }
        
        return action, conviction, explainability_context
