import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PPOActorCritic(nn.Module):
    """
    Neural Network underlying the PPO Agent.
    Designed with a bottleneck layer for feature importance extraction.
    """

    def __init__(
        self, input_dim: int, action_dim: int = 3
    ):  # Actions: Short(0), Flat(1), Long(2)
        super(PPOActorCritic, self).__init__()

        # Shared Feature Extractor
        self.feature_layer = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()

        # Actor Head (Policy)
        self.actor_net = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_dim), nn.Softmax(dim=-1)
        )

        # Critic Head (Value)
        self.critic_net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.state_dim = state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPOActorCritic(input_dim=state_dim).to(self.device)

        logger.info(f"Initialized PPO Agent on {self.device}")

    def train(self, features, actions, rewards=None) -> Dict[str, Any]:
        """
        Lightweight supervised policy warm-start.

        This is not a full PPO rollout trainer yet; it gives Milestone 6 a real
        train/save/load path for registry candidates using historical labels.
        """
        x_arr = np.asarray(features, dtype=float)
        y_arr = np.asarray(actions, dtype=int)
        if x_arr.ndim != 2:
            raise ValueError("PPOAgent.train expects a 2D feature matrix.")
        if len(x_arr) != len(y_arr):
            raise ValueError("Feature and action lengths must match.")
        if len(x_arr) == 0:
            raise ValueError("Cannot train PPO on an empty dataset.")

        cfg = self.config.get("models", {}).get("ppo", {})
        epochs = int(cfg.get("supervised_epochs", 20))
        lr = float(cfg.get("learning_rate", 1e-3))
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.NLLLoss()
        x = torch.tensor(x_arr, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_arr, dtype=torch.long, device=self.device)

        final_loss = 0.0
        for _ in range(max(epochs, 1)):
            optimizer.zero_grad()
            probs, _, _ = self.model(x)
            loss = loss_fn(torch.log(probs.clamp_min(1e-8)), y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        self.model.eval()
        with torch.no_grad():
            probs, _, _ = self.model(x)
        accuracy = float((probs.argmax(dim=1) == y).float().mean().item())
        return {
            "train_rows": int(len(x_arr)),
            "train_loss": final_loss,
            "train_accuracy": accuracy,
        }

    def save(self, model_path: str) -> str:
        """Persist the actor-critic checkpoint under a registry directory."""
        path = Path(model_path)
        path.mkdir(parents=True, exist_ok=True)
        artifact = path / "ppo_actor_critic.pt"
        torch.save(
            {
                "state_dim": self.state_dim,
                "model_state_dict": self.model.state_dict(),
            },
            artifact,
        )
        return str(artifact)

    def load(self, model_path: str):
        """Load a PPO checkpoint from a registry directory or direct file path."""
        path = Path(model_path)
        artifact = path if path.is_file() else path / "ppo_actor_critic.pt"
        checkpoint = torch.load(artifact, map_location=self.device)
        state_dim = int(checkpoint.get("state_dim", self.state_dim))
        if state_dim != self.state_dim:
            self.state_dim = state_dim
            self.model = PPOActorCritic(input_dim=state_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        return self

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
        # Approximate feature contribution from input weights and state.
        input_weights = self.model.feature_layer.weight.data.cpu().numpy()
        feature_contributions = input_weights.mean(axis=0) * state_vector

        explainability_context = {
            "model_type": "PPO",
            "action_probs": probs.tolist(),
            "state_value": float(state_value.item()),
            "feature_contributions": feature_contributions.tolist(),
        }

        return action, conviction, explainability_context
