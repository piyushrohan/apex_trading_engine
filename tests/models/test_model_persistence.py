import numpy as np
import pytest

from src.models.gbm_agent import GBMAgent
from src.models.ppo_agent import PPOAgent


@pytest.mark.unit
def test_gbm_agent_train_save_and_load(mock_config, tmp_path):
    """Verify GBM candidates can train and round-trip through registry artifacts."""
    features = np.array(
        [
            [0.0] * 10,
            [0.1] * 10,
            [1.0] * 10,
            [1.1] * 10,
            [-1.0] * 10,
            [-1.1] * 10,
        ]
    )
    labels = np.array([1, 1, 2, 2, 0, 0])
    agent = GBMAgent(mock_config)

    metrics = agent.train(features, labels)
    artifact = agent.save(str(tmp_path / "gbm_candidate"))
    loaded = GBMAgent(mock_config).load(str(tmp_path / "gbm_candidate"))
    action, conviction, context = loaded.act(features[0].tolist())

    assert metrics["train_rows"] == 6
    assert artifact.endswith("gbm_model.pkl")
    assert action in (0, 1, 2)
    assert 0.0 <= conviction <= 1.0
    assert len(context["feature_contributions"]) == 10


@pytest.mark.unit
def test_ppo_agent_train_save_and_load(mock_config, tmp_path):
    """Verify PPO warm-start checkpoints can be saved and restored."""
    config = dict(mock_config)
    config["models"] = {"ppo": {"supervised_epochs": 2, "learning_rate": 0.01}}
    features = np.array([[0.0] * 10, [1.0] * 10, [-1.0] * 10], dtype=float)
    labels = np.array([1, 2, 0])
    agent = PPOAgent(state_dim=10, config=config)

    metrics = agent.train(features, labels)
    artifact = agent.save(str(tmp_path / "ppo_candidate"))
    loaded = PPOAgent(state_dim=10, config=config).load(str(tmp_path / "ppo_candidate"))
    action, conviction, context = loaded.act(features[0].tolist())

    assert metrics["train_rows"] == 3
    assert artifact.endswith("ppo_actor_critic.pt")
    assert action in (0, 1, 2)
    assert 0.0 <= conviction <= 1.0
    assert "state_value" in context
