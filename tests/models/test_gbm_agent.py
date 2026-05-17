import numpy as np
import pytest

from src.models.gbm_agent import GBMAgent, MockLGBMClassifier


@pytest.mark.unit
def test_mock_lgbm_classifier_outputs_probabilities_and_importance():
    """Verify the mock GBM backend exposes model-like outputs."""
    model = MockLGBMClassifier()

    probs = model.predict_proba([[0.1] * 10])
    importance = model.feature_importance()

    assert probs.tolist() == [[0.1, 0.2, 0.7]]
    assert len(importance) == 10


@pytest.mark.unit
def test_gbm_agent_returns_action_conviction_and_explainability(mock_config):
    """Verify GBM action inference returns probabilities and feature context."""
    agent = GBMAgent(mock_config)
    agent.model.feature_importance = lambda: np.arange(10)

    action, conviction, context = agent.act([1.0] * 10)

    assert action == 2
    assert conviction == 0.7
    assert context["model_type"] == "GBM"
    assert context["action_probs"] == [0.1, 0.2, 0.7]
    assert context["feature_contributions"] == list(np.arange(10, dtype=float))
