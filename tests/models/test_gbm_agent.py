import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.models.gbm_agent import GBMAgent, MockLGBMClassifier, NumpyCentroidClassifier


@pytest.mark.unit
def test_mock_lgbm_classifier_outputs_probabilities_and_importance():
    """Verify the mock GBM backend exposes model-like outputs."""
    model = MockLGBMClassifier()

    assert model.fit([[0.1] * 10], [2]) is model
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


@pytest.mark.unit
def test_numpy_centroid_classifier_default_and_trained_paths():
    model = NumpyCentroidClassifier()

    default_probs = model.predict_proba([[0.1, 0.2]])
    default_importance = model.feature_importance()

    assert default_probs.tolist() == [[0.1, 0.2, 0.7]]
    assert np.allclose(default_importance, np.ones(10) / 10)

    model.fit([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [0, 1, 2])

    assert model.predict_proba([[0.9, 1.1]]).shape == (1, 3)
    assert np.isclose(model.feature_importance().sum(), 1.0)


@pytest.mark.unit
def test_gbm_agent_training_validation_and_lightgbm_backend(monkeypatch, mock_config):
    agent = GBMAgent(mock_config)

    with pytest.raises(ValueError, match="2D feature matrix"):
        agent.train([1.0, 2.0], [0, 1])
    with pytest.raises(ValueError, match="lengths must match"):
        agent.train([[1.0], [2.0]], [0])
    with pytest.raises(ValueError, match="empty dataset"):
        agent.train(np.empty((0, 1)), [])

    class FakeLGBM:
        def __init__(self, **params):
            self.params = params
            self.feature_importances_ = np.array([0.2, 0.8])

        def fit(self, x, y):
            self.y = np.asarray(y)
            return self

        def predict_proba(self, x):
            return np.tile(np.array([[0.8, 0.1, 0.1]]), (len(x), 1))

    monkeypatch.setitem(
        sys.modules, "lightgbm", SimpleNamespace(LGBMClassifier=FakeLGBM)
    )
    summary = agent.train([[0.0, 1.0], [1.0, 0.0]], [0, 0])

    assert summary["backend"] == "lightgbm"
    assert summary["train_accuracy"] == 1.0
    assert agent.model.params["num_class"] == 3


@pytest.mark.unit
def test_gbm_agent_feature_importance_attr_fallbacks(mock_config):
    agent = GBMAgent(mock_config)

    class WithFeatureImportances:
        feature_importances_ = np.array([0.5, 1.5])

    class WithoutImportances:
        pass

    agent.model = WithFeatureImportances()
    assert agent._feature_importance(4).tolist() == [0.5, 1.5, 0.5, 1.5]

    agent.model = WithoutImportances()
    assert agent._feature_importance(3).tolist() == [1.0, 1.0, 1.0]
