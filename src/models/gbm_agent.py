import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# In a real environment, this would be: import lightgbm as lgb
# Mocking for architectural blueprint purposes
class MockLGBMClassifier:
    def fit(self, x, y):
        return self

    def predict_proba(self, x):
        # Mock probabilities [Short, Flat, Long]
        return np.tile(np.array([[0.1, 0.2, 0.7]]), (len(x), 1))

    def feature_importance(self):
        return np.random.rand(10)


class NumpyCentroidClassifier:
    """Small deterministic fallback when LightGBM native libs are unavailable."""

    def __init__(self):
        self.classes_ = np.array([0, 1, 2])
        self.centroids_: Dict[int, np.ndarray] = {}
        self.importances_: np.ndarray | None = None

    def fit(self, x, y):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        for klass in self.classes_:
            rows = x_arr[y_arr == klass]
            self.centroids_[int(klass)] = (
                rows.mean(axis=0) if len(rows) else x_arr.mean(axis=0)
            )
        spread = np.std(x_arr, axis=0)
        self.importances_ = spread / (spread.sum() or 1.0)
        return self

    def predict_proba(self, x):
        x_arr = np.asarray(x, dtype=float)
        if not self.centroids_:
            return np.tile(np.array([[0.1, 0.2, 0.7]]), (len(x_arr), 1))

        distances = []
        for klass in self.classes_:
            centroid = self.centroids_[int(klass)]
            distances.append(np.linalg.norm(x_arr - centroid, axis=1))
        logits = -np.vstack(distances).T
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def feature_importance(self):
        if self.importances_ is None:
            return np.ones(10) / 10
        return self.importances_


logger = logging.getLogger(__name__)


class GBMAgent:
    """
    Gradient Boosting Machine (LightGBM/XGBoost) Agent.
    Often performs better than Deep RL in noisy, tabular financial data.
    Used as a fallback or primary model in specific regimes (e.g., Mean Reversion).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = "mock"
        self.model = MockLGBMClassifier()
        self.is_trained = False

        logger.info("Initialized GBM Agent.")

    def _build_model(self):
        gbm_cfg = self.config.get("models", {}).get("gbm", {})
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": gbm_cfg.get("n_estimators", 50),
            "learning_rate": gbm_cfg.get("learning_rate", 0.05),
            "max_depth": gbm_cfg.get("max_depth", -1),
            "random_state": gbm_cfg.get("random_state", 42),
            "verbosity": -1,
        }
        try:
            from lightgbm import LGBMClassifier

            self.backend = "lightgbm"
            return LGBMClassifier(**params)
        except Exception as exc:
            logger.warning(
                "LightGBM unavailable; using deterministic centroid fallback: %s",
                exc,
            )
            self.backend = "numpy_centroid"
            return NumpyCentroidClassifier()

    def train(self, features, labels) -> Dict[str, Any]:
        """Fit the GBM classifier on feature rows and integer actions."""
        x_arr = np.asarray(features, dtype=float)
        y_arr = np.asarray(labels, dtype=int)
        if x_arr.ndim != 2:
            raise ValueError("GBMAgent.train expects a 2D feature matrix.")
        if len(x_arr) != len(y_arr):
            raise ValueError("Feature and label lengths must match.")
        if len(x_arr) == 0:
            raise ValueError("Cannot train GBM on an empty dataset.")

        self.model = self._build_model()
        self.model.fit(x_arr, y_arr)
        self.is_trained = True
        train_probs = self.model.predict_proba(x_arr)
        accuracy = float((train_probs.argmax(axis=1) == y_arr).mean())
        return {
            "backend": self.backend,
            "train_rows": int(len(x_arr)),
            "train_accuracy": accuracy,
        }

    def save(self, model_path: str) -> str:
        """Persist the trained model under a registry artifact directory."""
        path = Path(model_path)
        path.mkdir(parents=True, exist_ok=True)
        artifact = path / "gbm_model.pkl"
        with open(artifact, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "backend": self.backend,
                    "is_trained": self.is_trained,
                },
                f,
            )
        return str(artifact)

    def load(self, model_path: str):
        """Load a GBM artifact from a registry directory or direct pickle path."""
        path = Path(model_path)
        artifact = path if path.is_file() else path / "gbm_model.pkl"
        with open(artifact, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.backend = payload.get("backend", "unknown")
        self.is_trained = payload.get("is_trained", True)
        return self

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
        feature_importance = self._feature_importance(len(state_vector))

        # Approximate feature contribution for this specific inference
        feature_contributions = feature_importance * state_vector

        explainability_context = {
            "model_type": "GBM",
            "action_probs": probs.tolist(),
            "feature_contributions": feature_contributions.tolist(),
        }

        return action, conviction, explainability_context

    def _feature_importance(self, expected_dim: int) -> np.ndarray:
        if hasattr(self.model, "feature_importance"):
            values = self.model.feature_importance()
        elif hasattr(self.model, "feature_importances_"):
            values = self.model.feature_importances_
        else:
            values = np.ones(expected_dim)

        arr = np.asarray(values, dtype=float)
        if len(arr) != expected_dim:
            arr = np.resize(arr, expected_dim)
        return arr
