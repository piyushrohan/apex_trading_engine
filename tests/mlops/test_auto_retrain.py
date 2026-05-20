import numpy as np
import pandas as pd
import pytest

from src.mlops import auto_retrain
from src.mlops.auto_retrain import AutoRetrainPipeline


class FakeRegistry:
    def __init__(self):
        self.registered = []
        self.shadow_promotions = []
        self.metrics_updates = []
        self.status_updates = []

    def register_model(self, model_id, model_type, metrics):
        self.registered.append((model_id, model_type, metrics))
        return f"/tmp/{model_id}"

    def promote_to_shadow(self, model_id):
        self.shadow_promotions.append(model_id)

    def update_model_metrics(self, model_id, metrics):
        self.metrics_updates.append((model_id, metrics))

    def set_model_status(self, model_id, status):
        self.status_updates.append((model_id, status))


class FakeCache:
    def __init__(self, db_path):
        self.db_path = db_path
        self.closed = False

    def load_ohlcv(self, symbol, interval):
        timestamps = pd.date_range("2026-05-16", periods=80, freq="3min")
        close = [3000 + i + ((-1) ** i * 2) for i in range(80)]
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": symbol,
                "timeframe": interval,
                "open": close,
                "high": [price + 5 for price in close],
                "low": [price - 5 for price in close],
                "close": close,
                "volume": [100 + i for i in range(80)],
            }
        )

    def close(self):
        self.closed = True


class FakeUUID:
    hex = "abcdef123456"


@pytest.mark.mlops
def test_auto_retrain_registers_and_promotes_safe_candidate(mock_config, monkeypatch):
    """Verify nightly retrain promotes a candidate that passes safety gates."""

    class PassingEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_oos(self, pnl, trades):
            return {"passed_safety": True, "sharpe": 2.0}

    monkeypatch.setattr(auto_retrain, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(auto_retrain, "DuckDBCacheManager", FakeCache)
    monkeypatch.setattr(auto_retrain, "ModelEvaluator", PassingEvaluator)
    monkeypatch.setattr(auto_retrain.uuid, "uuid4", lambda: FakeUUID())

    pipeline = AutoRetrainPipeline(mock_config)
    result = pipeline.execute_nightly_retrain()

    assert pipeline.registry.registered[0][0] == "gbm_ethusdc_vabcdef12"
    assert pipeline.registry.shadow_promotions == ["gbm_ethusdc_vabcdef12"]
    assert pipeline.registry.metrics_updates[0][0] == "gbm_ethusdc_vabcdef12"
    assert "walk_forward" in pipeline.registry.metrics_updates[0][1]
    assert pipeline.cache.closed is True
    assert result["status"] == "completed"


@pytest.mark.mlops
def test_auto_retrain_keeps_failed_candidate_out_of_shadow(mock_config, monkeypatch):
    """Verify nightly retrain does not promote a candidate that fails safety gates."""

    class FailingEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_oos(self, pnl, trades):
            return {"passed_safety": False, "sharpe": 0.2}

    monkeypatch.setattr(auto_retrain, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(auto_retrain, "DuckDBCacheManager", FakeCache)
    monkeypatch.setattr(auto_retrain, "ModelEvaluator", FailingEvaluator)
    monkeypatch.setattr(auto_retrain.uuid, "uuid4", lambda: FakeUUID())

    pipeline = AutoRetrainPipeline(mock_config)
    pipeline.execute_nightly_retrain()

    assert pipeline.registry.registered
    assert pipeline.registry.shadow_promotions == []
    assert pipeline.registry.status_updates[-1][1] == "REJECTED"
    assert pipeline.cache.closed is True


@pytest.mark.mlops
def test_required_walk_forward_gate_blocks_shadow_promotion(mock_config, monkeypatch):
    """Verify required walk-forward evidence can block an otherwise safe model."""

    class MixedEvaluator:
        def __init__(self, config):
            self.config = config
            self.calls = 0

        def evaluate_oos(self, pnl, trades):
            self.calls += 1
            passed = self.calls == 1
            return {"passed_safety": passed, "sharpe": 2.0 if passed else 0.1}

        def evaluate_stress(self, pnl, trades):
            return {"stress_passed": True}

    config = {
        **mock_config,
        "mlops": {
            "candidate_model_type": "GBM",
            "min_training_rows": 30,
            "walk_forward": {
                "enabled": True,
                "required": True,
                "folds": 2,
                "min_test_rows": 5,
                "min_pass_rate": 1.0,
            },
        },
    }
    monkeypatch.setattr(auto_retrain, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(auto_retrain, "DuckDBCacheManager", FakeCache)
    monkeypatch.setattr(auto_retrain, "ModelEvaluator", MixedEvaluator)
    monkeypatch.setattr(auto_retrain.uuid, "uuid4", lambda: FakeUUID())

    pipeline = AutoRetrainPipeline(config)
    result = pipeline.execute_nightly_retrain()

    walk_forward = result["metrics"]["walk_forward"]
    assert walk_forward["required"] is True
    assert walk_forward["passed"] is False
    assert pipeline.registry.shadow_promotions == []
    assert pipeline.registry.status_updates[-1][1] == "REJECTED"


@pytest.mark.mlops
def test_supervised_dataset_keeps_future_returns_out_of_features(mock_config):
    """Ensure next-bar returns are labels only, never direct feature inputs."""
    closes = [100.0, 101.0, 99.0, 102.0, 101.0, 103.0]
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-20", periods=len(closes), freq="3min"),
            "symbol": "ETHUSDC",
            "timeframe": "3m",
            "open": closes,
            "high": [price + 1 for price in closes],
            "low": [price - 1 for price in closes],
            "close": closes,
            "volume": [100, 110, 105, 120, 115, 130],
        }
    )
    pipeline = AutoRetrainPipeline.__new__(AutoRetrainPipeline)
    pipeline.config = {**mock_config, "mlops": {"label_return_threshold": 0.005}}

    dataset = pipeline._build_supervised_dataset(raw)
    close = raw["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    future_returns = close.shift(-1).sub(close).div(close).fillna(0.0)

    assert dataset["feature_3"].tolist() == (returns > 0.005).astype(float).tolist()
    assert dataset["feature_4"].tolist() == (returns < -0.005).astype(float).tolist()
    assert np.allclose(
        dataset["feature_6"],
        returns.rolling(5, min_periods=1).mean(),
    )
    assert (
        dataset["feature_3"].tolist() != (future_returns > 0.005).astype(float).tolist()
    )
