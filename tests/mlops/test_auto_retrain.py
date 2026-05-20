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
