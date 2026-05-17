import pytest

from src.mlops import auto_retrain
from src.mlops.auto_retrain import AutoRetrainPipeline


class FakeRegistry:
    def __init__(self):
        self.registered = []
        self.shadow_promotions = []

    def register_model(self, model_id, model_type, metrics):
        self.registered.append((model_id, model_type, metrics))
        return f"/tmp/{model_id}"

    def promote_to_shadow(self, model_id):
        self.shadow_promotions.append(model_id)


class FakeCache:
    def __init__(self, db_path):
        self.db_path = db_path
        self.closed = False

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
    pipeline.execute_nightly_retrain()

    assert pipeline.registry.registered[0][0] == "ppo_ethusdc_vabcdef12"
    assert pipeline.registry.shadow_promotions == ["ppo_ethusdc_vabcdef12"]
    assert pipeline.cache.closed is True


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
    assert pipeline.cache.closed is True
