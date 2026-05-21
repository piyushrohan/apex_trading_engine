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


class MetadataRegistry(FakeRegistry):
    def __init__(self):
        super().__init__()
        self.manifests = []

    def register_model(self, model_id, model_type, metrics, metadata=None):
        self.registered.append((model_id, model_type, metrics, metadata))
        return f"/tmp/{model_id}"

    def set_model_status(self, model_id, status, actor=None, reason=None):
        self.status_updates.append((model_id, status, actor, reason))

    def write_model_manifest(self, model_id, **kwargs):
        self.manifests.append((model_id, kwargs))


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
    assert "label_quality" in pipeline.registry.metrics_updates[0][1]
    assert "classifier_quality" in pipeline.registry.metrics_updates[0][1]
    assert pipeline.registry.metrics_updates[0][1]["quality_gate"]["passed"] is True
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
def test_auto_retrain_skips_when_supervised_rows_are_too_sparse(
    mock_config, monkeypatch
):
    config = {
        **mock_config,
        "mlops": {"min_training_rows": 10, "min_supervised_rows": 500},
    }
    monkeypatch.setattr(auto_retrain, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(auto_retrain, "DuckDBCacheManager", FakeCache)

    pipeline = AutoRetrainPipeline(config)
    result = pipeline.execute_nightly_retrain()

    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_supervised_data"
    assert pipeline.registry.registered == []
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
    future_returns = close.shift(-1).sub(close).div(close).dropna()
    expected_rows = len(raw) - 1

    assert len(dataset) == expected_rows
    assert (
        dataset["feature_3"].tolist()
        == (returns.iloc[:expected_rows] > 0.005).astype(float).tolist()
    )
    assert (
        dataset["feature_4"].tolist()
        == (returns.iloc[:expected_rows] < -0.005).astype(float).tolist()
    )
    assert np.allclose(
        dataset["feature_6"],
        returns.rolling(5, min_periods=1).mean().iloc[:expected_rows],
    )
    assert (
        dataset["feature_3"].tolist() != (future_returns > 0.005).astype(float).tolist()
    )


@pytest.mark.mlops
def test_supervised_dataset_uses_horizon_and_fee_adjusted_labels(mock_config):
    closes = [100.0, 100.02, 100.04, 100.20, 100.10, 99.90]
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-20", periods=len(closes), freq="3min"),
            "symbol": "ETHUSDC",
            "timeframe": "3m",
            "open": closes,
            "high": [price + 1 for price in closes],
            "low": [price - 1 for price in closes],
            "close": closes,
            "volume": [100, 101, 102, 103, 104, 105],
        }
    )
    pipeline = AutoRetrainPipeline.__new__(AutoRetrainPipeline)
    pipeline.config = {
        **mock_config,
        "mlops": {
            "label_return_threshold": 0.0001,
            "label_horizon_bars": 2,
            "label_cost_buffer_bps": 5.0,
        },
    }

    dataset = pipeline._build_supervised_dataset(raw)

    assert len(dataset) == len(raw) - 2
    assert dataset["label_threshold"].iloc[0] == 0.0005
    assert dataset["label_horizon_bars"].iloc[0] == 2
    assert set(dataset["label"]).issubset({0, 1, 2})


@pytest.mark.mlops
def test_quality_gate_blocks_short_history_and_unstable_labels(mock_config):
    pipeline = AutoRetrainPipeline.__new__(AutoRetrainPipeline)
    pipeline.config = {
        **mock_config,
        "mlops": {
            "quality": {
                "min_history_days": 30,
                "min_directional_ratio": 0.20,
                "max_dominant_label_ratio": 0.70,
                "max_near_threshold_ratio": 0.10,
                "near_threshold_band_fraction": 0.50,
            }
        },
    }
    dataset = pd.DataFrame(
        {
            "label": [1, 1, 1, 1, 2],
            "future_return": [0.00049, 0.00048, 0.00047, 0.00046, 0.001],
            "label_threshold": [0.0005] * 5,
            "label_horizon_bars": [3] * 5,
        }
    )

    label_quality = pipeline._label_quality_report(dataset)
    gate = pipeline._quality_gate(
        data_metadata={"history_days": 7},
        label_quality=label_quality,
        classifier_quality={"passed": True, "blockers": []},
    )

    assert label_quality["passed"] is False
    assert "dominant_label_too_high" in label_quality["blockers"]
    assert "labels_too_close_to_threshold" in label_quality["blockers"]
    assert gate["passed"] is False
    assert "history_window_too_short" in gate["blockers"]
    assert "label:dominant_label_too_high" in gate["blockers"]


@pytest.mark.mlops
def test_auto_retrain_metadata_registry_path_and_manifest(mock_config, monkeypatch):
    class PassingEvaluator:
        def __init__(self, config):
            self.config = config

        def evaluate_oos(self, pnl, trades):
            return {"passed_safety": True, "sharpe": 2.0}

        def evaluate_stress(self, pnl, trades):
            return {"stress_passed": True}

    registry = MetadataRegistry()
    monkeypatch.setattr(auto_retrain, "DuckDBCacheManager", FakeCache)
    monkeypatch.setattr(auto_retrain, "ModelEvaluator", PassingEvaluator)
    monkeypatch.setattr(auto_retrain.uuid, "uuid4", lambda: FakeUUID())

    pipeline = AutoRetrainPipeline(mock_config, registry=registry)
    result = pipeline.execute_nightly_retrain()

    model_id, _, _, metadata = registry.registered[0]
    assert model_id == result["model_id"]
    assert metadata["feature_reference"]["features"]
    assert registry.status_updates[0][2:] == ("auto_retrain", "artifact_saved")
    assert registry.manifests[0][0] == model_id


@pytest.mark.mlops
def test_auto_retrain_closes_cache_and_marks_failed_run(mock_config, monkeypatch):
    monkeypatch.setattr(auto_retrain, "ModelRegistry", FakeRegistry)
    monkeypatch.setattr(auto_retrain, "DuckDBCacheManager", FakeCache)
    config = {
        **mock_config,
        "mlops": {
            "candidate_model_type": "UNKNOWN",
            "min_training_rows": 30,
        },
    }

    pipeline = AutoRetrainPipeline(config)

    with pytest.raises(ValueError, match="Unsupported candidate model type"):
        pipeline.execute_nightly_retrain()
    assert pipeline.cache.closed is True


@pytest.mark.mlops
def test_auto_retrain_helper_branches_and_quality_blockers(mock_config, monkeypatch):
    pipeline = AutoRetrainPipeline.__new__(AutoRetrainPipeline)
    pipeline.config = {
        **mock_config,
        "mlops": {
            "quality": {
                "class_balance_weights": False,
                "max_brier_score": 0.01,
                "max_expected_calibration_error": 0.01,
                "min_trade_signal_coverage": 0.9,
                "trade_probability_threshold": 0.8,
            },
            "walk_forward": {"enabled": False, "required": True},
        },
    }

    dataset = pd.DataFrame(
        {
            **{f"feature_{idx}": [float(idx), float(idx + 1)] for idx in range(10)},
            "label": [0, 2],
            "future_return": [0.001, -0.001],
            "label_threshold": [0.0005, 0.0005],
            "label_horizon_bars": [1, 1],
        }
    )

    class ActionOnlyAgent:
        def act(self, row):
            return 1, 0.2, {"action_probs": [0.2, 0.7, 0.1]}

    assert np.array_equal(pipeline._sample_weights(dataset), np.ones(2))
    empty_labels = pipeline._label_quality_report(dataset.iloc[:0])
    assert empty_labels["blockers"] == ["empty_supervised_dataset"]

    empty_classifier = pipeline._classifier_quality_report(
        ActionOnlyAgent(), dataset.iloc[:0]
    )
    assert empty_classifier["blockers"] == ["empty_oos_dataset"]

    classifier = pipeline._classifier_quality_report(ActionOnlyAgent(), dataset)
    assert classifier["passed"] is False
    assert "brier_score_too_high" in classifier["blockers"]
    assert "calibration_error_too_high" in classifier["blockers"]
    assert "trade_signal_coverage_too_low" in classifier["blockers"]

    disabled_walk_forward = pipeline._walk_forward_validate(dataset, "GBM")
    assert disabled_walk_forward["reason"] == "disabled"
    assert disabled_walk_forward["required"] is True

    monkeypatch.setattr(auto_retrain, "PPOAgent", lambda **kwargs: "ppo-agent")
    monkeypatch.setattr(auto_retrain, "GBMAgent", lambda config: "gbm-agent")
    assert pipeline._build_agent("PPO") == "ppo-agent"
    assert pipeline._build_agent("LIGHTGBM") == "gbm-agent"
    with pytest.raises(ValueError):
        pipeline._build_agent("BAD")

    pipeline.config["mlops"]["walk_forward"] = {"enabled": True}
    small = pipeline._walk_forward_validate(dataset.iloc[:3], "GBM")
    assert small["reason"] == "insufficient_fold_data"
