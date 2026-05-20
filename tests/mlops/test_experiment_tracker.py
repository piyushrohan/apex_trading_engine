import pytest

from src.mlops.experiment_tracker import ExperimentTracker, stable_hash
from src.mlops.lifecycle import ModelLifecycleOrchestrator
from src.mlops.registry import ModelRegistry


@pytest.mark.mlops
def test_experiment_tracker_records_steps_and_completion(tmp_path):
    tracker = ExperimentTracker(str(tmp_path / "experiments.jsonl"))

    run = tracker.start_run("candidate_retrain", metadata={"config_hash": "abc"})
    tracker.log_step(run["run_id"], "train", "PASSED", metrics={"loss": 0.1})
    tracker.complete_run(
        run["run_id"],
        "COMPLETED",
        model_id="model-v1",
        metrics={"sharpe": 1.8},
    )

    rows = tracker.list_runs()

    assert len(rows) == 1
    assert rows[0]["status"] == "COMPLETED"
    assert rows[0]["model_id"] == "model-v1"
    assert rows[0]["steps"][0]["step"] == "train"
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


@pytest.mark.mlops
def test_lifecycle_orchestrator_records_promotion_review(tmp_path, mock_config):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    tracker = ExperimentTracker(str(tmp_path / "experiments.jsonl"))
    registry.register_model("shadow-v1", "GBM", {"sharpe": 1.2})
    registry.promote_to_shadow("shadow-v1")
    config = dict(mock_config)
    config["promotion"] = {
        "min_shadow_trades": 5,
        "min_sharpe_delta": 0.1,
        "max_shadow_drawdown": 0.10,
    }
    orchestrator = ModelLifecycleOrchestrator(
        config,
        registry=registry,
        tracker=tracker,
    )

    result = orchestrator.review_shadow_promotion(
        primary_metrics={"sharpe": 1.0, "total_trades": 30},
        shadow_metrics={"sharpe": 1.3, "total_trades": 20, "max_drawdown": 0.03},
    )

    runs = tracker.list_runs()
    assert result["decision"]["action"] == "promote"
    assert runs[0]["run_type"] == "promotion_review"
    assert runs[0]["status"] == "COMPLETED"
    assert runs[0]["model_id"] == "shadow-v1"
