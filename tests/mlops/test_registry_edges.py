import json

import pytest

from src.mlops import registry as registry_module
from src.mlops.registry import ModelRegistry


@pytest.mark.mlops
def test_registry_raises_for_unknown_promotions(tmp_path):
    """Verify invalid state transitions fail clearly."""
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))

    with pytest.raises(ValueError, match="missing-shadow"):
        registry.promote_to_shadow("missing-shadow")

    with pytest.raises(ValueError, match="missing-prod"):
        registry.promote_to_prod("missing-prod")


@pytest.mark.mlops
def test_registry_returns_none_without_active_prod(tmp_path):
    """Verify registry reports no production model before promotion."""
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))

    assert registry.get_prod_model_path() is None


@pytest.mark.mlops
def test_registry_returns_active_prod_path(tmp_path):
    """Verify active production model resolves to its artifact directory."""
    registry_dir = tmp_path / "models"
    registry = ModelRegistry(registry_dir=str(registry_dir))
    registry.register_model("prod-v1", "PPO", {"sharpe": 2.0})
    registry.promote_to_prod("prod-v1")

    assert registry.get_prod_model_path() == str(registry_dir / "prod-v1")


@pytest.mark.mlops
def test_registry_writes_manifest_and_updates_metadata(tmp_path, monkeypatch):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.register_model("model-v1", "GBM", {"sharpe": 1.0})
    monkeypatch.setattr(
        ModelRegistry, "_current_git_hash", staticmethod(lambda: "abc123")
    )

    manifest_path = registry.write_model_manifest(
        "model-v1",
        data_snapshot_id="snapshot-1",
        hyperparams={"depth": 3},
    )
    payload = json.loads(
        (tmp_path / "models" / "model-v1" / "manifest.json").read_text()
    )

    assert manifest_path == str(tmp_path / "models" / "model-v1" / "manifest.json")
    assert payload["git_hash"] == "abc123"
    assert payload["metrics"] == {"sharpe": 1.0}
    assert (
        registry.registry_data["models"]["model-v1"]["metadata"]["data_snapshot_id"]
        == "snapshot-1"
    )

    with pytest.raises(ValueError, match="missing"):
        registry.write_model_manifest("missing")


@pytest.mark.mlops
def test_current_git_hash_returns_output_or_none(monkeypatch):
    class Result:
        stdout = "abc123\n"

    monkeypatch.setattr(registry_module.subprocess, "run", lambda *a, **k: Result())
    assert ModelRegistry._current_git_hash() == "abc123"

    class EmptyResult:
        stdout = ""

    monkeypatch.setattr(
        registry_module.subprocess, "run", lambda *a, **k: EmptyResult()
    )
    assert ModelRegistry._current_git_hash() is None

    def raise_error(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(registry_module.subprocess, "run", raise_error)
    assert ModelRegistry._current_git_hash() is None


@pytest.mark.mlops
def test_registry_metric_status_and_shadow_archival_edges(tmp_path):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.register_model("shadow-v1", "GBM", {"sharpe": 1.0})
    registry.register_model("shadow-v2", "PPO", {"sharpe": 1.1})

    registry.update_model_metrics("shadow-v1", {"sharpe": 1.2})
    registry.set_model_status("shadow-v1", "READY")
    registry.promote_to_shadow("shadow-v1")
    registry.promote_to_shadow("shadow-v2")

    assert registry.registry_data["models"]["shadow-v1"]["metrics"] == {"sharpe": 1.2}
    assert registry.registry_data["models"]["shadow-v1"]["status"] == "ARCHIVED"
    assert registry.registry_data["models"]["shadow-v2"]["status"] == "SHADOW"

    with pytest.raises(ValueError, match="missing"):
        registry.update_model_metrics("missing", {})
    with pytest.raises(ValueError, match="missing"):
        registry.set_model_status("missing", "READY")


@pytest.mark.mlops
def test_registry_rollback_without_previous_prod_returns_none(tmp_path):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.register_model("prod-v1", "GBM", {"sharpe": 1.0})
    registry.promote_to_prod("prod-v1")

    assert registry.rollback_prod() is None
