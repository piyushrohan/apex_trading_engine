import pytest

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
