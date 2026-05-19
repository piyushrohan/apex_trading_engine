import pytest

from src.mlops.promotion_service import PromotionService
from src.mlops.registry import ModelRegistry


@pytest.mark.mlops
def test_promotion_service_promotes_shadow_outperformer(tmp_path, mock_config):
    """Verify shadow candidates promote only through the MLOps gate."""
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.register_model("prod-v1", "GBM", {"sharpe": 1.0})
    registry.promote_to_prod("prod-v1")
    registry.register_model("shadow-v2", "GBM", {"sharpe": 1.4})
    registry.promote_to_shadow("shadow-v2")
    config = dict(mock_config)
    config["promotion"] = {
        "min_shadow_trades": 10,
        "min_sharpe_delta": 0.2,
        "max_shadow_drawdown": 0.08,
    }
    service = PromotionService(config, registry=registry)

    decision = service.evaluate_and_apply(
        "shadow-v2",
        primary_metrics={"sharpe": 1.0, "max_drawdown": 0.05, "total_trades": 100},
        shadow_metrics={"sharpe": 1.35, "max_drawdown": 0.04, "total_trades": 50},
    )

    assert decision.action == "promote"
    assert registry.registry_data["active_prod"] == "shadow-v2"
    assert registry.registry_data["previous_prod"] == "prod-v1"


@pytest.mark.mlops
def test_promotion_service_discards_drawdown_breach(tmp_path, mock_config):
    """Verify unsafe shadow candidates are rejected, not promoted."""
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.register_model("shadow-v2", "GBM", {"sharpe": 2.0})
    config = dict(mock_config)
    config["promotion"] = {"min_shadow_trades": 10, "max_shadow_drawdown": 0.08}
    service = PromotionService(config, registry=registry)

    decision = service.evaluate_and_apply(
        "shadow-v2",
        primary_metrics={"sharpe": 1.0, "total_trades": 100},
        shadow_metrics={"sharpe": 2.0, "max_drawdown": 0.20, "total_trades": 50},
    )

    assert decision.action == "discard"
    assert registry.registry_data["models"]["shadow-v2"]["status"] == "REJECTED"


@pytest.mark.mlops
def test_promotion_service_rolls_back_previous_prod(tmp_path, mock_config):
    """Verify rollback restores the previous production model."""
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.register_model("prod-v1", "GBM", {"sharpe": 1.0})
    registry.promote_to_prod("prod-v1")
    registry.register_model("prod-v2", "GBM", {"sharpe": 1.5})
    registry.promote_to_prod("prod-v2")
    service = PromotionService(mock_config, registry=registry)

    restored = service.rollback_active_prod()

    assert restored == "prod-v1"
    assert registry.registry_data["active_prod"] == "prod-v1"
    assert registry.registry_data["models"]["prod-v2"]["status"] == "ROLLED_BACK"
