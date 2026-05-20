import pytest

from src.mlops.promotion_service import PromotionService
from src.mlops.registry import ModelRegistry


def approve_prod(registry, model_id):
    registry.promote_to_shadow(model_id)
    registry.approve_for_prod(model_id, reviewer="test")
    registry.promote_to_prod(model_id, reviewer="test")


@pytest.mark.mlops
def test_promotion_service_promotes_shadow_outperformer(tmp_path, mock_config):
    """Verify shadow candidates promote only through the MLOps gate."""
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.register_model("prod-v1", "GBM", {"sharpe": 1.0})
    approve_prod(registry, "prod-v1")
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
    approve_prod(registry, "prod-v1")
    registry.register_model("prod-v2", "GBM", {"sharpe": 1.5})
    approve_prod(registry, "prod-v2")
    service = PromotionService(mock_config, registry=registry)

    restored = service.rollback_active_prod()

    assert restored == "prod-v1"
    assert registry.registry_data["active_prod"] == "prod-v1"
    assert registry.registry_data["models"]["prod-v2"]["status"] == "ROLLED_BACK"


@pytest.mark.mlops
def test_promotion_service_holds_for_insufficient_history_and_small_edge(mock_config):
    config = dict(mock_config)
    config["promotion"] = {"min_shadow_trades": 10, "min_sharpe_delta": 0.25}
    service = PromotionService(config, registry=ModelRegistry.__new__(ModelRegistry))

    insufficient = service.evaluate(
        "shadow-v2",
        primary_metrics={"sharpe": 1.0},
        shadow_metrics={"sharpe": 1.4, "total_trades": 3},
    )
    small_edge = service.evaluate(
        "shadow-v2",
        primary_metrics={"sharpe": 1.0},
        shadow_metrics={"sharpe": 1.1, "max_drawdown": 0.01, "total_trades": 30},
    )

    assert insufficient.action == "hold"
    assert insufficient.reason == "insufficient_shadow_trades"
    assert small_edge.action == "hold"
    assert small_edge.reason == "shadow_edge_not_material"


@pytest.mark.mlops
def test_metrics_from_decision_log_handles_missing_sparse_and_rich_rows(
    tmp_path, mock_config
):
    service = PromotionService(
        mock_config, registry=ModelRegistry.__new__(ModelRegistry)
    )
    missing = tmp_path / "missing.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        "\n"
        + '{"book": {"id": "other"}, "action": 2, "equity": 999}\n'
        + '{"book": {"id": "shadow_a"}, "action": 1, "approved_fraction": 0.1, '
        '"equity": 1000}\n'
        + '{"book": {"id": "shadow_a"}, "action": 2, "equity": 1010}\n'
        + '{"book": {"id": "shadow_a"}, "action": 1, "equity": 990}\n'
        + '{"book": {"id": "shadow_a"}, "action": 0, "equity": 1020}\n',
        encoding="utf-8",
    )

    assert service.metrics_from_decision_log(
        decision_path=str(missing),
        book_id="shadow_a",
    ) == {"total_trades": 0, "sharpe": 0.0, "max_drawdown": 1.0}

    metrics = service.metrics_from_decision_log(
        decision_path=str(decisions),
        book_id="shadow_a",
    )

    assert metrics["total_trades"] == 3
    assert metrics["max_drawdown"] > 0
    assert metrics["sharpe"] != 0.0


@pytest.mark.mlops
def test_rollback_if_live_breach_uses_configured_limits(tmp_path, mock_config):
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))
    registry.register_model("prod-v1", "GBM", {"sharpe": 1.0})
    approve_prod(registry, "prod-v1")
    registry.register_model("prod-v2", "GBM", {"sharpe": 1.5})
    approve_prod(registry, "prod-v2")
    config = dict(mock_config)
    config["promotion"] = {"max_live_drawdown": 0.05, "min_live_sharpe": 0.0}
    service = PromotionService(config, registry=registry)

    assert (
        service.rollback_if_live_breach({"max_drawdown": 0.01, "sharpe": 0.2}) is None
    )
    restored = service.rollback_if_live_breach({"max_drawdown": 0.2, "sharpe": 0.2})

    assert restored == "prod-v1"
