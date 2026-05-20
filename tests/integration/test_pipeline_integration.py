import pytest

from src.mlops.registry import ModelRegistry


@pytest.mark.integration
def test_registry_state_transitions(tmp_path):
    """
    Integration Test:
    Verify the ModelRegistry state machine correctly promotes models
    from CANDIDATE -> SHADOW -> APPROVED -> PROD and archives old models.
    """
    # Use temporary directory for sandbox
    registry_dir = str(tmp_path / "models")
    registry = ModelRegistry(registry_dir=registry_dir)

    # Register V1
    registry.register_model("model_v1", "PPO", {"sharpe": 2.0})
    assert registry.registry_data["models"]["model_v1"]["status"] == "CANDIDATE"

    # Promote V1 to SHADOW
    registry.promote_to_shadow("model_v1")
    assert registry.registry_data["models"]["model_v1"]["status"] == "SHADOW"
    assert registry.registry_data["active_shadow"] == "model_v1"

    # Promote V1 to PROD
    registry.approve_for_prod("model_v1", reviewer="test")
    registry.promote_to_prod("model_v1", reviewer="test")
    assert registry.registry_data["models"]["model_v1"]["status"] == "PROD"
    assert registry.registry_data["active_prod"] == "model_v1"

    # Register and Promote V2
    registry.register_model("model_v2", "PPO", {"sharpe": 3.0})
    registry.promote_to_shadow("model_v2")
    registry.approve_for_prod("model_v2", reviewer="test")
    registry.promote_to_prod("model_v2", reviewer="test")

    # Verify V1 was automatically archived
    assert registry.registry_data["models"]["model_v1"]["status"] == "ARCHIVED"
    assert registry.registry_data["active_prod"] == "model_v2"
