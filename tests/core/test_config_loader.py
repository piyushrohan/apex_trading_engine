import pytest

from src.core.config_loader import apply_risk_profile, load_config


@pytest.mark.unit
def test_load_config_merges_balanced_profile():
    config = load_config("configs/base.yaml")
    assert config["risk"]["profile"] == "balanced"
    assert config["execution"]["max_leverage"] == 3.0
    assert config["execution"]["operator_mode"] == "paper"


@pytest.mark.unit
def test_apply_risk_profile_conservative():
    base = load_config("configs/base.yaml")
    conservative = apply_risk_profile(base, "conservative")
    assert conservative["execution"]["max_leverage"] == 2.0
    assert conservative["execution"]["max_daily_drawdown"] == 0.03


@pytest.mark.unit
def test_unknown_risk_profile_raises():
    base = load_config("configs/base.yaml")
    with pytest.raises(ValueError, match="Unknown risk profile"):
        apply_risk_profile(base, "nonexistent")
