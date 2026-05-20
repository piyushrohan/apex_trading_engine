import os

import pytest
import yaml

from src.core import config_loader
from src.core.config_loader import (
    _deep_merge,
    apply_risk_profile,
    load_config,
    load_risk_profiles,
)


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


@pytest.mark.unit
def test_missing_risk_profiles_return_empty_and_leave_config_unchanged(
    monkeypatch, tmp_path
):
    assert load_risk_profiles(str(tmp_path / "missing.yaml")) == {}

    base = {"risk": {"profile": "balanced"}, "execution": {"max_leverage": 2}}
    monkeypatch.setattr(config_loader, "load_risk_profiles", lambda: {})

    assert apply_risk_profile(base) is base


@pytest.mark.unit
def test_load_config_env_override_and_missing_file(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"execution": {"operator_mode": "paper"}, "risk": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("APEX_EXECUTION_MODE", "live")
    monkeypatch.setattr(config_loader, "load_risk_profiles", lambda: {})

    loaded = load_config(str(config_path))

    assert loaded["execution"]["operator_mode"] == "live"
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.yaml"))
    assert os.getenv("APEX_EXECUTION_MODE") == "live"


@pytest.mark.unit
def test_deep_merge_preserves_nested_execution_settings():
    merged = _deep_merge(
        {"risk": {"max": 1, "nested": {"left": True}}, "mode": "paper"},
        {"risk": {"nested": {"right": True}}},
    )

    assert merged["risk"]["max"] == 1
    assert merged["risk"]["nested"] == {"left": True, "right": True}
    assert merged["mode"] == "paper"
