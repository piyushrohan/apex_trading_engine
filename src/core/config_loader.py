import os
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = "configs/base.yaml"
DEFAULT_PROFILES_PATH = "configs/risk_profiles.yaml"


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_risk_profiles(
    profiles_path: str = DEFAULT_PROFILES_PATH,
) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(profiles_path):
        return {}
    with open(profiles_path, "r") as file:
        return yaml.safe_load(file) or {}


def apply_risk_profile(
    config: Dict[str, Any], profile_name: Optional[str] = None
) -> Dict[str, Any]:
    profiles = load_risk_profiles()
    if not profiles:
        return config

    name = profile_name or config.get("risk", {}).get("profile", "balanced")
    if name not in profiles:
        raise ValueError(f"Unknown risk profile: {name}")

    profile = profiles[name]
    execution_overlay = {
        "max_leverage": profile.get("max_leverage"),
        "max_daily_drawdown": profile.get("max_daily_drawdown"),
        "kelly_fraction_cap": profile.get("kelly_fraction_cap"),
        "max_gross_leverage": profile.get("max_gross_leverage"),
        "max_net_leverage": profile.get("max_net_leverage"),
        "max_hedge_ratio": profile.get("max_hedge_ratio"),
    }
    execution_overlay = {k: v for k, v in execution_overlay.items() if v is not None}

    merged = dict(config)
    merged["risk"] = {**config.get("risk", {}), "profile": name, **profile}
    merged["execution"] = _deep_merge(config.get("execution", {}), execution_overlay)
    return merged


def load_config(
    config_path: str = DEFAULT_CONFIG_PATH,
    risk_profile: Optional[str] = None,
) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    env_mode = os.getenv("APEX_EXECUTION_MODE")
    if env_mode:
        config.setdefault("execution", {})["operator_mode"] = env_mode

    return apply_risk_profile(config, risk_profile)
