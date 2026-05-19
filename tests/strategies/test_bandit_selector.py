import json

import pytest

from src.strategies.hedge.bandit_selector import ContextualBanditSelector
from src.strategies.hedge.base import HedgeContext
from src.strategies.hedge.registry import build_hedge_orchestrator


def _ctx() -> HedgeContext:
    return HedgeContext(
        symbol="ETHUSDC",
        regime="CHOP_COMPRESSION",
        mark_price=3500.0,
        feature_vector=[0.0] * 12,
        ppo_action_probs=[0.1, 0.1, 0.8],
        gbm_action_probs=[0.8, 0.1, 0.1],
        primary_action=2,
        primary_size_fraction=0.2,
        volatility_zscore=-0.4,
        funding_rate=0.0002,
    )


@pytest.mark.unit
def test_bandit_not_eligible_without_min_decisions(tmp_path):
    config = {
        "hedge": {
            "bandit": {
                "min_decisions": 2,
                "state_path": str(tmp_path / "state.json"),
            }
        },
        "shadow": {"decision_log_path": str(tmp_path / "decisions.jsonl")},
    }
    selector = ContextualBanditSelector(config, ["a", "b"])
    assert selector.is_eligible() is False


@pytest.mark.unit
def test_bandit_becomes_eligible_after_decision_history(tmp_path):
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps({"hedge": {"candidates": {"a": 0.3, "b": 0.4}}})
        + "\n"
        + json.dumps({"hedge": {"candidates": {"a": 0.5, "b": 0.6}}})
        + "\n",
        encoding="utf-8",
    )
    config = {
        "hedge": {
            "bandit": {
                "min_decisions": 2,
                "state_path": str(tmp_path / "state.json"),
            }
        },
        "shadow": {"decision_log_path": str(decisions)},
    }
    selector = ContextualBanditSelector(config, ["a", "b"])
    assert selector.is_eligible() is True


@pytest.mark.unit
def test_orchestrator_falls_back_to_rule_based_before_gate(tmp_path, mock_config):
    config = dict(mock_config)
    config["hedge"] = {
        "enabled": True,
        "selection": "contextual_bandit",
        "min_score": 0.3,
        "strategies": {
            "signal_disagreement": {"enabled": True},
            "protective_hedge": {"enabled": True},
        },
        "bandit": {
            "min_decisions": 10,
            "state_path": str(tmp_path / "state.json"),
        },
    }
    config["shadow"] = {"decision_log_path": str(tmp_path / "decisions.jsonl")}

    orch = build_hedge_orchestrator(config)
    _, payload = orch.evaluate(_ctx())
    assert payload["selection_mode"] == "rule_based"
    assert "candidates_rule_shadow" not in payload


@pytest.mark.unit
def test_orchestrator_uses_bandit_after_gate_and_logs_rule_scores(
    tmp_path, mock_config
):
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        json.dumps(
            {
                "hedge": {
                    "candidates": {
                        "signal_disagreement": 0.7,
                        "protective_hedge": 0.2,
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = dict(mock_config)
    config["hedge"] = {
        "enabled": True,
        "selection": "contextual_bandit",
        "min_score": 0.3,
        "strategies": {
            "signal_disagreement": {"enabled": True},
            "protective_hedge": {"enabled": True},
        },
        "bandit": {
            "min_decisions": 1,
            "state_path": str(tmp_path / "state.json"),
        },
    }
    config["shadow"] = {"decision_log_path": str(decisions)}

    orch = build_hedge_orchestrator(config)
    _, payload = orch.evaluate(_ctx())
    assert payload["selection_mode"] == "contextual_bandit"
    assert payload["bandit_arm"] in {"signal_disagreement", "protective_hedge"}
    assert "candidates_rule_shadow" in payload
    assert set(payload["candidates_rule_shadow"].keys()) == {
        "signal_disagreement",
        "protective_hedge",
    }
