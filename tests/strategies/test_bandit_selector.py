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


@pytest.mark.unit
def test_bandit_selector_empty_and_missing_rule_score_paths(tmp_path):
    config = {
        "hedge": {"bandit": {"state_path": str(tmp_path / "state.json")}},
        "shadow": {"decision_log_path": str(tmp_path / "decisions.jsonl")},
    }
    empty = ContextualBanditSelector(config, [])
    assert empty.is_eligible() is False

    selector = ContextualBanditSelector(config, ["protective_hedge"])
    assert selector.select_arm(_ctx(), {}) == (None, {}, False)
    assert selector.select_arm(_ctx(), {"maker_grid_hedge": 0.7}) == (None, {}, False)

    selector.record_reward("missing", _ctx(), 1.0)
    assert not (tmp_path / "state.json").exists()


@pytest.mark.unit
def test_bandit_record_reward_persists_state_and_detects_tie(tmp_path):
    config = {
        "hedge": {
            "bandit": {
                "exploration_factor": 0.0,
                "state_path": str(tmp_path / "state.json"),
                "min_decisions": 1,
            }
        },
        "shadow": {"decision_log_path": str(tmp_path / "decisions.jsonl")},
    }
    selector = ContextualBanditSelector(config, ["a", "b"])

    arm, scores, exploration = selector.select_arm(_ctx(), {"a": 0.2, "b": 0.3})
    assert arm in {"a", "b"}
    assert set(scores) == {"a", "b"}
    assert exploration is True

    selector.record_reward("a", _ctx(), 0.75)

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert selector.counts["a"] == 1
    assert state["counts"]["a"] == 1
    assert len(state["a"]["a"]) == selector.dim


@pytest.mark.unit
def test_bandit_reward_log_updates_valid_rows_only(tmp_path):
    decisions = tmp_path / "decisions.jsonl"
    decisions.write_text(
        "\n"
        + json.dumps({"hedge": {"selected": "missing"}, "hedge_reward": 1.0})
        + "\n"
        + json.dumps({"hedge": {"selected": "a"}})
        + "\n"
        + json.dumps(
            {
                "symbol": "ETHUSDC",
                "regime": "CHOP_COMPRESSION",
                "mark_price": 3510.0,
                "action": 2,
                "hedge": {"selected": "a", "candidates": {"a": 0.8}},
                "hedge_reward": 0.25,
                "bandit_context": {
                    "ppo_action_probs": [0.1, 0.2, 0.7],
                    "gbm_action_probs": [0.7, 0.2, 0.1],
                    "primary_action": 2,
                    "primary_size_fraction": 0.3,
                    "volatility_zscore": 5.0,
                    "funding_rate": 0.001,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "hedge": {
            "bandit": {"state_path": str(tmp_path / "state.json"), "min_decisions": 1}
        },
        "shadow": {"decision_log_path": str(decisions)},
    }

    selector = ContextualBanditSelector(config, ["a"])

    assert selector.total_observations == 3
    assert selector.update_from_reward_log() == 1
    assert selector.counts["a"] == 2


@pytest.mark.unit
def test_bandit_load_state_restores_counts_and_matrices(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "counts": {"a": 4},
                "a": {"a": [[2, 0, 0, 0, 0, 0]] * 6},
                "b": {"a": [1, 2, 3, 4, 5, 6]},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "hedge": {"bandit": {"state_path": str(state_path), "min_decisions": 4}},
        "shadow": {"decision_log_path": str(tmp_path / "missing.jsonl")},
    }

    selector = ContextualBanditSelector(config, ["a"])

    assert selector.is_eligible() is True
    assert selector._a["a"][0][0] == 2
    assert selector._b["a"].tolist() == [1, 2, 3, 4, 5, 6]


@pytest.mark.unit
def test_bandit_update_from_missing_reward_log_returns_zero(tmp_path):
    config = {
        "hedge": {"bandit": {"state_path": str(tmp_path / "state.json")}},
        "shadow": {"decision_log_path": str(tmp_path / "missing.jsonl")},
    }

    selector = ContextualBanditSelector(config, ["a"])

    assert selector.update_from_reward_log() == 0
