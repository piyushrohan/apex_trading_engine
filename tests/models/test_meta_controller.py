from types import SimpleNamespace

import pytest

from src.models.meta_controller import MetaController


class FakeAgent:
    def __init__(self, action):
        self.action = action

    def act(self, state_vector):
        return self.action, 0.9, {"feature_contributions": [0.1] * len(state_vector)}


@pytest.mark.unit
def test_meta_controller_routes_chop_to_gbm(mock_config):
    """Verify chop regimes route through the GBM specialist."""
    controller = MetaController(mock_config)
    controller.ppo_agent = FakeAgent(2)
    controller.gbm_agent = FakeAgent(1)

    action, conviction, context = controller.get_action([0.1] * 10, "CHOP_COMPRESSION")

    assert action == 1
    assert conviction == 0.9
    assert context["selected_by_meta"] == "GBM"
    assert context["active_regime"] == "CHOP_COMPRESSION"


@pytest.mark.unit
def test_meta_controller_defaults_unknown_regime_to_ppo(mock_config):
    """Verify unknown regimes fall back to PPO."""
    controller = MetaController(mock_config)
    controller.ppo_agent = FakeAgent(2)
    controller.gbm_agent = FakeAgent(1)

    action, _, context = controller.get_action([0.1] * 10, "UNKNOWN_REGIME")

    assert action == 2
    assert context["selected_by_meta"] == "PPO"


@pytest.mark.unit
def test_meta_controller_quarantines_native_gbm_runtime_crash(
    mock_config, tmp_path, monkeypatch
):
    """Verify unsafe LightGBM/Torch artifact mixes are blocked before pickle load."""
    artifact_dir = tmp_path / "model"
    artifact_dir.mkdir()
    (artifact_dir / "gbm_model.pkl").write_bytes(b"not-loaded-in-parent")
    controller = MetaController(mock_config)
    controller.gbm_agent.load = pytest.fail

    def fake_run(command, **kwargs):
        assert "-X" in command
        assert "faulthandler" in command
        assert str(artifact_dir) in command
        return SimpleNamespace(
            returncode=-11,
            stderr="Fatal Python error: Segmentation fault",
            stdout="",
        )

    monkeypatch.setattr("src.models.meta_controller.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="unsafe in combined PPO/GBM runtime"):
        controller.load_model_artifact("GBM", str(artifact_dir))


@pytest.mark.unit
def test_meta_controller_can_disable_gbm_runtime_preflight(mock_config, monkeypatch):
    """Verify local tests and controlled ops can bypass preflight when needed."""
    config = dict(mock_config)
    config["models"] = {
        "gbm": {
            "combined_runtime_preflight_enabled": False,
        }
    }
    controller = MetaController(config)
    calls = []
    controller.gbm_agent.load = lambda path: calls.append(path)
    monkeypatch.setattr(
        "src.models.meta_controller.subprocess.run",
        lambda *args, **kwargs: pytest.fail("preflight should be disabled"),
    )

    controller.load_model_artifact("LIGHTGBM", "/tmp/model")

    assert calls == ["/tmp/model"]
