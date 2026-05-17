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
