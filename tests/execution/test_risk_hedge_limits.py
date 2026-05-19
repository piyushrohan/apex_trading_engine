import pytest

from src.execution.risk_engine import RiskEngine


@pytest.mark.risk
def test_hedge_gross_leverage_cap(mock_config):
    config = dict(mock_config)
    config["execution"]["position_mode"] = "hedge"
    config["execution"]["max_gross_leverage"] = 1.0
    engine = RiskEngine(config)

    ok, reason = engine.check_hedge_limits(
        long_qty=0.5,
        short_qty=0.5,
        equity=1000.0,
        mark_price=2000.0,
        add_long=0.1,
    )
    assert not ok
    assert "gross" in reason


@pytest.mark.risk
def test_hedge_ratio_cap_on_hedge_leg(mock_config):
    config = dict(mock_config)
    config["execution"]["position_mode"] = "hedge"
    config["execution"]["max_hedge_ratio"] = 0.2
    engine = RiskEngine(config)

    ok, reason = engine.check_hedge_limits(
        long_qty=1.0,
        short_qty=0.25,
        equity=100_000.0,
        mark_price=1000.0,
        add_short=0.2,
        is_hedge_leg=True,
    )
    assert not ok
    assert "hedge ratio" in reason.lower()


@pytest.mark.risk
def test_approve_order_passes_hedge_limits(mock_config):
    config = dict(mock_config)
    config["execution"]["position_mode"] = "hedge"
    config["execution"]["max_gross_leverage"] = 3.0
    config["execution"]["max_net_leverage"] = 2.0
    engine = RiskEngine(config)

    approved = engine.approve_order(
        "BUY",
        0.1,
        current_exposure=0.0,
        long_qty=0.0,
        short_qty=0.0,
        equity=10000.0,
        mark_price=3500.0,
    )
    assert approved == 0.1
