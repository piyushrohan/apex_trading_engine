import pytest

from src.execution.grid_adapter import MakerGridAdapter


@pytest.mark.unit
def test_maker_grid_adapter_builds_two_sided_post_only_orders():
    adapter = MakerGridAdapter(tick_size=0.01)
    plan = adapter.build_grid(
        symbol="ETHUSDC",
        mid_price=3500.0,
        total_quantity=0.6,
        levels=3,
        spacing_ticks=2,
    )
    assert len(plan.orders) == 6
    assert plan.orders[0].side == "BUY"
    assert plan.orders[0].position_side == "LONG"
    assert plan.orders[1].side == "SELL"
    assert plan.orders[1].position_side == "SHORT"
    assert plan.orders[0].price == 3499.98
    assert plan.orders[-1].price == 3500.06
