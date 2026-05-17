import pandas as pd
import pytest

from src.pipelines.backtest import BacktestEngine


@pytest.mark.replay
def test_backtest_logs_long_and_short_position_closures(mock_config):
    """Verify backtest closes both long and short positions when targets flip."""
    engine = BacktestEngine(mock_config)
    actions = iter([2, 0, 1, 2])
    engine.meta_controller.get_action = lambda state, regime: (
        next(actions),
        0.9,
        {},
    )
    rows = []
    for idx, price in enumerate([100.0, 110.0, 100.0, 105.0]):
        row = {"close": price, "regime_str": "MEAN_REVERSION", "regime_id": 4}
        for feature_idx in range(10):
            row[f"feature_{feature_idx}"] = idx + feature_idx
        rows.append(row)

    equity_curve, trades = engine.run(pd.DataFrame(rows))

    assert len(equity_curve) == 4
    assert [trade["side"] for trade in trades] == ["LONG", "SHORT"]
    assert trades[0]["pnl"] > 0
    assert trades[1]["pnl"] > 0
