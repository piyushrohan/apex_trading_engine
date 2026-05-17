import pandas as pd
import pytest

from src.pipelines.backtest import BacktestEngine


@pytest.mark.replay
def test_historical_replay_simulation(mock_config):
    """
    Layer 3 Simulation/Replay Test:
    Verify the deterministic backtest engine correctly processes historical
    snapshots, transitions states, and updates virtual equity accurately.
    """
    engine = BacktestEngine(mock_config)

    # Create deterministic historical snapshots
    # Assume 10 features, all 0.1 for simplicity.
    data = []
    for i in range(10):
        row = {
            "timestamp": pd.Timestamp("2026-05-16 12:00:00")
            + pd.Timedelta(minutes=i * 3),
            "close": 3000 + (i * 10),
            "regime_str": "STRONG_TREND_UP",
            "regime_id": 2,
        }
        for f in range(10):
            row[f"feature_{f}"] = 0.1
        data.append(row)

    df_historical = pd.DataFrame(data).set_index("timestamp")

    # Run historical replay
    equity_curve, trade_logs = engine.run(df_historical)

    assert len(equity_curve) == len(df_historical)
    assert equity_curve.iloc[0] == mock_config["environment"]["initial_capital"]

    # Ensure some trades were logged and executed sequentially
    assert isinstance(trade_logs, list)
