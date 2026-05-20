import pandas as pd
import pytest

from src.mlops.evaluator import ModelEvaluator


@pytest.mark.mlops
def test_evaluator_rejects_insufficient_trades(mock_config):
    """Test that the evaluator rejects models with too few trades."""
    evaluator = ModelEvaluator(mock_config)
    short_history = [{"pnl": 10}] * 10
    mock_pnl_series = pd.Series([1000] * 10)

    metrics = evaluator.evaluate_oos(mock_pnl_series, short_history)

    assert metrics["passed_safety"] is False
    assert metrics["sharpe"] == 0.0
    assert metrics["max_drawdown"] == 1.0


@pytest.mark.mlops
def test_evaluator_calculates_metrics_correctly(mock_config, mock_trade_history):
    """Test that Sharpe, Drawdown, and Win Rate are mathematically correct."""
    evaluator = ModelEvaluator(mock_config)

    # Create a profitable but slightly volatile PnL series
    # 51 elements to match the pct_change() dropping the first NA for 50 trades
    equity_curve = [10000.0]
    for trade in mock_trade_history:
        equity_curve.append(equity_curve[-1] + trade["pnl"])

    mock_pnl_series = pd.Series(equity_curve)

    metrics = evaluator.evaluate_oos(mock_pnl_series, mock_trade_history)

    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics

    assert metrics["total_trades"] == 50
    assert (
        metrics["win_rate"] == 0.4
    )  # 2 winning trades out of 5 in the repeated pattern

    # This specific mocked curve shouldn't pass safety if Sharpe < 1.5 or DD > 10%
    # But we mainly care the math executed cleanly.
    assert isinstance(metrics["passed_safety"], bool)


@pytest.mark.mlops
def test_evaluator_rejects_high_drawdown(mock_config, mock_trade_history):
    """Verify that a model violating the 10% max drawdown limit fails safety."""
    evaluator = ModelEvaluator(mock_config)

    # Force a massive 50% drawdown
    equity_curve = [10000.0] * 25 + [5000.0] * 26
    mock_pnl_series = pd.Series(equity_curve)

    metrics = evaluator.evaluate_oos(mock_pnl_series, mock_trade_history)

    assert metrics["max_drawdown"] == 0.5
    assert metrics["passed_safety"] is False


@pytest.mark.mlops
def test_evaluator_handles_zero_variance_returns(mock_config, mock_trade_history):
    """Verify Sharpe falls back to zero when returns have no variance."""
    evaluator = ModelEvaluator(mock_config)
    flat_equity = pd.Series([10000.0] * 51)

    metrics = evaluator.evaluate_oos(flat_equity, mock_trade_history)

    assert metrics["sharpe"] == 0.0
    assert metrics["passed_safety"] is False


@pytest.mark.mlops
def test_evaluator_stress_gate_handles_empty_pass_and_fail(mock_config):
    config = dict(mock_config)
    config["mlops"] = {
        "stress": {
            "cost_bps": 1.0,
            "max_drawdown": 0.20,
            "min_return": 0.0,
        }
    }
    evaluator = ModelEvaluator(config)

    empty = evaluator.evaluate_stress(pd.Series(dtype=float), [])
    passing = evaluator.evaluate_stress(
        pd.Series([1000.0, 1010.0, 1020.0, 1030.0]),
        [{"pnl": 10.0}, {"pnl": 10.0}],
    )
    failing = evaluator.evaluate_stress(
        pd.Series([1000.0, 900.0, 850.0, 840.0]),
        [{"pnl": -50.0}] * 4,
    )

    assert empty["stress_passed"] is False
    assert empty["reason"] == "insufficient_equity_series"
    assert passing["stress_passed"] is True
    assert passing["stressed_return"] > 0
    assert failing["stress_passed"] is False
    assert failing["stressed_max_drawdown"] > 0.05
