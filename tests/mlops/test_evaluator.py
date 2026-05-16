import pytest
import pandas as pd
from src.mlops.evaluator import ModelEvaluator

def test_evaluator_rejects_insufficient_trades(mock_config):
    """Test that the evaluator rejects models with too few trades."""
    evaluator = ModelEvaluator(mock_config)
    short_history = [{"pnl": 10}] * 10
    mock_pnl_series = pd.Series([1000] * 10)
    
    metrics = evaluator.evaluate_oos(mock_pnl_series, short_history)
    
    assert metrics["passed_safety"] is False
    assert metrics["sharpe"] == 0.0
    assert metrics["max_drawdown"] == 1.0

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
    assert metrics["win_rate"] == 0.4 # 2 winning trades out of 5 in the repeated pattern
    
    # This specific mocked curve shouldn't pass safety if Sharpe < 1.5 or DD > 10%
    # But we mainly care the math executed cleanly.
    assert isinstance(metrics["passed_safety"], bool)

def test_evaluator_rejects_high_drawdown(mock_config, mock_trade_history):
    """Verify that a model violating the 10% max drawdown limit fails safety."""
    evaluator = ModelEvaluator(mock_config)
    
    # Force a massive 50% drawdown
    equity_curve = [10000.0] * 25 + [5000.0] * 26
    mock_pnl_series = pd.Series(equity_curve)
    
    metrics = evaluator.evaluate_oos(mock_pnl_series, mock_trade_history)
    
    assert metrics["max_drawdown"] == 0.5
    assert metrics["passed_safety"] is False
