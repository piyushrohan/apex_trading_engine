from unittest.mock import MagicMock

import pytest

from src.pipelines import trading_pipeline
from src.pipelines.trading_pipeline import TradingPipeline


@pytest.mark.unit
def test_trading_pipeline_paper_operator_mode(mock_config, monkeypatch):
    monkeypatch.setattr(
        trading_pipeline,
        "DataIngestionService",
        lambda *a, **k: MagicMock(
            bootstrap_historical=MagicMock(),
            start_live=MagicMock(),
            stop=MagicMock(),
            close=MagicMock(),
            get_last_mark_price=MagicMock(return_value=None),
        ),
    )
    """Verify paper operator mode is honored on the shared pipeline."""
    config = dict(mock_config)
    config["execution"] = {
        **config.get("execution", {}),
        "operator_mode": "paper",
        "position_mode": "one_way",
    }
    pipeline = TradingPipeline(config)
    assert pipeline.operator_mode == "paper"
    assert pipeline.primary_book.role == "primary"
