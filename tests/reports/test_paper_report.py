import json
import sys

import pandas as pd
import pytest

from src.data.cache_manager import DuckDBCacheManager
from src.reports import paper_report
from src.reports.paper_report import (
    _max_drawdown,
    _sharpe_from_equity,
    generate_paper_report,
    load_journal_fills,
    load_journal_trades,
)


@pytest.mark.unit
def test_paper_report_from_snapshots(tmp_path, mock_config):
    db_path = str(tmp_path / "paper.duckdb")
    mock_config["data"]["storage"] = {"db_path": db_path}
    cache = DuckDBCacheManager(db_path=db_path)
    for i, eq in enumerate([1000.0, 1010.0, 1005.0, 1020.0]):
        cache.insert_paper_equity_snapshot(
            book_id="primary",
            equity=eq,
            long_qty=0.1,
            short_qty=0.0,
            mark_price=3500.0 + i,
            regime="MEAN_REVERSION",
        )
    cache.close()

    report = generate_paper_report(mock_config)
    assert report["snapshots"] == 4
    assert report["final_equity"] == 1020.0
    assert report["max_drawdown"] >= 0


@pytest.mark.unit
def test_load_journal_trades_filters_paper_mode(tmp_path):
    journal = tmp_path / "trade_journal.jsonl"
    journal.write_text(
        json.dumps(
            {
                "execution": {"mode": "paper"},
                "decision": "LONG",
                "book": {"role": "primary"},
            }
        )
        + "\n"
        + json.dumps({"execution": {"mode": "live"}, "decision": "SHORT"})
        + "\n"
    )
    trades = load_journal_trades(str(journal))
    assert len(trades) == 1
    assert trades[0]["decision"] == "LONG"


@pytest.mark.unit
def test_paper_report_empty_cache_and_journal_fill_counts(tmp_path, mock_config):
    db_path = str(tmp_path / "paper.duckdb")
    journal = tmp_path / "trade_journal.jsonl"
    mock_config["data"]["storage"] = {"db_path": db_path}
    DuckDBCacheManager(db_path=db_path).close()
    journal.write_text(
        "\n"
        + json.dumps(
            {
                "execution": {"mode": "paper"},
                "decision": "LONG",
                "book": {"role": "primary"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "paper_fill",
                "execution": {"mode": "paper"},
                "book": {"role": "primary"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "paper_fill",
                "execution": {"mode": "live"},
                "book": {"role": "primary"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = generate_paper_report(mock_config, journal_path=str(journal))

    assert report["snapshots"] == 0
    assert report["directional_decisions"] == 1
    assert report["filled_orders"] == 1
    assert report["fill_rate"] == 1.0
    assert load_journal_trades(str(tmp_path / "missing.jsonl")) == []
    assert load_journal_fills(str(tmp_path / "missing.jsonl")) == []


@pytest.mark.unit
def test_paper_report_metric_helpers_cover_empty_and_flat_equity():
    assert _max_drawdown(pd.Series(dtype=float)) == 0.0
    assert _sharpe_from_equity(pd.Series([100.0, 100.0])) == 0.0
    assert _sharpe_from_equity(pd.Series([100.0, 100.0, 100.0])) == 0.0


@pytest.mark.unit
def test_paper_report_main_prints_json(tmp_path, monkeypatch, capsys, mock_config):
    db_path = str(tmp_path / "paper.duckdb")
    mock_config["data"]["storage"] = {"db_path": db_path}
    DuckDBCacheManager(db_path=db_path).close()
    monkeypatch.setattr(sys, "argv", ["paper_report", "--config", "config.yaml"])
    monkeypatch.setattr(paper_report, "load_config", lambda path: mock_config)

    paper_report.main()

    output = json.loads(capsys.readouterr().out)
    assert output["book_id"] == "primary"
