import json

import pytest

from src.data.cache_manager import DuckDBCacheManager
from src.reports.paper_report import generate_paper_report, load_journal_trades


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
