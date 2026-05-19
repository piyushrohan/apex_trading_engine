import json

import pytest

from src.reports.hedge_report import generate_hedge_report


@pytest.mark.unit
def test_hedge_report_aggregates_selected_and_scores(tmp_path):
    journal = tmp_path / "journal.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    row = {
        "execution": {"mode": "paper"},
        "hedge": {
            "enabled": True,
            "selected": "protective_hedge",
            "candidates": {"protective_hedge": 0.8, "maker_grid_hedge": 0.3},
        },
        "pnl": 12.5,
    }
    journal.write_text(json.dumps(row) + "\n", encoding="utf-8")
    decisions.write_text(
        json.dumps(
            {
                "hedge": {
                    "enabled": True,
                    "selected": "maker_grid_hedge",
                    "candidates": {
                        "protective_hedge": 0.2,
                        "maker_grid_hedge": 0.7,
                    },
                },
                "hedge_pnl": -1.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = generate_hedge_report(str(journal), str(decisions), days=7)
    assert report["total_selected"] == 2
    assert report["strategies"]["protective_hedge"]["selected_count"] == 1
    assert report["strategies"]["protective_hedge"]["avg_score"] == 0.5
    assert report["strategies"]["maker_grid_hedge"]["pnl"] == -1.5
