import json
import sys
from datetime import datetime, timedelta, timezone

import pytest

from src.reports import hedge_report
from src.reports.hedge_report import _parse_timestamp, generate_hedge_report


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


@pytest.mark.unit
def test_hedge_report_skips_old_disabled_invalid_and_missing_rows(tmp_path):
    journal = tmp_path / "journal.jsonl"
    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    journal.write_text(
        "\n"
        + json.dumps(
            {
                "timestamp": old_ts,
                "hedge": {
                    "enabled": True,
                    "selected": "old_strategy",
                    "candidates": {"old_strategy": 1.0},
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "timestamp": "not-a-date",
                "hedge": {"enabled": False, "selected": "disabled_strategy"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "timestamp": "not-a-date",
                "hedge": {
                    "enabled": True,
                    "candidates": {"observed_only": 0.4},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert _parse_timestamp(None) is None
    assert _parse_timestamp("not-a-date") is None
    assert (
        generate_hedge_report(
            str(tmp_path / "missing.jsonl"),
            str(tmp_path / "missing_decisions.jsonl"),
            days=1,
        )["total_selected"]
        == 0
    )

    report = generate_hedge_report(
        str(journal), str(tmp_path / "missing.jsonl"), days=7
    )

    assert "old_strategy" not in report["strategies"]
    assert "disabled_strategy" not in report["strategies"]
    assert report["strategies"]["observed_only"]["avg_score"] == 0.4
    assert report["total_selected"] == 0


@pytest.mark.unit
def test_hedge_report_main_uses_configured_paths(tmp_path, monkeypatch, capsys):
    journal = tmp_path / "journal.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    journal.write_text("", encoding="utf-8")
    decisions.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hedge_report", "--days", "3"])
    monkeypatch.setattr(
        hedge_report,
        "load_config",
        lambda path: {
            "explainability": {"journal_path": str(journal)},
            "shadow": {"decision_log_path": str(decisions)},
        },
    )

    hedge_report.main()

    output = json.loads(capsys.readouterr().out)
    assert output["window_days"] == 3
