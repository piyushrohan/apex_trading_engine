import json

import pytest

from src.mlops.performance_calibration import calibration_from_journal


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


@pytest.mark.mlops
def test_calibration_uses_realized_pnl_when_sample_is_sufficient(tmp_path):
    journal = tmp_path / "journal.jsonl"
    rows = [
        {
            "timestamp": f"2026-05-20T00:0{idx}:00+00:00",
            "decision": "LONG",
            "active_regime": "TREND",
            "execution": {"mode": "paper"},
            "book": {"id": "primary"},
            "realized_pnl": pnl,
        }
        for idx, pnl in enumerate([10.0, -5.0, 20.0, -10.0, 15.0])
    ]
    _write_rows(journal, rows)

    calibration = calibration_from_journal(str(journal), regime="TREND", min_samples=5)

    assert calibration.source == "journal_realized_pnl"
    assert calibration.sample_size == 5
    assert calibration.win_rate == 0.6
    assert calibration.win_loss_ratio == 2.0


@pytest.mark.mlops
def test_calibration_infers_equity_deltas_and_skips_bad_rows(tmp_path):
    journal = tmp_path / "journal.jsonl"
    rows = [
        {
            "timestamp": "2026-05-20T00:00:00+00:00",
            "action": 2,
            "execution": {"mode": "paper"},
            "book": {"id": "primary"},
            "regime": "MEAN_REVERSION",
            "equity": 1000.0,
        },
        {"not": "json-compatible-enough"},
        {
            "timestamp": "2026-05-20T00:01:00+00:00",
            "action": 2,
            "execution": {"mode": "paper"},
            "book": {"id": "primary"},
            "regime": "MEAN_REVERSION",
            "equity": 1010.0,
        },
        {
            "timestamp": "2026-05-20T00:02:00+00:00",
            "action": 0,
            "execution": {"mode": "paper"},
            "book": {"id": "primary"},
            "regime": "MEAN_REVERSION",
            "equity": 1005.0,
        },
    ]
    journal.write_text(
        "\n".join([json.dumps(rows[0]), "{bad-json", *map(json.dumps, rows[1:])]),
        encoding="utf-8",
    )

    calibration = calibration_from_journal(
        str(journal), regime="MEAN_REVERSION", min_samples=2
    )

    assert calibration.source == "journal_equity_delta"
    assert calibration.sample_size == 2
    assert calibration.win_rate == 0.5
    assert calibration.win_loss_ratio == 2.0


@pytest.mark.mlops
def test_calibration_falls_back_to_all_regimes_when_specific_regime_is_sparse(
    tmp_path,
):
    journal = tmp_path / "journal.jsonl"
    rows = []
    for idx, pnl in enumerate([4.0, -2.0, 6.0, -3.0]):
        rows.append(
            {
                "timestamp": f"2026-05-20T00:0{idx}:00+00:00",
                "decision": "SHORT",
                "active_regime": "TREND" if idx == 0 else "MEAN_REVERSION",
                "execution": {"mode": "paper"},
                "book": {"id": "primary"},
                "realized_pnl": pnl,
            }
        )
    _write_rows(journal, rows)

    calibration = calibration_from_journal(str(journal), regime="TREND", min_samples=3)

    assert calibration.source == "journal_realized_pnl_all_regime_fallback"
    assert calibration.regime == "TREND"
    assert calibration.sample_size == 4
    assert calibration.win_rate == 0.5


@pytest.mark.mlops
def test_calibration_uses_defaults_for_missing_or_insufficient_journal(tmp_path):
    missing = tmp_path / "missing.jsonl"
    missing_calibration = calibration_from_journal(
        str(missing),
        min_samples=3,
        default_win_rate=0.51,
        default_win_loss_ratio=1.1,
    )

    assert missing_calibration.source == "default_missing_journal"
    assert missing_calibration.win_rate == 0.51
    assert missing_calibration.win_loss_ratio == 1.1

    journal = tmp_path / "journal.jsonl"
    _write_rows(
        journal,
        [
            {
                "timestamp": "2026-05-20T00:00:00+00:00",
                "decision": "LONG",
                "execution": {"mode": "paper"},
                "book": {"id": "primary"},
                "realized_pnl": 10,
            }
        ],
    )

    sparse_calibration = calibration_from_journal(str(journal), min_samples=3)

    assert sparse_calibration.source == "default_insufficient_samples"
    assert sparse_calibration.sample_size == 1
