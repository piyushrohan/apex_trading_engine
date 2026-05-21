import pandas as pd
import pytest

from src.mlops.feature_drift import (
    build_feature_reference,
    compare_feature_drift,
    latest_feature_frame_from_ohlcv,
)


@pytest.mark.mlops
def test_feature_reference_and_drift_detect_ok_and_warning():
    train = pd.DataFrame({"feature_0": [0.0, 1.0, 2.0], "feature_1": [5.0, 5.0, 5.0]})
    reference = build_feature_reference(train, ["feature_0", "feature_1"])

    ok = compare_feature_drift(reference, pd.DataFrame({"feature_0": [1.0]}))
    drifted = compare_feature_drift(
        reference,
        pd.DataFrame({"feature_0": [9.0], "feature_1": [5.0]}),
        warning_z=2.0,
        critical_z=6.0,
    )

    assert reference["rows"] == 3
    assert ok["status"] == "ok"
    assert drifted["status"] == "critical"
    assert drifted["max_abs_z"] > 6


@pytest.mark.mlops
def test_feature_drift_reports_unavailable_without_reference_or_current_rows():
    missing = compare_feature_drift({}, pd.DataFrame({"feature_0": [1.0]}))
    empty = compare_feature_drift(
        {"features": {"feature_0": {"mean": 0, "std": 1}}}, pd.DataFrame()
    )

    assert missing["reason"] == "missing_feature_reference"
    assert empty["reason"] == "empty_current_feature_frame"


@pytest.mark.mlops
def test_latest_feature_frame_from_ohlcv_builds_expected_columns():
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-21", periods=12, freq="3min"),
            "open": range(100, 112),
            "high": range(101, 113),
            "low": range(99, 111),
            "close": range(100, 112),
            "volume": range(10, 22),
        }
    )

    features = latest_feature_frame_from_ohlcv(raw)

    assert list(features.columns) == [f"feature_{idx}" for idx in range(10)]
    assert len(features) == len(raw)
    assert features["feature_8"].iloc[-1] >= 0


@pytest.mark.mlops
def test_feature_drift_handles_missing_empty_and_critical_features():
    reference = build_feature_reference(
        pd.DataFrame(
            {
                "feature_ok": [1.0, 1.0, 1.0],
                "feature_empty_reference": [None, None, None],
                "feature_missing": [2.0, 2.0, 2.0],
                "feature_empty_current": [3.0, 3.0, 3.0],
                "feature_critical": [0.0, 0.0, 0.0],
            }
        ),
        [
            "feature_ok",
            "feature_empty_reference",
            "feature_missing",
            "feature_empty_current",
            "feature_critical",
        ],
    )

    report = compare_feature_drift(
        reference,
        pd.DataFrame(
            {
                "feature_ok": [1.0],
                "feature_empty_current": [None],
                "feature_critical": [10.0],
            }
        ),
        warning_z=2.0,
        critical_z=3.0,
    )

    statuses = {row["feature"]: row["status"] for row in report["features"]}
    assert "feature_empty_reference" not in reference["features"]
    assert report["status"] == "critical"
    assert statuses["feature_missing"] == "missing_current_feature"
    assert statuses["feature_empty_current"] == "empty_current_feature"
    assert statuses["feature_critical"] == "critical"

    empty_ohlcv = latest_feature_frame_from_ohlcv(
        pd.DataFrame(), ["feature_0", "feature_1"]
    )
    assert list(empty_ohlcv.columns) == ["feature_0", "feature_1"]
