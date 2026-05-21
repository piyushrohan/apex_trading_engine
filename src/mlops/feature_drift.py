"""Feature drift diagnostics for active model evidence."""

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def build_feature_reference(
    frame: pd.DataFrame, feature_columns: Iterable[str]
) -> Dict[str, Any]:
    """Build compact training-distribution evidence for model manifests."""
    columns = list(feature_columns)
    refs: Dict[str, Dict[str, float]] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        refs[column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "p05": float(series.quantile(0.05)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
        }
    return {"rows": int(len(frame)), "features": refs}


def compare_feature_drift(
    reference: Mapping[str, Any],
    current: pd.DataFrame,
    *,
    warning_z: float = 2.5,
    critical_z: float = 4.0,
) -> Dict[str, Any]:
    """Compare current feature means against the stored training reference."""
    features = reference.get("features", {}) if reference else {}
    if not features:
        return {
            "status": "unavailable",
            "reason": "missing_feature_reference",
            "features": [],
            "max_abs_z": None,
        }
    if current.empty:
        return {
            "status": "unavailable",
            "reason": "empty_current_feature_frame",
            "features": [],
            "max_abs_z": None,
        }

    rows = []
    max_abs_z = 0.0
    for name, stats in features.items():
        if name not in current:
            rows.append({"feature": name, "status": "missing_current_feature"})
            continue
        series = pd.to_numeric(current[name], errors="coerce").dropna()
        if series.empty:
            rows.append({"feature": name, "status": "empty_current_feature"})
            continue
        ref_std = max(abs(float(stats.get("std") or 0.0)), 1e-9)
        current_mean = float(series.mean())
        z_score = (current_mean - float(stats.get("mean", 0.0))) / ref_std
        max_abs_z = max(max_abs_z, abs(z_score))
        if abs(z_score) >= critical_z:
            status = "critical"
        elif abs(z_score) >= warning_z:
            status = "warning"
        else:
            status = "ok"
        rows.append(
            {
                "feature": name,
                "status": status,
                "reference_mean": float(stats.get("mean", 0.0)),
                "current_mean": current_mean,
                "reference_std": ref_std,
                "z_score": float(z_score),
                "current_p05": float(series.quantile(0.05)),
                "current_p95": float(series.quantile(0.95)),
            }
        )

    if any(row.get("status") == "critical" for row in rows):
        status = "critical"
    elif any(row.get("status") == "warning" for row in rows):
        status = "warning"
    else:
        status = "ok"
    return {
        "status": status,
        "features": rows,
        "max_abs_z": float(max_abs_z),
        "current_rows": int(len(current)),
        "reference_rows": int(reference.get("rows") or 0),
    }


def latest_feature_frame_from_ohlcv(
    ohlcv: pd.DataFrame, feature_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """Build the same compact feature set used by auto-retrain from OHLCV."""
    if ohlcv.empty:
        return pd.DataFrame(columns=feature_columns or [])
    data = ohlcv.sort_values("timestamp").copy()
    close = data["close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    volume = data["volume"].astype(float)
    vol = returns.rolling(10, min_periods=1).std().fillna(0.0)
    volume_z = (
        (volume - volume.rolling(10, min_periods=1).mean())
        / volume.rolling(10, min_periods=1).std().replace(0, np.nan)
    ).fillna(0.0)
    high_low = (data["high"].astype(float) - data["low"].astype(float)) / close
    trend = (
        close.ewm(span=5, adjust=False)
        .mean()
        .sub(close.ewm(span=15, adjust=False).mean())
        .div(close)
    )
    threshold = 0.0005
    frame = pd.DataFrame(
        {
            "feature_0": returns,
            "feature_1": volume_z,
            "feature_2": returns.cumsum(),
            "feature_3": (returns > threshold).astype(float),
            "feature_4": (returns < -threshold).astype(float),
            "feature_5": returns.rolling(5, min_periods=1).corr(volume).fillna(0),
            "feature_6": returns.rolling(5, min_periods=1).mean(),
            "feature_7": high_low.fillna(0.0),
            "feature_8": vol,
            "feature_9": trend.fillna(0.0),
        }
    )
    if feature_columns:
        return frame[[col for col in feature_columns if col in frame]]
    return frame
