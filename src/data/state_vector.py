"""Build normalized 10-dim state vectors for RL/GBM agents."""

from typing import List

import pandas as pd

STATE_FEATURE_COLUMNS = [
    "price_momentum",
    "volume_accumulation",
    "cvd_signal",
    "buy_liquidity_sweep",
    "sell_liquidity_sweep",
    "eth_btc_beta",
    "eth_btc_zscore",
    "atr_norm",
    "volatility_zscore",
    "trend_slope",
]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_state_vector(row: pd.Series) -> List[float]:
    """
    Map a feature-enriched OHLCV row to the model input vector.
    Order must stay aligned with ExplainabilityEngine.FEATURE_MAP indices.
    """
    close = max(_safe_float(row.get("close"), 1.0), 1e-9)
    open_ = _safe_float(row.get("open"), close)
    volume = max(_safe_float(row.get("volume"), 0.0), 1e-9)

    momentum = (close - open_) / close
    vol_accum = _safe_float(row.get("net_volume"), 0.0) / volume
    cvd_raw = _safe_float(row.get("cvd"), 0.0)
    cvd_signal = max(min(cvd_raw / (volume * 100.0), 1.0), -1.0)

    buy_sweep = 1.0 if bool(row.get("is_buy_liquidity_sweep", False)) else 0.0
    sell_sweep = 1.0 if bool(row.get("is_sell_liquidity_sweep", False)) else 0.0

    beta = _safe_float(row.get("eth_btc_beta"), 1.0)
    zscore = _safe_float(row.get("eth_btc_zscore"), 0.0)
    atr_norm = _safe_float(row.get("atr"), 0.0) / close
    vol_z = _safe_float(row.get("volatility_zscore"), 0.0)
    trend = _safe_float(row.get("trend_slope"), 0.0)

    return [
        momentum,
        vol_accum,
        cvd_signal,
        buy_sweep,
        sell_sweep,
        beta,
        zscore,
        atr_norm,
        vol_z,
        trend,
    ]
