import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class PerformanceCalibration:
    """Conservative sizing inputs derived from observed paper/live evidence."""

    win_rate: float
    win_loss_ratio: float
    sample_size: int
    source: str
    regime: Optional[str] = None
    mode: Optional[str] = None
    book_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calibration_from_journal(
    journal_path: str,
    *,
    mode: str = "paper",
    book_id: str = "primary",
    regime: Optional[str] = None,
    min_samples: int = 20,
    default_win_rate: float = 0.55,
    default_win_loss_ratio: float = 1.2,
) -> PerformanceCalibration:
    """
    Estimate Kelly inputs from journal outcomes, falling back to safe defaults.

    The journal currently contains a mix of decision, fill, hedge-bandit, and
    lifecycle rows. This helper intentionally uses only directional rows with
    explicit realized PnL or enough equity marks to infer deltas.
    """
    rows = _read_jsonl(journal_path)
    if not rows:
        return _default_calibration(
            "default_missing_journal",
            regime=regime,
            mode=mode,
            book_id=book_id,
            sample_size=0,
            default_win_rate=default_win_rate,
            default_win_loss_ratio=default_win_loss_ratio,
        )

    calibration = _calibrate_rows(
        rows,
        mode=mode,
        book_id=book_id,
        regime=regime,
        min_samples=min_samples,
        default_win_rate=default_win_rate,
        default_win_loss_ratio=default_win_loss_ratio,
    )
    if calibration.sample_size >= min_samples:
        return calibration

    if regime is not None:
        fallback = _calibrate_rows(
            rows,
            mode=mode,
            book_id=book_id,
            regime=None,
            min_samples=min_samples,
            default_win_rate=default_win_rate,
            default_win_loss_ratio=default_win_loss_ratio,
        )
        if fallback.sample_size >= min_samples:
            return PerformanceCalibration(
                win_rate=fallback.win_rate,
                win_loss_ratio=fallback.win_loss_ratio,
                sample_size=fallback.sample_size,
                source=f"{fallback.source}_all_regime_fallback",
                regime=regime,
                mode=mode,
                book_id=book_id,
            )

    return calibration


def _read_jsonl(journal_path: str) -> list[Dict[str, Any]]:
    path = Path(journal_path)
    rows: list[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except FileNotFoundError:
        return []
    return rows


def _calibrate_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    mode: str,
    book_id: str,
    regime: Optional[str],
    min_samples: int,
    default_win_rate: float,
    default_win_loss_ratio: float,
) -> PerformanceCalibration:
    matched = [
        row
        for row in rows
        if _matches_scope(row, mode=mode, book_id=book_id, regime=regime)
    ]
    explicit = _explicit_outcomes(matched)
    if explicit:
        return _from_outcomes(
            explicit,
            source="journal_realized_pnl",
            regime=regime,
            mode=mode,
            book_id=book_id,
            min_samples=min_samples,
            default_win_rate=default_win_rate,
            default_win_loss_ratio=default_win_loss_ratio,
        )

    inferred = _equity_delta_outcomes(matched)
    if inferred:
        return _from_outcomes(
            inferred,
            source="journal_equity_delta",
            regime=regime,
            mode=mode,
            book_id=book_id,
            min_samples=min_samples,
            default_win_rate=default_win_rate,
            default_win_loss_ratio=default_win_loss_ratio,
        )

    return _default_calibration(
        "default_no_directional_outcomes",
        regime=regime,
        mode=mode,
        book_id=book_id,
        sample_size=0,
        default_win_rate=default_win_rate,
        default_win_loss_ratio=default_win_loss_ratio,
    )


def _matches_scope(
    row: Dict[str, Any],
    *,
    mode: str,
    book_id: str,
    regime: Optional[str],
) -> bool:
    execution = row.get("execution") or {}
    if execution.get("mode") not in (None, mode):
        return False
    book = row.get("book") or {}
    if book.get("id") not in (None, book_id):
        return False
    if not _is_directional(row):
        return False
    if regime is None:
        return True
    return _row_regime(row) == regime


def _is_directional(row: Dict[str, Any]) -> bool:
    decision = str(row.get("decision", "")).upper()
    if decision in {"LONG", "SHORT"}:
        return True
    action = row.get("action")
    return action in (0, 2, "0", "2")


def _row_regime(row: Dict[str, Any]) -> Optional[str]:
    buckets = row.get("confidence_buckets") or {}
    regime_bucket = buckets.get("regime") or {}
    return (
        row.get("active_regime")
        or row.get("regime")
        or regime_bucket.get("active_regime")
    )


def _explicit_outcomes(rows: Iterable[Dict[str, Any]]) -> list[float]:
    outcomes = []
    pnl_keys = ("realized_pnl", "trade_pnl", "closed_pnl", "pnl_delta")
    for row in rows:
        value = None
        for key in pnl_keys:
            if key in row:
                value = _safe_float(row.get(key))
                break
        if value is not None and abs(value) > 1e-12:
            outcomes.append(value)
    return outcomes


def _equity_delta_outcomes(rows: Iterable[Dict[str, Any]]) -> list[float]:
    outcomes = []
    previous_equity: Optional[float] = None
    for row in sorted(rows, key=lambda item: str(item.get("timestamp", ""))):
        equity = _safe_float(row.get("equity"))
        if equity is None:
            continue
        if previous_equity is not None:
            delta = equity - previous_equity
            if abs(delta) > 1e-12:
                outcomes.append(delta)
        previous_equity = equity
    return outcomes


def _from_outcomes(
    outcomes: list[float],
    *,
    source: str,
    regime: Optional[str],
    mode: str,
    book_id: str,
    min_samples: int,
    default_win_rate: float,
    default_win_loss_ratio: float,
) -> PerformanceCalibration:
    wins = [value for value in outcomes if value > 0]
    losses = [abs(value) for value in outcomes if value < 0]
    sample_size = len(wins) + len(losses)
    if sample_size < min_samples:
        return _default_calibration(
            "default_insufficient_samples",
            regime=regime,
            mode=mode,
            book_id=book_id,
            sample_size=sample_size,
            default_win_rate=default_win_rate,
            default_win_loss_ratio=default_win_loss_ratio,
        )

    win_rate = _clamp(len(wins) / sample_size, 0.05, 0.95)
    if wins and losses:
        win_loss_ratio = _clamp(mean(wins) / mean(losses), 0.1, 10.0)
    else:
        win_loss_ratio = default_win_loss_ratio
    return PerformanceCalibration(
        win_rate=float(win_rate),
        win_loss_ratio=float(win_loss_ratio),
        sample_size=sample_size,
        source=source,
        regime=regime,
        mode=mode,
        book_id=book_id,
    )


def _default_calibration(
    source: str,
    *,
    regime: Optional[str],
    mode: str,
    book_id: str,
    sample_size: int,
    default_win_rate: float,
    default_win_loss_ratio: float,
) -> PerformanceCalibration:
    return PerformanceCalibration(
        win_rate=float(default_win_rate),
        win_loss_ratio=float(default_win_loss_ratio),
        sample_size=int(sample_size),
        source=source,
        regime=regime,
        mode=mode,
        book_id=book_id,
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
