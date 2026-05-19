"""Hedge strategy attribution from journal and hedge-bandit decision logs."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.core.config_loader import load_config


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def generate_hedge_report(
    journal_path: str = "data_lake/trade_journal.jsonl",
    decision_path: str = "data_lake/hedge_bandit/training/decisions.jsonl",
    days: int = 7,
) -> Dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    by_strategy: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"selected_count": 0, "score_sum": 0.0, "score_count": 0, "pnl": 0.0}
    )

    def include(row: Dict[str, Any]) -> bool:
        ts = _parse_timestamp(row.get("timestamp"))
        return ts is None or ts >= cutoff

    for row in list(_iter_jsonl(Path(journal_path))) + list(
        _iter_jsonl(Path(decision_path))
    ):
        if not include(row):
            continue
        hedge = row.get("hedge") or {}
        if not hedge.get("enabled", False):
            continue

        selected = hedge.get("selected")
        if selected:
            by_strategy[selected]["selected_count"] += 1
            by_strategy[selected]["pnl"] += float(
                row.get("hedge_pnl", row.get("pnl", 0.0)) or 0.0
            )

        for name, score in (hedge.get("candidates") or {}).items():
            by_strategy[name]["score_sum"] += float(score)
            by_strategy[name]["score_count"] += 1

    strategies = {}
    for name, stats in by_strategy.items():
        count = stats["score_count"]
        strategies[name] = {
            "selected_count": stats["selected_count"],
            "score_observations": count,
            "avg_score": round(stats["score_sum"] / count, 4) if count else 0.0,
            "pnl": round(stats["pnl"], 8),
        }

    return {
        "window_days": days,
        "strategies": strategies,
        "total_selected": sum(v["selected_count"] for v in strategies.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="APEX hedge strategy report")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    journal_path = config.get("explainability", {}).get(
        "journal_path", "data_lake/trade_journal.jsonl"
    )
    decision_path = config.get("shadow", {}).get(
        "decision_log_path", "data_lake/hedge_bandit/training/decisions.jsonl"
    )
    print(
        json.dumps(
            generate_hedge_report(journal_path, decision_path, args.days),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
