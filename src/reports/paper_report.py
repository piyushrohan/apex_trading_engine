"""Paper trading performance report from equity snapshots and trade journal."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.config_loader import load_config
from src.data.cache_manager import DuckDBCacheManager


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return float(abs(dd.min()))


def _sharpe_from_equity(equity: pd.Series, bars_per_day: int = 480) -> float:
    if len(equity) < 3:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(bars_per_day * 365))


def load_journal_trades(journal_path: str, mode: str = "paper") -> List[dict]:
    path = Path(journal_path)
    if not path.exists():
        return []
    trades = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("execution", {}).get("mode") != mode:
            continue
        if row.get("decision") in ("LONG", "SHORT"):
            trades.append(row)
    return trades


def load_journal_fills(journal_path: str, mode: str = "paper") -> List[dict]:
    path = Path(journal_path)
    if not path.exists():
        return []
    fills = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("execution", {}).get("mode") != mode:
            continue
        if row.get("event") == "paper_fill":
            fills.append(row)
    return fills


def generate_paper_report(
    config: Optional[dict] = None,
    book_id: str = "primary",
    journal_path: str = "data_lake/trade_journal.jsonl",
) -> Dict[str, Any]:
    config = config or load_config()
    db_path = (
        config.get("data", {})
        .get("storage", {})
        .get("db_path", "data_lake/apex_market_data.duckdb")
    )
    if Path(db_path).exists():
        cache = None
        try:
            cache = DuckDBCacheManager(db_path=db_path, read_only=True)
            equity_df = cache.load_paper_equity_snapshots(book_id)
        except Exception:
            equity_df = pd.DataFrame()
        finally:
            if cache is not None:
                cache.close()
    else:
        equity_df = pd.DataFrame()

    journal_trades = load_journal_trades(journal_path)
    journal_fills = load_journal_fills(journal_path)
    non_flat = [t for t in journal_trades if t.get("book", {}).get("role") == "primary"]
    primary_fills = [
        f for f in journal_fills if f.get("book", {}).get("role") == "primary"
    ]
    fill_rate = len(primary_fills) / len(non_flat) if non_flat else None

    if equity_df.empty:
        return {
            "book_id": book_id,
            "snapshots": 0,
            "total_journal_decisions": len(journal_trades),
            "directional_decisions": len(non_flat),
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "final_equity": None,
            "filled_orders": len(primary_fills),
            "fill_rate": fill_rate,
        }

    equity = equity_df["equity"]
    report = {
        "book_id": book_id,
        "snapshots": len(equity_df),
        "start_equity": float(equity.iloc[0]),
        "final_equity": float(equity.iloc[-1]),
        "sharpe": round(_sharpe_from_equity(equity), 4),
        "max_drawdown": round(_max_drawdown(equity), 4),
        "total_journal_decisions": len(journal_trades),
        "directional_decisions": len(non_flat),
        "filled_orders": len(primary_fills),
        "fill_rate": round(fill_rate, 4) if fill_rate is not None else None,
        "long_signals": sum(1 for t in non_flat if t.get("decision") == "LONG"),
        "short_signals": sum(1 for t in non_flat if t.get("decision") == "SHORT"),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="APEX paper trading report")
    parser.add_argument(
        "--days", type=int, default=7, help="Reserved for future window filter"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    report = generate_paper_report(config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
