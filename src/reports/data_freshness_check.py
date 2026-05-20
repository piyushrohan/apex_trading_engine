"""Data freshness and DuckDB integrity checks for the local data lake."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb

from src.core.config_loader import load_config
from src.data.cache_manager import DuckDBCacheManager
from src.reports.ops_common import (
    add_finding,
    age_minutes,
    should_exit_nonzero,
    status_from_findings,
    utc_now,
    write_report,
)

EXPECTED_TABLES = {
    "ohlcv",
    "ticks",
    "features",
    "paper_equity_snapshots",
    "market_snapshots",
}


def _warn_or_error(strict: bool) -> str:
    return "error" if strict else "warning"


def _fetchone(conn: duckdb.DuckDBPyConnection, sql: str, params=None) -> Any:
    return conn.execute(sql, params or []).fetchone()[0]


def _show_tables(conn: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in conn.execute("SHOW TABLES").fetchall()}


def _row_count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    return int(_fetchone(conn, f"SELECT COUNT(*) FROM {table}"))


def _latest(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    where_sql: str = "",
    params=None,
) -> Any:
    query = f"SELECT MAX(timestamp) FROM {table}"
    if where_sql:
        query += f" WHERE {where_sql}"
    return _fetchone(conn, query, params or [])


def _duplicate_count(conn: duckdb.DuckDBPyConnection, table: str, keys: str) -> int:
    query = f"""
        SELECT COUNT(*) FROM (
            SELECT {keys}, COUNT(*) AS row_count
            FROM {table}
            GROUP BY {keys}
            HAVING COUNT(*) > 1
        )
    """
    return int(_fetchone(conn, query))


def _null_count(conn: duckdb.DuckDBPyConnection, table: str, columns: list[str]) -> int:
    clauses = " OR ".join(f"{column} IS NULL" for column in columns)
    return int(_fetchone(conn, f"SELECT COUNT(*) FROM {table} WHERE {clauses}"))


def _add_freshness_finding(
    findings: list[Dict[str, Any]],
    *,
    strict: bool,
    table: str,
    code: str,
    latest_timestamp: Any,
    max_age_minutes: float,
    now: Optional[datetime],
) -> None:
    latest_age = age_minutes(latest_timestamp, now=now)
    if latest_age is None:
        add_finding(
            findings,
            _warn_or_error(strict),
            f"{code}_timestamp_missing",
            f"{table} has no latest timestamp for the requested scope.",
        )
    elif latest_age > max_age_minutes:
        add_finding(
            findings,
            _warn_or_error(strict),
            f"{code}_stale",
            f"{table} latest timestamp is stale.",
            age_minutes=round(latest_age, 2),
            max_age_minutes=max_age_minutes,
            latest_timestamp=str(latest_timestamp),
        )


def _feature_json_errors(
    conn: duckdb.DuckDBPyConnection,
    *,
    symbol: str,
    timeframe: str,
    limit: int = 1000,
) -> int:
    rows = conn.execute(
        """
        SELECT features_json
        FROM features
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        [symbol, timeframe, limit],
    ).fetchall()
    errors = 0
    for (payload,) in rows:
        try:
            json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            errors += 1
    return errors


def generate_data_freshness_report(
    config: Dict[str, Any],
    *,
    db_path: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    now: Optional[datetime] = None,
    max_ohlcv_age_minutes: float = 30.0,
    max_tick_age_minutes: float = 30.0,
    max_market_age_minutes: float = 60.0,
    max_feature_age_minutes: float = 60.0,
    max_equity_age_minutes: float = 60.0,
    strict: bool = False,
) -> Dict[str, Any]:
    """Inspect DuckDB table health, key integrity, gaps, and timestamp freshness."""
    data_cfg = config.get("data", {})
    db_path = db_path or data_cfg.get("storage", {}).get(
        "db_path", "data_lake/apex_market_data.duckdb"
    )
    symbol = symbol or data_cfg.get("target_symbol", "ETHUSDC")
    timeframe = timeframe or data_cfg.get("target_interval", "3m")
    findings: list[Dict[str, Any]] = []
    table_counts: Dict[str, int] = {}
    latest: Dict[str, Optional[str]] = {}
    gaps: list[tuple[str, str]] = []
    db_file = Path(db_path)

    if not db_file.exists():
        add_finding(
            findings,
            _warn_or_error(strict),
            "duckdb_missing",
            f"DuckDB database does not exist at {db_path}.",
        )
        return {
            "title": "APEX Data Freshness And DuckDB Integrity Check",
            "generated_at": (now or utc_now()).isoformat(),
            "status": status_from_findings(findings),
            "summary": {
                "db_path": db_path,
                "db_exists": False,
                "symbol": symbol,
                "timeframe": timeframe,
                "tables": {},
            },
            "findings": findings,
        }

    conn = None
    try:
        conn = duckdb.connect(str(db_file), read_only=True)
        tables = _show_tables(conn)
        missing = sorted(EXPECTED_TABLES - tables)
        for table in missing:
            add_finding(
                findings,
                _warn_or_error(strict),
                "duckdb_table_missing",
                "Expected DuckDB table is missing.",
                table=table,
            )

        for table in sorted(EXPECTED_TABLES & tables):
            table_counts[table] = _row_count(conn, table)
            if table_counts[table] == 0:
                add_finding(
                    findings,
                    _warn_or_error(strict),
                    "duckdb_table_empty",
                    "Expected DuckDB table has no rows.",
                    table=table,
                )

        if "ohlcv" in tables:
            latest_ohlcv = _latest(
                conn,
                "ohlcv",
                "symbol = ? AND timeframe = ?",
                [symbol, timeframe],
            )
            latest["ohlcv"] = str(latest_ohlcv) if latest_ohlcv else None
            _add_freshness_finding(
                findings,
                strict=strict,
                table="ohlcv",
                code="ohlcv",
                latest_timestamp=latest_ohlcv,
                max_age_minutes=max_ohlcv_age_minutes,
                now=now,
            )
            duplicate_keys = _duplicate_count(
                conn, "ohlcv", "symbol, timeframe, timestamp"
            )
            null_rows = _null_count(
                conn,
                "ohlcv",
                [
                    "timestamp",
                    "symbol",
                    "timeframe",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
            bad_volume = int(
                _fetchone(conn, "SELECT COUNT(*) FROM ohlcv WHERE volume < 0")
            )
            if duplicate_keys:
                add_finding(
                    findings,
                    "error",
                    "ohlcv_duplicate_keys",
                    "OHLCV table contains duplicate primary-key rows.",
                    count=duplicate_keys,
                )
            if null_rows:
                add_finding(
                    findings,
                    "error",
                    "ohlcv_null_required_fields",
                    "OHLCV table contains nulls in required fields.",
                    count=null_rows,
                )
            if bad_volume:
                add_finding(
                    findings,
                    "error",
                    "ohlcv_negative_volume",
                    "OHLCV table contains negative volume values.",
                    count=bad_volume,
                )
            cache = None
            try:
                cache = DuckDBCacheManager(str(db_file), read_only=True)
                gaps = [
                    (str(start), str(end))
                    for start, end in cache.detect_ohlcv_gaps(symbol, timeframe)
                ]
            finally:
                if cache is not None:
                    cache.close()
            if gaps:
                add_finding(
                    findings,
                    _warn_or_error(strict),
                    "ohlcv_gaps_detected",
                    "OHLCV cache has missing candle intervals.",
                    count=len(gaps),
                    sample=gaps[:5],
                )

        if "ticks" in tables:
            latest_tick = _latest(conn, "ticks", "symbol = ?", [symbol])
            latest["ticks"] = str(latest_tick) if latest_tick else None
            _add_freshness_finding(
                findings,
                strict=strict,
                table="ticks",
                code="ticks",
                latest_timestamp=latest_tick,
                max_age_minutes=max_tick_age_minutes,
                now=now,
            )
            duplicates = _duplicate_count(conn, "ticks", "symbol, trade_id")
            if duplicates:
                add_finding(
                    findings,
                    "error",
                    "tick_duplicate_keys",
                    "Ticks table contains duplicate trade ids.",
                    count=duplicates,
                )

        if "market_snapshots" in tables:
            latest_market = _latest(conn, "market_snapshots", "symbol = ?", [symbol])
            latest["market_snapshots"] = str(latest_market) if latest_market else None
            _add_freshness_finding(
                findings,
                strict=strict,
                table="market_snapshots",
                code="market_snapshots",
                latest_timestamp=latest_market,
                max_age_minutes=max_market_age_minutes,
                now=now,
            )
            duplicates = _duplicate_count(conn, "market_snapshots", "symbol, timestamp")
            if duplicates:
                add_finding(
                    findings,
                    "error",
                    "market_snapshot_duplicate_keys",
                    "Market snapshots contain duplicate keys.",
                    count=duplicates,
                )

        if "features" in tables:
            latest_features = _latest(
                conn,
                "features",
                "symbol = ? AND timeframe = ?",
                [symbol, timeframe],
            )
            latest["features"] = str(latest_features) if latest_features else None
            _add_freshness_finding(
                findings,
                strict=strict,
                table="features",
                code="features",
                latest_timestamp=latest_features,
                max_age_minutes=max_feature_age_minutes,
                now=now,
            )
            duplicates = _duplicate_count(
                conn, "features", "symbol, timeframe, timestamp, feature_set_id"
            )
            json_errors = _feature_json_errors(conn, symbol=symbol, timeframe=timeframe)
            if duplicates:
                add_finding(
                    findings,
                    "error",
                    "feature_duplicate_keys",
                    "Feature cache contains duplicate feature-set rows.",
                    count=duplicates,
                )
            if json_errors:
                add_finding(
                    findings,
                    "error",
                    "feature_json_invalid",
                    "Feature cache contains invalid JSON payloads.",
                    count=json_errors,
                )

        if "paper_equity_snapshots" in tables:
            latest_equity = _latest(conn, "paper_equity_snapshots")
            latest["paper_equity_snapshots"] = (
                str(latest_equity) if latest_equity else None
            )
            _add_freshness_finding(
                findings,
                strict=strict,
                table="paper_equity_snapshots",
                code="paper_equity_snapshots",
                latest_timestamp=latest_equity,
                max_age_minutes=max_equity_age_minutes,
                now=now,
            )
            duplicates = _duplicate_count(
                conn, "paper_equity_snapshots", "book_id, timestamp"
            )
            if duplicates:
                add_finding(
                    findings,
                    "error",
                    "paper_equity_duplicate_keys",
                    "Paper equity snapshots contain duplicate keys.",
                    count=duplicates,
                )
    except Exception as exc:
        add_finding(
            findings,
            "error",
            "duckdb_unreadable",
            "DuckDB database could not be opened or scanned.",
            error=str(exc),
        )
    finally:
        if conn is not None:
            conn.close()

    summary = {
        "db_path": db_path,
        "db_exists": True,
        "db_size_bytes": db_file.stat().st_size,
        "symbol": symbol,
        "timeframe": timeframe,
        "tables": table_counts,
        "latest": latest,
        "ohlcv_gap_count": len(gaps),
    }
    return {
        "title": "APEX Data Freshness And DuckDB Integrity Check",
        "generated_at": (now or utc_now()).isoformat(),
        "status": status_from_findings(findings),
        "summary": summary,
        "ohlcv_gaps": gaps[:50],
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="APEX data freshness and DuckDB integrity check"
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--db-path")
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe")
    parser.add_argument("--max-ohlcv-age-minutes", type=float, default=30.0)
    parser.add_argument("--max-tick-age-minutes", type=float, default=30.0)
    parser.add_argument("--max-market-age-minutes", type=float, default=60.0)
    parser.add_argument("--max-feature-age-minutes", type=float, default=60.0)
    parser.add_argument("--max-equity-age-minutes", type=float, default=60.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    report = generate_data_freshness_report(
        config,
        db_path=args.db_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        max_ohlcv_age_minutes=args.max_ohlcv_age_minutes,
        max_tick_age_minutes=args.max_tick_age_minutes,
        max_market_age_minutes=args.max_market_age_minutes,
        max_feature_age_minutes=args.max_feature_age_minutes,
        max_equity_age_minutes=args.max_equity_age_minutes,
        strict=args.strict,
    )
    rendered = write_report(report, output=args.output, fmt=args.format)
    if not args.output:
        print(rendered)
    if should_exit_nonzero(report, fail_on_warning=args.fail_on_warning):
        sys.exit(1)


if __name__ == "__main__":
    main()
