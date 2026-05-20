import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import pandas as pd

from src.data.intervals import interval_to_timedelta

logger = logging.getLogger(__name__)


class DuckDBCacheManager:
    """
    High-performance analytical cache manager for the Data Lake using DuckDB.
    Supports incremental inserts, fast reads, and Parquet backups.
    """

    def __init__(self, db_path: str = "data_lake/apex.duckdb", read_only: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self._ensure_directories()
        self.conn = duckdb.connect(self.db_path, read_only=read_only)
        if not read_only:
            self._init_schemas()

    def _ensure_directories(self):
        """Ensures that the DuckDB and parquet data lake directories exist."""
        Path("data_lake/raw_ticks").mkdir(parents=True, exist_ok=True)
        Path("data_lake/ohlcv").mkdir(parents=True, exist_ok=True)
        Path("data_lake/features").mkdir(parents=True, exist_ok=True)
        Path("data_lake/orderflow").mkdir(parents=True, exist_ok=True)

    def _init_schemas(self):
        """Initializes tables if they do not exist."""
        # OHLCV Store
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                timeframe VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
        """
        )

        # Raw Ticks Store
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                price DOUBLE,
                quantity DOUBLE,
                is_buyer_maker BOOLEAN,
                trade_id BIGINT,
                PRIMARY KEY (symbol, trade_id)
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                timeframe VARCHAR,
                feature_set_id VARCHAR,
                features_json VARCHAR,
                PRIMARY KEY (symbol, timeframe, timestamp, feature_set_id)
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
                timestamp TIMESTAMP,
                book_id VARCHAR,
                equity DOUBLE,
                long_qty DOUBLE,
                short_qty DOUBLE,
                mark_price DOUBLE,
                regime VARCHAR,
                PRIMARY KEY (book_id, timestamp)
            )
        """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                funding_rate DOUBLE,
                open_interest DOUBLE,
                mark_price DOUBLE,
                PRIMARY KEY (symbol, timestamp)
            )
        """
        )

        logger.info("DuckDB schemas initialized.")

    def insert_ohlcv(self, df: pd.DataFrame):
        """
        Inserts a pandas DataFrame of OHLCV data into the database.
        Expected columns: timestamp, symbol, timeframe, open, high, low, close, volume
        """
        if df.empty:
            return

        try:
            # We use INSERT OR IGNORE / ON CONFLICT to avoid duplicate primary keys
            self.conn.execute(
                """
                INSERT INTO ohlcv
                SELECT * FROM df
                ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
            """
            )
            logger.info(f"Inserted {len(df)} OHLCV rows into DuckDB cache.")
        except Exception as e:
            logger.error(f"Failed to insert OHLCV data: {e}")

    def get_latest_timestamp(
        self, symbol: str, timeframe: str
    ) -> Optional[pd.Timestamp]:
        """Gets the latest cached timestamp for a symbol and timeframe."""
        result = self.conn.execute(
            """
            SELECT MAX(timestamp)
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """,
            [symbol, timeframe],
        ).fetchone()[0]
        return pd.Timestamp(result) if result else None

    def get_latest_tick_timestamp(self, symbol: str) -> Optional[pd.Timestamp]:
        result = self.conn.execute(
            """
            SELECT MAX(timestamp) FROM ticks WHERE symbol = ?
        """,
            [symbol],
        ).fetchone()[0]
        return pd.Timestamp(result) if result else None

    def insert_ticks(self, df: pd.DataFrame) -> int:
        """
        Insert raw aggTrade ticks. Columns:
        timestamp, symbol, price, quantity, is_buyer_maker, trade_id
        """
        if df.empty:
            return 0
        try:
            before = self.conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            self.conn.execute(
                """
                INSERT INTO ticks
                SELECT * FROM df
                ON CONFLICT (symbol, trade_id) DO NOTHING
            """
            )
            after = self.conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            inserted = after - before
            logger.info(f"Inserted {inserted} tick rows into DuckDB cache.")
            return inserted
        except Exception as exc:
            logger.error(f"Failed to insert ticks: {exc}")
            return 0

    def insert_features(
        self,
        df: pd.DataFrame,
        feature_set_id: str = "default",
    ) -> int:
        """
        Insert feature rows with timestamp, symbol, timeframe, and features.
        """
        if df.empty:
            return 0
        payload = df.copy()
        if "features_json" not in payload.columns:
            if "features" in payload.columns:
                payload["features_json"] = payload["features"].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
                )
            else:
                raise ValueError(
                    "features DataFrame requires features or features_json"
                )
        payload["feature_set_id"] = feature_set_id
        try:
            self.conn.execute(
                """
                INSERT INTO features
                SELECT
                    timestamp, symbol, timeframe, feature_set_id, features_json
                FROM payload
                ON CONFLICT (symbol, timeframe, timestamp, feature_set_id) DO NOTHING
            """
            )
            logger.info(f"Inserted {len(payload)} feature rows.")
            return len(payload)
        except Exception as exc:
            logger.error(f"Failed to insert features: {exc}")
            return 0

    def insert_market_snapshot(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
        mark_price: Optional[float] = None,
    ):
        self.conn.execute(
            """
            INSERT INTO market_snapshots
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (symbol, timestamp) DO UPDATE SET
                funding_rate = excluded.funding_rate,
                open_interest = excluded.open_interest,
                mark_price = excluded.mark_price
        """,
            [timestamp, symbol, funding_rate, open_interest, mark_price],
        )

    def detect_ohlcv_gaps(
        self, symbol: str, timeframe: str
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Returns list of (gap_start, gap_end) timestamps missing from OHLCV cache.
        """
        df = self.load_ohlcv(symbol, timeframe)
        if df.empty:
            return []

        step = interval_to_timedelta(timeframe)
        gap_threshold = step * 1.5
        df = df.sort_values("timestamp")
        gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        for i in range(1, len(df)):
            prev_ts = pd.Timestamp(df.iloc[i - 1]["timestamp"])
            curr_ts = pd.Timestamp(df.iloc[i]["timestamp"])
            delta = curr_ts - prev_ts
            if delta > gap_threshold:
                gaps.append((prev_ts + step, curr_ts - step))

        return gaps

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: str = None,
        end_time: str = None,
    ) -> pd.DataFrame:
        """Loads historical OHLCV data from DuckDB into a DataFrame."""
        query = """
            SELECT * FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params: list = [symbol, timeframe]
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        query += " ORDER BY timestamp ASC"
        return self.conn.execute(query, params).df()

    def backup_to_parquet(self):
        """Backs up database tables to partitioned Parquet files."""
        try:
            # Export OHLCV
            self.conn.execute(
                """
                COPY (SELECT * FROM ohlcv) 
                TO 'data_lake/ohlcv/' 
                (
                    FORMAT PARQUET,
                    PARTITION_BY (symbol, timeframe),
                    OVERWRITE_OR_IGNORE TRUE
                )
            """
            )
            logger.info("Backed up OHLCV to partitioned Parquet files.")
        except Exception as e:
            logger.error(f"Failed to backup to Parquet: {e}")

    def insert_paper_equity_snapshot(
        self,
        book_id: str,
        equity: float,
        long_qty: float,
        short_qty: float,
        mark_price: float,
        regime: str,
        timestamp: Optional[pd.Timestamp] = None,
    ):
        ts = timestamp or pd.Timestamp.now(tz="UTC")
        self.conn.execute(
            """
            INSERT INTO paper_equity_snapshots
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (book_id, timestamp) DO UPDATE SET
                equity = excluded.equity,
                long_qty = excluded.long_qty,
                short_qty = excluded.short_qty,
                mark_price = excluded.mark_price,
                regime = excluded.regime
        """,
            [ts, book_id, equity, long_qty, short_qty, mark_price, regime],
        )

    def load_paper_equity_snapshots(self, book_id: str = "primary") -> pd.DataFrame:
        return self.conn.execute(
            """
            SELECT * FROM paper_equity_snapshots
            WHERE book_id = ?
            ORDER BY timestamp ASC
        """,
            [book_id],
        ).df()

    def close(self):
        self.conn.close()
