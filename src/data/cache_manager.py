import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBCacheManager:
    """
    High-performance analytical cache manager for the Data Lake using DuckDB.
    Supports incremental inserts, fast reads, and Parquet backups.
    """

    def __init__(self, db_path: str = "data_lake/apex.duckdb"):
        self.db_path = db_path
        self._ensure_directories()
        self.conn = duckdb.connect(self.db_path)
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
                trade_id BIGINT
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

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> pd.Timestamp:
        """Gets the latest cached timestamp for a symbol and timeframe."""
        query = f"""
            SELECT MAX(timestamp) 
            FROM ohlcv 
            WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        """
        result = self.conn.execute(query).fetchone()[0]
        return pd.Timestamp(result) if result else None

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: str = None,
        end_time: str = None,
    ) -> pd.DataFrame:
        """Loads historical OHLCV data from DuckDB into a DataFrame."""
        query = (
            "SELECT * FROM ohlcv "
            f"WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
        )

        if start_time:
            query += f" AND timestamp >= '{start_time}'"
        if end_time:
            query += f" AND timestamp <= '{end_time}'"

        query += " ORDER BY timestamp ASC"

        return self.conn.execute(query).df()

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

    def close(self):
        self.conn.close()
