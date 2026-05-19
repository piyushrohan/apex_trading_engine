import logging
import uuid
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.cache_manager import DuckDBCacheManager
from src.mlops.evaluator import ModelEvaluator
from src.mlops.registry import ModelRegistry
from src.models.gbm_agent import GBMAgent
from src.models.ppo_agent import PPOAgent
from src.pipelines.backtest import BacktestEngine

logger = logging.getLogger(__name__)


class CandidateController:
    """Backtest adapter exposing the MetaController get_action contract."""

    def __init__(self, agent):
        self.agent = agent

    def get_action(self, state_vector: list, regime_str: str):
        return self.agent.act(state_vector)


class AutoRetrainPipeline:
    """
    Nightly MLOps pipeline.
    1. Extracts latest dataset from DuckDB.
    2. Trains a new candidate model.
    3. Evaluates OOS against strict safety gates.
    4. Auto-promotes to SHADOW if successful.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = ModelRegistry()
        self.evaluator = ModelEvaluator(config)
        self.cache = DuckDBCacheManager(
            config.get("data", {})
            .get("storage", {})
            .get("db_path", "data_lake/apex.duckdb")
        )

    def execute_nightly_retrain(self):
        """Main execution flow for the cron job."""
        logger.info("Starting Nightly Auto-Retrain Pipeline...")
        try:
            symbol = self.config.get("data", {}).get("target_symbol", "ETHUSDC")
            interval = self.config.get("data", {}).get("target_interval", "3m")

            logger.info(f"Extracting {symbol} {interval} data from DuckDB...")
            raw = self.cache.load_ohlcv(symbol, interval)
            min_rows = self.config.get("mlops", {}).get("min_training_rows", 30)
            if len(raw) < min_rows:
                logger.warning(
                    "Skipping retrain: only %s rows available, need %s",
                    len(raw),
                    min_rows,
                )
                return {"status": "skipped", "reason": "insufficient_data"}

            dataset = self._build_supervised_dataset(raw)
            split_idx = max(int(len(dataset) * 0.7), 1)
            train_df = dataset.iloc[:split_idx]
            oos_df = dataset.iloc[split_idx:]
            if oos_df.empty:
                oos_df = dataset.tail(min(len(dataset), 10))

            model_type = (
                self.config.get("mlops", {}).get("candidate_model_type", "GBM").upper()
            )
            new_model_id = f"{model_type.lower()}_ethusdc_v{uuid.uuid4().hex[:8]}"
            logger.info(f"Training new candidate model: {new_model_id}")

            agent = self._build_agent(model_type)
            train_metrics = agent.train(
                train_df[self._feature_columns()].to_numpy(),
                train_df["label"].to_numpy(),
            )

            metadata = {
                "data_rows": int(len(raw)),
                "train_rows": int(len(train_df)),
                "oos_rows": int(len(oos_df)),
            }
            try:
                model_path = self.registry.register_model(
                    new_model_id,
                    model_type,
                    {"training": train_metrics},
                    metadata=metadata,
                )
            except TypeError:
                model_path = self.registry.register_model(
                    new_model_id, model_type, {"training": train_metrics}
                )
            agent.save(model_path)

            logger.info("Evaluating candidate model Out-Of-Sample...")
            backtest = BacktestEngine(self.config)
            backtest.meta_controller = CandidateController(agent)
            pnl_series, trade_history = backtest.run(self._to_backtest_frame(oos_df))
            metrics = self.evaluator.evaluate_oos(pnl_series, trade_history)
            metrics["training"] = train_metrics
            self.registry.update_model_metrics(new_model_id, metrics)
            if hasattr(self.registry, "write_model_manifest"):
                self.registry.write_model_manifest(
                    new_model_id,
                    data_snapshot_id=self._data_snapshot_id(raw),
                    hyperparams=self.config.get("models", {}).get(
                        model_type.lower(), {}
                    ),
                    metrics=metrics,
                )

            if metrics.get("passed_safety", False):
                logger.info(
                    "Candidate %s passed safety gates. Promoting to SHADOW.",
                    new_model_id,
                )
                self.registry.promote_to_shadow(new_model_id)
            else:
                self.registry.set_model_status(new_model_id, "REJECTED")
                logger.warning(
                    "Candidate %s failed safety gates. Marking REJECTED.",
                    new_model_id,
                )

            logger.info("Nightly Auto-Retrain Pipeline completed.")
            return {"status": "completed", "model_id": new_model_id, "metrics": metrics}
        finally:
            self.cache.close()

    def _build_agent(self, model_type: str):
        if model_type == "PPO":
            return PPOAgent(state_dim=10, config=self.config)
        if model_type in ("GBM", "LIGHTGBM"):
            return GBMAgent(config=self.config)
        raise ValueError(f"Unsupported candidate model type: {model_type}")

    @staticmethod
    def _feature_columns() -> list[str]:
        return [f"feature_{idx}" for idx in range(10)]

    def _build_supervised_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a compact labeled feature matrix from OHLCV rows."""
        data = df.sort_values("timestamp").copy()
        close = data["close"].astype(float)
        returns = close.pct_change().fillna(0.0)
        future_returns = close.shift(-1).sub(close).div(close).fillna(0.0)
        vol = returns.rolling(10, min_periods=1).std().fillna(0.0)
        volume = data["volume"].astype(float)
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

        features = pd.DataFrame(
            {
                "timestamp": data["timestamp"],
                "close": close,
                "regime_str": "MEAN_REVERSION",
                "feature_0": returns,
                "feature_1": volume_z,
                "feature_2": returns.cumsum(),
                "feature_3": (future_returns > 0.001).astype(float),
                "feature_4": (future_returns < -0.001).astype(float),
                "feature_5": returns.rolling(5, min_periods=1).corr(volume).fillna(0),
                "feature_6": future_returns.rolling(5, min_periods=1).mean(),
                "feature_7": high_low.fillna(0.0),
                "feature_8": vol,
                "feature_9": trend.fillna(0.0),
            }
        )
        threshold = self.config.get("mlops", {}).get("label_return_threshold", 0.0005)
        features["label"] = np.select(
            [future_returns < -threshold, future_returns > threshold],
            [0, 2],
            default=1,
        )
        return features.dropna().reset_index(drop=True)

    def _to_backtest_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        frame = dataset[["timestamp", "close", "regime_str", *self._feature_columns()]]
        return frame.set_index("timestamp")

    @staticmethod
    def _data_snapshot_id(df: pd.DataFrame) -> str:
        start = pd.to_datetime(df["timestamp"].min()).isoformat()
        end = pd.to_datetime(df["timestamp"].max()).isoformat()
        return f"ohlcv:{start}:{end}:{len(df)}"


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline = AutoRetrainPipeline(config)
    pipeline.execute_nightly_retrain()
