import logging
import uuid
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.cache_manager import DuckDBCacheManager
from src.mlops.evaluator import ModelEvaluator
from src.mlops.experiment_tracker import ExperimentTracker, stable_hash
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

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        registry: Optional[ModelRegistry] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        self.config = config
        mlops_cfg = config.get("mlops", {})
        if registry is not None:
            self.registry = registry
        else:
            try:
                self.registry = ModelRegistry(
                    registry_dir=mlops_cfg.get("registry_dir", "data_lake/models")
                )
            except TypeError:
                self.registry = ModelRegistry()
        self.tracker = tracker or ExperimentTracker.from_config(config)
        self.evaluator = ModelEvaluator(config)
        self.cache = DuckDBCacheManager(
            config.get("data", {})
            .get("storage", {})
            .get("db_path", "data_lake/apex.duckdb")
        )

    def execute_nightly_retrain(self):
        """Main execution flow for the cron job."""
        logger.info("Starting Nightly Auto-Retrain Pipeline...")
        run = self.tracker.start_run(
            "candidate_retrain",
            metadata={
                "config_hash": stable_hash(self.config),
                "candidate_model_type": self.config.get("mlops", {}).get(
                    "candidate_model_type", "GBM"
                ),
            },
        )
        run_id = run["run_id"]
        new_model_id = None
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
                self.tracker.complete_run(
                    run_id,
                    "SKIPPED",
                    metadata={"reason": "insufficient_data", "rows": len(raw)},
                )
                return {
                    "status": "skipped",
                    "reason": "insufficient_data",
                    "run_id": run_id,
                }

            data_metadata = self._data_snapshot_metadata(raw)
            self.tracker.log_step(
                run_id,
                "data_snapshot",
                "PASSED",
                metadata=data_metadata,
            )

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
            self.tracker.log_step(
                run_id,
                "train",
                "PASSED",
                metrics=train_metrics,
                metadata={"model_id": new_model_id, "model_type": model_type},
            )

            metadata = {
                "run_id": run_id,
                "data_rows": int(len(raw)),
                "train_rows": int(len(train_df)),
                "oos_rows": int(len(oos_df)),
                "data_snapshot_id": data_metadata["data_snapshot_id"],
                "data_checksum": data_metadata["data_checksum"],
                "feature_version": self.config.get("mlops", {}).get(
                    "feature_version", "v1"
                ),
                "config_hash": stable_hash(self.config),
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
                    new_model_id,
                    model_type,
                    {"training": train_metrics},
                )
            agent.save(model_path)
            try:
                self.registry.set_model_status(
                    new_model_id,
                    "EVALUATING",
                    actor="auto_retrain",
                    reason="artifact_saved",
                )
            except TypeError:
                self.registry.set_model_status(new_model_id, "EVALUATING")

            logger.info("Evaluating candidate model Out-Of-Sample...")
            backtest = BacktestEngine(self.config)
            backtest.meta_controller = CandidateController(agent)
            pnl_series, trade_history = backtest.run(self._to_backtest_frame(oos_df))
            metrics = self.evaluator.evaluate_oos(pnl_series, trade_history)
            if hasattr(self.evaluator, "evaluate_stress"):
                stress_metrics = self.evaluator.evaluate_stress(
                    pnl_series, trade_history
                )
            else:
                stress_metrics = {"stress_passed": metrics.get("passed_safety", False)}
            walk_forward_metrics = self._walk_forward_validate(dataset, model_type)
            metrics["training"] = train_metrics
            metrics["stress"] = stress_metrics
            metrics["walk_forward"] = walk_forward_metrics
            self.registry.update_model_metrics(new_model_id, metrics)
            self.tracker.log_step(
                run_id,
                "oos_backtest",
                "PASSED" if metrics.get("passed_safety") else "FAILED",
                metrics=metrics,
                metadata={"model_id": new_model_id},
            )
            self.tracker.log_step(
                run_id,
                "stress",
                "PASSED" if stress_metrics.get("stress_passed") else "FAILED",
                metrics=stress_metrics,
                metadata={"model_id": new_model_id},
            )
            self.tracker.log_step(
                run_id,
                "walk_forward",
                "PASSED" if walk_forward_metrics.get("passed") else "FAILED",
                metrics=walk_forward_metrics,
                metadata={"model_id": new_model_id},
            )
            if hasattr(self.registry, "write_model_manifest"):
                self.registry.write_model_manifest(
                    new_model_id,
                    data_snapshot_id=data_metadata["data_snapshot_id"],
                    hyperparams=self.config.get("models", {}).get(
                        model_type.lower(), {}
                    ),
                    metrics=metrics,
                )

            passed_all = bool(
                metrics.get("passed_safety", False)
                and stress_metrics.get("stress_passed", False)
                and (
                    not walk_forward_metrics.get("required", False)
                    or walk_forward_metrics.get("passed", False)
                )
            )
            if passed_all:
                logger.info(
                    "Candidate %s passed safety gates. Promoting to SHADOW.",
                    new_model_id,
                )
                self.registry.promote_to_shadow(new_model_id)
                self.tracker.log_step(
                    run_id,
                    "shadow_registration",
                    "PASSED",
                    metadata={"model_id": new_model_id},
                )
            else:
                try:
                    self.registry.set_model_status(
                        new_model_id,
                        "REJECTED",
                        actor="auto_retrain",
                        reason="offline_stress_or_walk_forward_gate_failed",
                    )
                except TypeError:
                    self.registry.set_model_status(new_model_id, "REJECTED")
                logger.warning(
                    "Candidate %s failed safety gates. Marking REJECTED.",
                    new_model_id,
                )

            logger.info("Nightly Auto-Retrain Pipeline completed.")
            self.tracker.complete_run(
                run_id,
                "COMPLETED" if passed_all else "REJECTED",
                model_id=new_model_id,
                metrics=metrics,
            )
            return {
                "status": "completed",
                "model_id": new_model_id,
                "run_id": run_id,
                "metrics": metrics,
            }
        except Exception as exc:
            self.tracker.complete_run(
                run_id,
                "FAILED",
                model_id=new_model_id,
                metadata={"error": str(exc)},
            )
            raise
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

    def _walk_forward_validate(
        self, dataset: pd.DataFrame, model_type: str
    ) -> Dict[str, Any]:
        """Run expanding-window folds to expose temporal stability."""
        cfg = self.config.get("mlops", {}).get("walk_forward", {})
        if not cfg.get("enabled", True):
            return {
                "enabled": False,
                "required": bool(cfg.get("required", False)),
                "passed": True,
                "reason": "disabled",
                "folds": [],
            }

        total_rows = len(dataset)
        requested_folds = max(int(cfg.get("folds", 3)), 1)
        min_train_fraction = float(cfg.get("min_train_fraction", 0.45))
        min_train_rows = max(int(total_rows * min_train_fraction), 10)
        min_test_rows = max(int(cfg.get("min_test_rows", 10)), 1)
        remaining_rows = total_rows - min_train_rows
        if remaining_rows < min_test_rows:
            return {
                "enabled": True,
                "required": bool(cfg.get("required", False)),
                "passed": True,
                "reason": "insufficient_fold_data",
                "total_rows": int(total_rows),
                "folds": [],
            }

        fold_size = max(remaining_rows // requested_folds, min_test_rows)
        fold_results = []
        for fold_idx in range(requested_folds):
            train_end = min_train_rows + fold_idx * fold_size
            test_end = min(train_end + fold_size, total_rows)
            if test_end - train_end < min_test_rows:
                continue

            train_df = dataset.iloc[:train_end]
            test_df = dataset.iloc[train_end:test_end]
            agent = self._build_agent(model_type)
            train_metrics = agent.train(
                train_df[self._feature_columns()].to_numpy(),
                train_df["label"].to_numpy(),
            )
            backtest = BacktestEngine(self.config)
            backtest.meta_controller = CandidateController(agent)
            pnl_series, trade_history = backtest.run(self._to_backtest_frame(test_df))
            fold_metrics = self.evaluator.evaluate_oos(pnl_series, trade_history)
            fold_results.append(
                {
                    "fold": len(fold_results) + 1,
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "train_accuracy": train_metrics.get("train_accuracy"),
                    "passed_safety": bool(fold_metrics.get("passed_safety", False)),
                    "sharpe": float(fold_metrics.get("sharpe", 0.0)),
                    "max_drawdown": float(fold_metrics.get("max_drawdown", 1.0)),
                    "total_trades": int(fold_metrics.get("total_trades", 0)),
                }
            )

        if not fold_results:
            return {
                "enabled": True,
                "required": bool(cfg.get("required", False)),
                "passed": True,
                "reason": "no_valid_folds",
                "total_rows": int(total_rows),
                "folds": [],
            }

        pass_count = sum(1 for fold in fold_results if fold["passed_safety"])
        pass_rate = pass_count / len(fold_results)
        min_pass_rate = float(cfg.get("min_pass_rate", 0.66))
        return {
            "enabled": True,
            "required": bool(cfg.get("required", False)),
            "passed": pass_rate >= min_pass_rate,
            "pass_rate": float(pass_rate),
            "min_pass_rate": min_pass_rate,
            "fold_count": len(fold_results),
            "avg_sharpe": float(np.mean([fold["sharpe"] for fold in fold_results])),
            "worst_drawdown": float(max(fold["max_drawdown"] for fold in fold_results)),
            "total_trades": int(sum(fold["total_trades"] for fold in fold_results)),
            "folds": fold_results,
        }

    @staticmethod
    def _data_snapshot_id(df: pd.DataFrame) -> str:
        start = pd.to_datetime(df["timestamp"].min()).isoformat()
        end = pd.to_datetime(df["timestamp"].max()).isoformat()
        return f"ohlcv:{start}:{end}:{len(df)}"

    def _data_snapshot_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        snapshot_id = self._data_snapshot_id(df)
        checksum_frame = df.sort_values("timestamp").reset_index(drop=True)
        checksum = stable_hash(
            checksum_frame[["timestamp", "open", "high", "low", "close", "volume"]]
            .astype(str)
            .to_dict("records")
        )
        timestamps = pd.to_datetime(checksum_frame["timestamp"])
        gaps = 0
        if len(timestamps) > 2:
            expected_delta = timestamps.diff().dropna().mode()
            if not expected_delta.empty:
                gaps = int((timestamps.diff().dropna() > expected_delta.iloc[0]).sum())
        return {
            "data_snapshot_id": snapshot_id,
            "data_checksum": checksum,
            "rows": int(len(df)),
            "start": pd.to_datetime(df["timestamp"].min()).isoformat(),
            "end": pd.to_datetime(df["timestamp"].max()).isoformat(),
            "gaps": gaps,
        }


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline = AutoRetrainPipeline(config)
    pipeline.execute_nightly_retrain()
