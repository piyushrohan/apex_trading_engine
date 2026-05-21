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
            min_supervised_rows = self.config.get("mlops", {}).get(
                "min_supervised_rows", min_rows
            )
            if len(dataset) < min_supervised_rows:
                logger.warning(
                    "Skipping retrain: only %s supervised rows available, need %s",
                    len(dataset),
                    min_supervised_rows,
                )
                self.tracker.complete_run(
                    run_id,
                    "SKIPPED",
                    metadata={
                        "reason": "insufficient_supervised_data",
                        "rows": len(dataset),
                    },
                )
                return {
                    "status": "skipped",
                    "reason": "insufficient_supervised_data",
                    "run_id": run_id,
                }
            model_type = (
                self.config.get("mlops", {}).get("candidate_model_type", "GBM").upper()
            )
            new_model_id = f"{model_type.lower()}_ethusdc_v{uuid.uuid4().hex[:8]}"
            label_quality = self._label_quality_report(dataset)
            self.tracker.log_step(
                run_id,
                "label_quality",
                "PASSED" if label_quality["passed"] else "FAILED",
                metrics=label_quality,
                metadata={"model_id": new_model_id},
            )
            split_idx = max(int(len(dataset) * 0.7), 1)
            train_df = dataset.iloc[:split_idx]
            oos_df = dataset.iloc[split_idx:]
            if oos_df.empty:
                oos_df = dataset.tail(min(len(dataset), 10))

            logger.info(f"Training new candidate model: {new_model_id}")

            agent = self._build_agent(model_type)
            train_metrics = agent.train(
                train_df[self._feature_columns()].to_numpy(),
                train_df["label"].to_numpy(),
                self._sample_weights(train_df),
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
            classifier_quality = self._classifier_quality_report(agent, oos_df)
            if hasattr(self.evaluator, "evaluate_stress"):
                stress_metrics = self.evaluator.evaluate_stress(
                    pnl_series, trade_history
                )
            else:
                stress_metrics = {"stress_passed": metrics.get("passed_safety", False)}
            walk_forward_metrics = self._walk_forward_validate(dataset, model_type)
            metrics["training"] = train_metrics
            metrics["label_quality"] = label_quality
            metrics["classifier_quality"] = classifier_quality
            metrics["stress"] = stress_metrics
            metrics["walk_forward"] = walk_forward_metrics
            quality_gate = self._quality_gate(
                data_metadata=data_metadata,
                label_quality=label_quality,
                classifier_quality=classifier_quality,
            )
            metrics["quality_gate"] = quality_gate
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
            self.tracker.log_step(
                run_id,
                "model_quality_gate",
                "PASSED" if quality_gate.get("passed") else "FAILED",
                metrics=quality_gate,
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
                and quality_gate.get("passed", False)
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
                    rejection_reason = (
                        "model_quality_gate_failed"
                        if not quality_gate.get("passed", False)
                        else "offline_stress_or_walk_forward_gate_failed"
                    )
                    self.registry.set_model_status(
                        new_model_id,
                        "REJECTED",
                        actor="auto_retrain",
                        reason=rejection_reason,
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
        mlops_cfg = self.config.get("mlops", {})
        horizon = max(int(mlops_cfg.get("label_horizon_bars", 1)), 1)
        threshold = float(mlops_cfg.get("label_return_threshold", 0.0005))
        cost_buffer = float(mlops_cfg.get("label_cost_buffer_bps", 0.0)) / 10000.0
        label_threshold = max(threshold, cost_buffer)
        future_returns = close.shift(-horizon).sub(close).div(close)
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
                "feature_3": (returns > threshold).astype(float),
                "feature_4": (returns < -threshold).astype(float),
                "feature_5": returns.rolling(5, min_periods=1).corr(volume).fillna(0),
                "feature_6": returns.rolling(5, min_periods=1).mean(),
                "feature_7": high_low.fillna(0.0),
                "feature_8": vol,
                "feature_9": trend.fillna(0.0),
            }
        )
        features["future_return"] = future_returns
        features["label_threshold"] = label_threshold
        features["label_horizon_bars"] = horizon
        features["label"] = np.select(
            [future_returns < -label_threshold, future_returns > label_threshold],
            [0, 2],
            default=1,
        )
        return features.dropna().reset_index(drop=True)

    def _to_backtest_frame(self, dataset: pd.DataFrame) -> pd.DataFrame:
        frame = dataset[["timestamp", "close", "regime_str", *self._feature_columns()]]
        return frame.set_index("timestamp")

    def _sample_weights(self, dataset: pd.DataFrame) -> np.ndarray:
        """Weight rare directional labels higher without changing labels."""
        cfg = self.config.get("mlops", {}).get("quality", {})
        if not cfg.get("class_balance_weights", True):
            return np.ones(len(dataset), dtype=float)

        labels = dataset["label"].astype(int)
        counts = labels.value_counts().to_dict()
        total = max(len(labels), 1)
        classes = max(len(counts), 1)
        weights = labels.map(
            lambda label: total / (classes * max(int(counts.get(int(label), 0)), 1))
        ).astype(float)
        cap = float(cfg.get("max_sample_weight", 5.0))
        return weights.clip(lower=0.1, upper=cap).to_numpy()

    def _label_quality_report(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Describe whether labels are balanced, tradeable, and stable."""
        cfg = self.config.get("mlops", {}).get("quality", {})
        labels = dataset["label"].astype(int)
        total = int(len(labels))
        counts = {
            str(label): int(count) for label, count in labels.value_counts().items()
        }
        if total == 0:
            return {
                "passed": False,
                "blockers": ["empty_supervised_dataset"],
                "rows": 0,
                "label_counts": {},
            }

        directional = int((labels != 1).sum())
        ratios = labels.value_counts(normalize=True).to_dict()
        dominant_label_ratio = float(max(ratios.values())) if ratios else 1.0
        directional_ratio = float(directional / total)
        future_abs = dataset["future_return"].astype(float).abs()
        thresholds = dataset["label_threshold"].astype(float)
        band_fraction = float(cfg.get("near_threshold_band_fraction", 0.20))
        near_threshold = future_abs.sub(thresholds).abs() <= thresholds * band_fraction
        probabilities = np.array(list(ratios.values()), dtype=float)
        entropy = 0.0
        if len(probabilities) > 1:
            entropy = float(
                -np.sum(probabilities * np.log(probabilities))
                / np.log(min(3, len(probabilities)))
            )

        blockers = []
        min_directional = float(cfg.get("min_directional_ratio", 0.0))
        max_dominant = float(cfg.get("max_dominant_label_ratio", 1.0))
        max_near = float(cfg.get("max_near_threshold_ratio", 1.0))
        if directional_ratio < min_directional:
            blockers.append("directional_labels_too_sparse")
        if dominant_label_ratio > max_dominant:
            blockers.append("dominant_label_too_high")
        near_ratio = float(near_threshold.mean())
        if near_ratio > max_near:
            blockers.append("labels_too_close_to_threshold")

        return {
            "passed": not blockers,
            "blockers": blockers,
            "rows": total,
            "label_counts": counts,
            "directional_ratio": directional_ratio,
            "dominant_label_ratio": dominant_label_ratio,
            "near_threshold_ratio": near_ratio,
            "label_entropy": entropy,
            "threshold": float(thresholds.iloc[0]),
            "horizon_bars": int(dataset["label_horizon_bars"].iloc[0]),
        }

    def _classifier_quality_report(
        self, agent, dataset: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate probability calibration on the OOS label set."""
        if dataset.empty:
            return {
                "passed": False,
                "blockers": ["empty_oos_dataset"],
                "samples": 0,
            }

        probs = self._predict_probabilities(agent, dataset[self._feature_columns()])
        labels = dataset["label"].astype(int).to_numpy()
        predicted = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        accuracy = predicted == labels
        one_hot = np.eye(probs.shape[1])[labels]
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        ece = self._expected_calibration_error(confidence, accuracy)
        threshold = float(
            self.config.get("mlops", {})
            .get("quality", {})
            .get("trade_probability_threshold", 0.55)
        )
        directional = predicted != 1
        trade_coverage = float(((confidence >= threshold) & directional).mean())

        cfg = self.config.get("mlops", {}).get("quality", {})
        blockers = []
        if brier > float(cfg.get("max_brier_score", 1.0)):
            blockers.append("brier_score_too_high")
        if ece > float(cfg.get("max_expected_calibration_error", 1.0)):
            blockers.append("calibration_error_too_high")
        if trade_coverage < float(cfg.get("min_trade_signal_coverage", 0.0)):
            blockers.append("trade_signal_coverage_too_low")

        return {
            "passed": not blockers,
            "blockers": blockers,
            "samples": int(len(labels)),
            "accuracy": float(accuracy.mean()),
            "brier_score": brier,
            "expected_calibration_error": ece,
            "avg_confidence": float(confidence.mean()),
            "trade_signal_coverage": trade_coverage,
            "trade_probability_threshold": threshold,
            "prediction_counts": {
                str(label): int((predicted == label).sum())
                for label in range(probs.shape[1])
            },
        }

    @staticmethod
    def _predict_probabilities(agent, features: pd.DataFrame) -> np.ndarray:
        x_arr = features.to_numpy(dtype=float)
        if hasattr(agent, "model") and hasattr(agent.model, "predict_proba"):
            return np.asarray(agent.model.predict_proba(x_arr), dtype=float)
        predictions = [
            agent.act(row.tolist())[2].get("action_probs")
            for _, row in features.iterrows()
        ]
        return np.asarray(predictions, dtype=float)

    @staticmethod
    def _expected_calibration_error(
        confidence: np.ndarray, correct: np.ndarray, bins: int = 10
    ) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for left, right in zip(edges[:-1], edges[1:]):
            in_bin = (confidence > left) & (confidence <= right)
            if not in_bin.any():
                continue
            bin_weight = in_bin.mean()
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidence[in_bin].mean()
            ece += float(bin_weight * abs(bin_accuracy - bin_confidence))
        return float(ece)

    def _quality_gate(
        self,
        *,
        data_metadata: Dict[str, Any],
        label_quality: Dict[str, Any],
        classifier_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate data, label, and probability evidence into promotion blockers."""
        cfg = self.config.get("mlops", {}).get("quality", {})
        blockers = []
        min_history_days = float(cfg.get("min_history_days", 0.0))
        history_days = float(data_metadata.get("history_days", 0.0))
        if history_days < min_history_days:
            blockers.append("history_window_too_short")
        blockers.extend(
            f"label:{blocker}" for blocker in label_quality.get("blockers", [])
        )
        blockers.extend(
            f"classifier:{blocker}"
            for blocker in classifier_quality.get("blockers", [])
        )
        return {
            "passed": not blockers,
            "blockers": blockers,
            "history_days": history_days,
            "min_history_days": min_history_days,
            "label_passed": bool(label_quality.get("passed", False)),
            "classifier_passed": bool(classifier_quality.get("passed", False)),
        }

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
        history_days = 0.0
        if len(timestamps) > 1:
            history_days = float(
                (timestamps.max() - timestamps.min()).total_seconds() / 86400.0
            )
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
            "history_days": history_days,
            "gaps": gaps,
        }


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline = AutoRetrainPipeline(config)
    pipeline.execute_nightly_retrain()
