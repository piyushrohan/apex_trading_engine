# Model Quality Upgrade Milestone

This milestone turns retraining from "fit a model and check OOS PnL" into a
more disciplined candidate-quality workflow. It does not claim the alpha problem
is solved; it makes weak candidates easier to diagnose and harder to promote.

## What Changed

The governed retrain path now records four extra quality layers:

1. **Fee-adjusted horizon labels**: labels use `label_horizon_bars` and the
   larger of `label_return_threshold` or `label_cost_buffer_bps`.
2. **Label quality report**: each candidate records label balance, directional
   label ratio, dominant label ratio, label entropy, and near-threshold label
   sensitivity.
3. **Class-balanced training weights**: GBM training can weight sparse classes
   so directional labels are not drowned by FLAT labels.
4. **Classifier probability diagnostics**: OOS probabilities are scored with
   accuracy, Brier score, expected calibration error, confidence, prediction
   counts, and trade-signal coverage.

The model can pass normal OOS and stress checks but still be rejected by the
model-quality gate if the data window is too short, labels are unstable, labels
are too imbalanced, or probability quality is too weak.

## Default Quality Gates

Configured in `configs/base.yaml`:

```yaml
mlops:
  label_horizon_bars: 3
  label_cost_buffer_bps: 4.0
  quality:
    min_history_days: 90
    min_directional_ratio: 0.12
    max_dominant_label_ratio: 0.80
    max_near_threshold_ratio: 0.45
    trade_probability_threshold: 0.55
    min_trade_signal_coverage: 0.05
    max_expected_calibration_error: 0.35
    max_brier_score: 0.80
```

These defaults intentionally block the current short-history training runs from
being treated as production-grade evidence. You can still train candidates, but
short data windows should remain research output, not promotion candidates.

## How To Read A Rejection

Run:

```bash
python -m src.mlops.auto_retrain
jq '.models["<model_id>"].metrics.quality_gate' data_lake/models/registry.json
jq '.models["<model_id>"].metrics.label_quality' data_lake/models/registry.json
jq '.models["<model_id>"].metrics.classifier_quality' data_lake/models/registry.json
```

Common blockers:

- `history_window_too_short`: the local DuckDB window is too small for robust
  crypto regime coverage.
- `label:directional_labels_too_sparse`: too few LONG/SHORT examples survived
  the fee-adjusted threshold.
- `label:dominant_label_too_high`: one class dominates the training target.
- `label:labels_too_close_to_threshold`: many labels are tiny price moves that
  could flip after noise, fees, or slippage.
- `classifier:calibration_error_too_high`: confidence does not match realized
  correctness on OOS labels.
- `classifier:trade_signal_coverage_too_low`: the model rarely produces strong
  directional signals.

## Better Model Recipe

Use this milestone as the first guardrail, then improve the data and alpha:

1. Backfill at least 90 days of 3m/5m data before expecting candidates to clear
   the quality gate.
2. Add cross-asset BTC/ETH context, funding, open interest, spread, order-book
   imbalance, taker flow, and weekend/session features.
3. Train challengers by horizon and regime, then compare them against the
   current champion through shadow-forward evidence.
4. Promote only models whose OOS, stress, walk-forward, label quality,
   probability calibration, and shadow evidence all agree.
