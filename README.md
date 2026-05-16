# APEX Institutional AI Trading System

APEX is a production-grade, fully autonomous AI trading system built specifically for **Binance Futures (ETHUSDC)**. Engineered as an institutional-grade execution desk, the system emphasizes zero-fee maker execution, continuous machine learning (MLOps), real-time risk management, and complete explainability of the AI's decision-making process.

## 🌟 Core Philosophy

The system rejects the "black box" approach to AI trading. At all times, APEX provides "X-ray vision" into its reasoning, detailing precisely why a trade was opened, what market regime is active, and the confidence intervals of the underlying signals. 

- **Capital Preservation**: Strict daily drawdown kill-switches and dynamic Kelly-criterion position sizing.
- **Explainability**: Every decision generates structured reasoning, weighted signal contributions, and regime explanations.
- **Zero-Fee Execution**: Optimized exclusively for the ETHUSDC perpetual contract using `timeInForce="GTX"` (Post-Only) limits.

## 🏗️ Architecture

The APEX engine is divided into several independent, highly resilient modules:

1. **Data Lake & Caching (`src/data/`)**: High-performance local data storage powered by `DuckDB` and Parquet for incremental raw tick and OHLCV storage.
2. **Execution & Risk (`src/execution/`)**: Maker-only limit order management, slippage alpha-decay tracking, and live account reconciliation via Binance User Data Streams.
3. **Intelligence Layer (`src/models/`)**: Adaptive regime detection, cumulative volume delta (CVD) calculations, and reinforcement learning / gradient boosting agents.
4. **MLOps Pipeline (`src/mlops/`)**: Continuous retraining, shadow testing, and automated model promotion to production.

## 📁 Repository Structure

```text
apex_trading_engine/
├── configs/                  # Environment, risk, and model configuration yaml files
├── data_lake/                # Local data storage (DuckDB / Parquet)
├── src/
│   ├── core/                 # Config loaders, loggers, and AES key encryption
│   ├── data/                 # Async WebSockets, REST backfills, feature engineering
│   ├── execution/            # Risk Engine, Maker-Only orders, Position Syncing
│   ├── models/               # PPO/GBM agents, meta-controllers, regime detectors
│   ├── mlops/                # Evaluators, explainability, auto-promotion registry
│   └── pipelines/            # Live trading, shadow trading, and backtest loops
├── frontend/                 # Institutional Terminal UI (React)
└── tests/                    # Unit and Integration test suite
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- `pip install -r requirements.txt`

### Security Setup
APEX uses symmetric encryption (Fernet) to protect API keys. Keys are never stored in plaintext.

1. Set your master key as an environment variable:
   ```bash
   export APEX_MASTER_KEY="<your_generated_fernet_key>"
   ```
2. The system will securely encrypt and save your Binance API keys to `configs/.keys.enc` on the first run.

## 🛡️ Risk Disclaimer

This software is for institutional research and automated execution. Cryptocurrency futures are highly volatile and carry a significant risk of loss. The developers assume no responsibility for financial losses incurred. Always thoroughly backtest and run in "Shadow Mode" before deploying live capital.
