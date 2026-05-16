import pandas as pd
import logging
from typing import Dict, Any, List, Tuple

from src.models.meta_controller import MetaController

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Deterministic Historical Simulation Engine.
    Used during MLOps pipeline evaluation to simulate out-of-sample 
    performance of candidate models on DuckDB historical data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_controller = MetaController(config)
        self.initial_capital = config.get('environment', {}).get('initial_capital', 1000.0)
        self.transaction_fee = config.get('environment', {}).get('transaction_cost_pct', 0.0) # 0 for Maker ETHUSDC
        
    def run(self, df_features: pd.DataFrame) -> Tuple[pd.Series, List[dict]]:
        """
        Runs the simulation sequentially over the historical dataframe.
        Assumes df_features contains 'close', 'regime_str', and all 10 state vector columns.
        Returns the PnL equity curve and a list of trade logs.
        """
        logger.info(f"Starting deterministic backtest over {len(df_features)} bars.")
        
        equity = self.initial_capital
        equity_curve = []
        trades = []
        
        position = 0 # 1 for Long, -1 for Short
        entry_price = 0.0
        
        # Simplified vector extraction logic
        feature_cols = [col for col in df_features.columns if col not in ['timestamp', 'close', 'regime_str', 'regime_id']]
        
        for idx, row in df_features.iterrows():
            current_price = row['close']
            regime = row.get('regime_str', 'MEAN_REVERSION')
            
            # State vector array
            state_vector = row[feature_cols].values.tolist()
            
            action, conviction, _ = self.meta_controller.get_action(state_vector, regime)
            
            # Action logic: 0=Short, 1=Flat, 2=Long
            target_position = 0
            if action == 0:
                target_position = -1
            elif action == 2:
                target_position = 1
                
            # If position changed, log trade
            if target_position != position:
                if position != 0:
                    # Close existing position
                    pnl = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
                    pnl_value = equity * pnl
                    equity += pnl_value
                    
                    trades.append({
                        "exit_time": row.name if hasattr(row, 'name') else idx,
                        "pnl": pnl_value,
                        "pnl_pct": pnl,
                        "side": "LONG" if position == 1 else "SHORT"
                    })
                    
                position = target_position
                if position != 0:
                    entry_price = current_price
                    
            equity_curve.append(equity)
            
        logger.info(f"Backtest complete. Final Equity: {equity:.2f}, Total Trades: {len(trades)}")
        return pd.Series(equity_curve), trades
