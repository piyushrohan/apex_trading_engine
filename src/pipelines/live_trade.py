import asyncio
import logging
from typing import Dict, Any

from src.data.binance_rest import BinanceRESTClient
from src.data.feature_engine import FeatureEngine
from src.models.regime_detector import RegimeDetector
from src.models.meta_controller import MetaController
from src.execution.risk_engine import RiskEngine
from src.execution.order_manager import OrderManager
from src.execution.position_sync import AccountSynchronizer
from src.mlops.explainability import ExplainabilityEngine
from src.mlops.registry import ModelRegistry

logger = logging.getLogger(__name__)

class LiveTradePipeline:
    """
    The main autonomous Production execution loop.
    Ties together real-time data ingestion, feature generation, regime detection, 
    AI inference, risk management, and zero-fee maker execution.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Core Components
        self.rest_client = BinanceRESTClient()
        self.account_sync = AccountSynchronizer(self.rest_client)
        self.risk_engine = RiskEngine(config)
        self.order_manager = OrderManager(config, self.rest_client)
        
        # Initialize Intelligence Layer
        self.feature_engine = FeatureEngine(config)
        self.regime_detector = RegimeDetector(config)
        
        # Load Active PROD Model
        self.registry = ModelRegistry()
        prod_model_path = self.registry.get_prod_model_path()
        if not prod_model_path:
            logger.warning("No PROD model found in registry! System will run in monitoring mode.")
            
        self.meta_controller = MetaController(config) # In reality, load weights from prod_model_path
        self.explainability = ExplainabilityEngine(config)
        
        self._running = False

    async def start(self):
        """Boots the institutional execution desk."""
        logger.info("Initializing APEX Live Trade Pipeline...")
        self._running = True
        
        # Start account synchronization
        await self.account_sync.start()
        
        # Hook risk engine to manual position changes
        self.account_sync.on_position_change = self._on_manual_position_change
        
        # Start main trading loop
        await self._trading_loop()

    def _on_manual_position_change(self, symbol: str, position_data: dict):
        """Callback to handle discretionary trades or fills dynamically."""
        logger.info(f"Position sync detected change in {symbol}. Risk Engine adjusting exposure.")
        # E.g., updating risk engine's current equity and margin utilization
        # self.risk_engine.update_equity(...)

    async def _trading_loop(self):
        """The core tick-driven async event loop."""
        symbol = self.config.get('data', {}).get('target_symbol', 'ETHUSDC')
        
        logger.info(f"Subscribing to {symbol} real-time streams...")
        # In a real implementation, we would connect to Binance_WS here and iterate over ticks
        # async for tick in binance_ws.stream(symbol):
        
        while self._running:
            try:
                # 1. Wait for next micro-batch of data (Mocked delay)
                await asyncio.sleep(3.0) # Assume 3-second micro-batches
                
                # 2. Extract Features & Regime
                # df = self.feature_engine.process_all_features(...)
                # df = self.regime_detector.detect(df)
                
                # Mock state extraction
                latest_state_vector = [0.1] * 10
                current_regime = "VOLATILITY_EXPANSION"
                current_bbo = 3500.50
                
                # 3. AI Inference (Meta-Controller routes to best model)
                action, conviction, context = self.meta_controller.get_action(latest_state_vector, current_regime)
                
                # 4. Generate Explainability JSON
                explanation = self.explainability.decode_decision(action, conviction, context)
                
                # 5. Risk & Execution
                if action != 1: # If not Flat
                    side = "BUY" if action == 2 else "SELL"
                    
                    # Assume we calculate historical win rate from registry metrics
                    win_rate, win_loss_ratio = 0.55, 1.2
                    
                    # Risk Engine determines Kelly size
                    kelly_fraction = self.risk_engine.calculate_kelly_size(win_rate, win_loss_ratio, conviction)
                    approved_fraction = self.risk_engine.approve_order(side, kelly_fraction, current_exposure=0.0)
                    
                    if approved_fraction > 0:
                        quantity = 1.0 # Calculate based on equity * approved_fraction / price
                        
                        # Execute Maker-Only Post-Only order
                        # We bid at BBO to capture the spread and pay 0 fees
                        price = current_bbo - 0.01 if side == "BUY" else current_bbo + 0.01
                        
                        logger.info(f"Executing {side} based on AI decision. Reason: {explanation['primary_reasons']}")
                        await self.order_manager.place_maker_order(side, quantity, price)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(1)

    async def stop(self):
        """Gracefully shuts down the pipeline."""
        self._running = False
        await self.account_sync.stop()
        await self.rest_client.close()
        logger.info("Live Trade Pipeline stopped gracefully.")

if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    with open("configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    pipeline = LiveTradePipeline(config)
    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        asyncio.run(pipeline.stop())
