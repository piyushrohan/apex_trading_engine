"""Operator live mode entrypoint (primary real orders)."""

import asyncio
import logging

from src.core.config_loader import load_config
from src.pipelines.trading_pipeline import TradingPipeline

logger = logging.getLogger(__name__)


class LiveTradePipeline(TradingPipeline):
    """Backward-compatible alias for live operator mode."""

    def __init__(self, config):
        config = dict(config)
        config.setdefault("execution", {})["operator_mode"] = "live"
        super().__init__(config)


def build_pipeline(config_path: str = "configs/base.yaml") -> LiveTradePipeline:
    config = load_config(config_path)
    return LiveTradePipeline(config)


async def main(config_path: str = "configs/base.yaml"):
    pipeline = build_pipeline(config_path)
    try:
        await pipeline.start()
    except KeyboardInterrupt:
        await pipeline.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())
