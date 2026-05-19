"""Operator paper mode entrypoint (primary virtual book)."""

import asyncio
import logging

from src.core.config_loader import load_config
from src.pipelines.trading_pipeline import TradingPipeline

logger = logging.getLogger(__name__)


def build_pipeline(config_path: str = "configs/base.yaml") -> TradingPipeline:
    config = load_config(config_path)
    config.setdefault("execution", {})["operator_mode"] = "paper"
    return TradingPipeline(config)


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
