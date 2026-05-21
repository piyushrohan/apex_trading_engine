from src.data.binance_rest import BinanceRESTClient
from src.execution.adapters.base import ExecutionAdapter
from src.execution.adapters.live import LiveExecutionAdapter
from src.execution.adapters.paper import PaperExecutionAdapter
from src.execution.order_lifecycle import OrderLifecycleRecorder


def get_operator_mode(config: dict) -> str:
    return config.get("execution", {}).get("operator_mode", "paper")


def create_execution_adapter(
    config: dict,
    rest_client: BinanceRESTClient,
    book_id: str = "primary",
    lifecycle_recorder: OrderLifecycleRecorder | None = None,
) -> ExecutionAdapter:
    """
    Primary book adapter: paper operator mode → virtual sim;
    live operator mode → signed REST.
    Shadow lanes always use PaperExecutionAdapter (see TradingPipeline).
    """
    if get_operator_mode(config) == "live":
        return LiveExecutionAdapter(config, rest_client, lifecycle_recorder)
    return PaperExecutionAdapter(book_id=book_id, lifecycle_recorder=lifecycle_recorder)
