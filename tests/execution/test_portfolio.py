import pytest

from src.execution.portfolio import PortfolioBook, PortfolioService


@pytest.mark.unit
def test_portfolio_hedge_mode_legs():
    book = PortfolioBook(
        book_id="primary",
        role="primary",
        model_id="test",
        symbol="ETHUSDC",
        initial_equity=1000.0,
        equity=1000.0,
        position_mode="hedge",
    )
    book.apply_fill("BUY", 1.0, 3500.0, position_side="LONG")
    book.apply_fill("SELL", 0.5, 3500.0, position_side="SHORT")
    assert book.long_qty == 1.0
    assert book.short_qty == 0.5
    assert book.gross_qty == 1.5


@pytest.mark.unit
def test_portfolio_service_primary_book():
    svc = PortfolioService(position_mode="hedge")
    book = svc.get_or_create_book("primary", "primary", "m1", "ETHUSDC", 1000.0)
    assert svc.primary_book() is book
    assert svc.current_exposure_fraction("primary", 3500.0) == 0.0
