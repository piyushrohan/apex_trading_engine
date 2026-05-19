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


@pytest.mark.unit
def test_portfolio_book_notional_leverage_and_hedge_ratio_edges():
    book = PortfolioBook(
        book_id="shadow",
        role="shadow",
        model_id="m2",
        symbol="ETHUSDC",
        initial_equity=1000.0,
        equity=0.0,
        long_qty=1.0,
        short_qty=0.25,
        entry_price_long=3000.0,
        entry_price_short=3200.0,
    )

    assert book.net_qty == 0.75
    assert book.gross_notional(3500.0) == 4375.0
    assert book.net_leverage(3500.0) == 0.0
    assert book.gross_leverage(3500.0) == 0.0
    assert book.hedge_ratio() == 0.25
    assert book.primary_leg_qty() == 1.0

    book.short_qty = 0.0
    assert book.hedge_ratio() == 0.0


@pytest.mark.unit
def test_portfolio_book_one_way_adds_reduces_and_marks_equity():
    book = PortfolioBook(
        book_id="primary",
        role="primary",
        model_id="m1",
        symbol="ETHUSDC",
        initial_equity=1000.0,
        equity=1000.0,
    )

    book.apply_fill("BUY", 2.0, 100.0)
    book.apply_fill("SELL", 1.0, 110.0)
    assert book.long_qty == 1.0
    assert book.equity == 1010.0
    assert book.high_water_mark == 1010.0

    book.apply_fill("SELL", 1.0, 120.0)
    assert book.long_qty == 0.0
    assert book.entry_price_long == 0.0

    book.apply_fill("SELL", 2.0, 100.0)
    book.apply_fill("BUY", 1.0, 90.0)
    assert book.short_qty == 1.0
    assert book.equity == 1010.0

    book.apply_fill("BUY", 1.0, 80.0)
    assert book.short_qty == 0.0
    assert book.entry_price_short == 0.0


@pytest.mark.unit
def test_portfolio_book_hedge_reductions_sync_and_flatten():
    book = PortfolioBook(
        book_id="primary",
        role="primary",
        model_id="m1",
        symbol="ETHUSDC",
        initial_equity=1000.0,
        equity=1000.0,
        position_mode="hedge",
    )

    book.apply_fill("BUY", 2.0, 100.0, position_side="LONG")
    book.apply_fill("SELL", 0.5, 110.0, position_side="LONG")
    book.apply_fill("SELL", 1.0, 120.0, position_side="SHORT")
    book.apply_fill("BUY", 0.25, 115.0, position_side="SHORT")

    assert book.long_qty == 1.5
    assert book.short_qty == 0.75

    book.sync_legs(
        long_qty=-1.0,
        short_qty=0.5,
        entry_long=100.0,
        entry_short=125.0,
        mark_price=120.0,
    )
    assert book.long_qty == 0.0
    assert book.entry_price_long == 0.0
    assert book.short_qty == 0.5
    assert book.entry_price_short == 125.0

    book.flatten_all(mark_price=120.0)
    assert book.long_qty == 0.0
    assert book.short_qty == 0.0
    assert book.entry_price_short == 0.0


@pytest.mark.unit
def test_portfolio_service_exposure_edges():
    svc = PortfolioService(position_mode="hedge")

    assert svc.primary_book() is None
    assert svc.current_exposure_fraction("missing", 3500.0) == 0.0
    assert svc.gross_exposure_fraction("missing", 3500.0) == 0.0

    book = svc.get_or_create_book("primary", "primary", "m1", "ETHUSDC", 1000.0)
    book.sync_legs(1.0, 0.25, entry_long=3000.0, entry_short=3200.0)

    assert svc.current_exposure_fraction("primary", 100.0) == 0.075
    assert svc.gross_exposure_fraction("primary", 100.0) == 0.125

    book.equity = 0.0
    assert svc.current_exposure_fraction("primary", 100.0) == 0.0
    assert svc.gross_exposure_fraction("primary", 100.0) == 0.0
