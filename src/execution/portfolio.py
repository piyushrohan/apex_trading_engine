import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PortfolioBook:
    """One virtual or live logical book (primary or shadow)."""

    book_id: str
    role: str
    model_id: str
    symbol: str
    initial_equity: float
    equity: float
    long_qty: float = 0.0
    short_qty: float = 0.0
    entry_price_long: float = 0.0
    entry_price_short: float = 0.0
    high_water_mark: float = 0.0
    position_mode: str = "one_way"

    def __post_init__(self):
        if self.high_water_mark <= 0:
            self.high_water_mark = self.initial_equity

    @property
    def net_qty(self) -> float:
        return self.long_qty - self.short_qty

    @property
    def gross_qty(self) -> float:
        return self.long_qty + self.short_qty

    def net_notional(self, mark_price: float) -> float:
        return abs(self.net_qty) * mark_price

    def gross_notional(self, mark_price: float) -> float:
        return self.gross_qty * mark_price

    def net_leverage(self, mark_price: float) -> float:
        if self.equity <= 0:
            return 0.0
        return self.net_notional(mark_price) / self.equity

    def gross_leverage(self, mark_price: float) -> float:
        if self.equity <= 0:
            return 0.0
        return self.gross_notional(mark_price) / self.equity

    def hedge_ratio(self) -> float:
        """min(long, short) / max(long, short) by quantity; 0 if flat."""
        if self.long_qty <= 0 or self.short_qty <= 0:
            return 0.0
        return min(self.long_qty, self.short_qty) / max(self.long_qty, self.short_qty)

    def primary_leg_qty(self) -> float:
        """Dominant directional leg size (for hedge-ratio vs primary)."""
        return max(self.long_qty, self.short_qty)

    def update_equity_mark(self, mark_price: float):
        unrealized = 0.0
        if self.long_qty > 0 and self.entry_price_long > 0:
            unrealized += (mark_price - self.entry_price_long) * self.long_qty
        if self.short_qty > 0 and self.entry_price_short > 0:
            unrealized += (self.entry_price_short - mark_price) * self.short_qty
        self.equity = self.initial_equity + unrealized
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity

    def apply_fill(
        self,
        side: str,
        qty: float,
        price: float,
        position_side: Optional[str] = None,
    ):
        ps = (position_side or "BOTH").upper()
        if self.position_mode == "hedge" and ps in ("LONG", "SHORT"):
            if ps == "LONG":
                if side == "BUY":
                    self._add_long(qty, price)
                else:
                    self._reduce_long(qty, price)
            else:
                if side == "SELL":
                    self._add_short(qty, price)
                else:
                    self._reduce_short(qty, price)
        else:
            if side == "BUY":
                if self.net_qty < 0:
                    self._reduce_short(qty, price)
                else:
                    self._add_long(qty, price)
            else:
                if self.net_qty > 0:
                    self._reduce_long(qty, price)
                else:
                    self._add_short(qty, price)
        self.update_equity_mark(price)

    def sync_legs(
        self,
        long_qty: float,
        short_qty: float,
        entry_long: float = 0.0,
        entry_short: float = 0.0,
        mark_price: Optional[float] = None,
    ):
        """Overwrite leg quantities from exchange account sync."""
        self.long_qty = max(float(long_qty), 0.0)
        self.short_qty = max(float(short_qty), 0.0)
        self.entry_price_long = entry_long if self.long_qty > 0 else 0.0
        self.entry_price_short = entry_short if self.short_qty > 0 else 0.0
        if mark_price is not None:
            self.update_equity_mark(mark_price)

    def _add_long(self, qty: float, price: float):
        total = self.long_qty + qty
        if total > 0:
            self.entry_price_long = (
                (self.entry_price_long * self.long_qty) + price * qty
            ) / total
        self.long_qty = total

    def _reduce_long(self, qty: float, price: float):
        self.long_qty = max(0.0, self.long_qty - qty)
        if self.long_qty == 0:
            self.entry_price_long = 0.0

    def _add_short(self, qty: float, price: float):
        total = self.short_qty + qty
        if total > 0:
            self.entry_price_short = (
                (self.entry_price_short * self.short_qty) + price * qty
            ) / total
        self.short_qty = total

    def _reduce_short(self, qty: float, price: float):
        self.short_qty = max(0.0, self.short_qty - qty)
        if self.short_qty == 0:
            self.entry_price_short = 0.0

    def flatten_all(self, mark_price: float):
        self.long_qty = 0.0
        self.short_qty = 0.0
        self.entry_price_long = 0.0
        self.entry_price_short = 0.0
        self.update_equity_mark(mark_price)


@dataclass
class PortfolioService:
    """Manages multiple books (primary operator + shadow MLOps lanes)."""

    books: Dict[str, PortfolioBook] = field(default_factory=dict)
    position_mode: str = "one_way"

    def get_or_create_book(
        self,
        book_id: str,
        role: str,
        model_id: str,
        symbol: str,
        initial_equity: float,
    ) -> PortfolioBook:
        if book_id not in self.books:
            self.books[book_id] = PortfolioBook(
                book_id=book_id,
                role=role,
                model_id=model_id,
                symbol=symbol,
                initial_equity=initial_equity,
                equity=initial_equity,
                position_mode=self.position_mode,
            )
        return self.books[book_id]

    def primary_book(self) -> Optional[PortfolioBook]:
        for book in self.books.values():
            if book.role == "primary":
                return book
        return None

    def current_exposure_fraction(self, book_id: str, mark_price: float) -> float:
        book = self.books.get(book_id)
        if not book or book.equity <= 0:
            return 0.0
        if book.position_mode == "hedge":
            return book.net_leverage(mark_price)
        return book.net_notional(mark_price) / book.equity

    def gross_exposure_fraction(self, book_id: str, mark_price: float) -> float:
        book = self.books.get(book_id)
        if not book or book.equity <= 0:
            return 0.0
        return book.gross_leverage(mark_price)
