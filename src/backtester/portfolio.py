from __future__ import annotations

from datetime import datetime

from backtester.models import Fill, Position, Side


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.equity_curve: list[tuple[datetime, float]] = []

    def process_fill(self, fill: Fill) -> None:
        self.fills.append(fill)

        if fill.ticker not in self.positions:
            self.positions[fill.ticker] = Position(ticker=fill.ticker)

        cash_delta = self.positions[fill.ticker].update(fill)
        self.cash += cash_delta

    def record_equity(self, date: datetime, prices: dict[str, float]) -> None:
        holdings_value = 0.0
        for ticker, position in self.positions.items():
            if position.quantity > 0 and ticker in prices:
                holdings_value += position.quantity * prices[ticker]

        total_equity = self.cash + holdings_value
        self.equity_curve.append((date, total_equity))

    @property
    def total_return(self) -> float:
        if not self.equity_curve:
            return 0.0
        final = self.equity_curve[-1][1]
        return (final - self.initial_cash) / self.initial_cash

    @property
    def trade_count(self) -> int:
        return len(self.fills)
