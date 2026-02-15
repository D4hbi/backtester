from __future__ import annotations

from backtester.models import Bar, Signal, Side
from backtester.strategy import Strategy


class SmaCrossover(Strategy):
    def __init__(
        self,
        ticker: str,
        fast_period: int = 10,
        slow_period: int = 50,
        quantity: int = 100,
    ) -> None:
        super().__init__(ticker)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.quantity = quantity
        self._in_position = False

    def on_bar(self, bar: Bar) -> Signal | None:
        if len(self._bars) < self.slow_period:
            return None

        closes = self.closes()
        fast_sma = closes.iloc[-self.fast_period:].mean()
        slow_sma = closes.iloc[-self.slow_period:].mean()

        prev_closes = closes.iloc[:-1]
        if len(prev_closes) < self.slow_period:
            return None
        prev_fast = prev_closes.iloc[-self.fast_period:].mean()
        prev_slow = prev_closes.iloc[-self.slow_period:].mean()

        if prev_fast <= prev_slow and fast_sma > slow_sma and not self._in_position:
            self._in_position = True
            return Signal(
                date=bar.date,
                ticker=self.ticker,
                side=Side.BUY,
                quantity=self.quantity,
            )

        if prev_fast >= prev_slow and fast_sma < slow_sma and self._in_position:
            self._in_position = False
            return Signal(
                date=bar.date,
                ticker=self.ticker,
                side=Side.SELL,
                quantity=self.quantity,
            )

        return None
