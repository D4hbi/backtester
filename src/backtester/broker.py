from __future__ import annotations

from backtester.models import Bar, Order, Fill, Side


class Broker:
    def __init__(
        self,
        slippage_pct: float = 0.001,
        commission_per_share: float = 0.005,
    ) -> None:
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share

    def execute(self, order: Order, bar: Bar) -> Fill:
        base_price = bar.open

        if order.side == Side.BUY:
            fill_price = base_price * (1 + self.slippage_pct)
        else:
            fill_price = base_price * (1 - self.slippage_pct)

        commission = self.commission_per_share * order.quantity

        return Fill(
            date=order.date,
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            fill_price=round(fill_price, 4),
            commission=round(commission, 4),
        )
