from __future__ import annotations

import pandas as pd

from backtester.broker import Broker
from backtester.models import Bar, Order, OrderType, Side, Signal
from backtester.portfolio import Portfolio
from backtester.risk import RiskManager
from backtester.strategy import Strategy


class Engine:
    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        slippage_pct: float = 0.001,
        commission_per_share: float = 0.005,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.strategy = strategy
        self.data = data
        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.broker = Broker(
            slippage_pct=slippage_pct,
            commission_per_share=commission_per_share,
        )
        self.risk_manager = risk_manager
        self._pending_signals: list[Signal] = []

    def run(self) -> Portfolio:
        for date, row in self.data.iterrows():
            bar = Bar(
                date=date,
                open=row["Open"],
                high=row["High"],
                low=row["Low"],
                close=row["Close"],
                volume=int(row["Volume"]),
            )

            self._execute_pending(bar)

            self.strategy._push_bar(bar)
            signal = self.strategy.on_bar(bar)

            if signal is not None:
                if self.risk_manager is not None and signal.side == Side.BUY:
                    signal = self.risk_manager.check(
                        signal, self.portfolio.cash, bar.close
                    )
                if signal is not None:
                    self._pending_signals.append(signal)

            self.portfolio.record_equity(
                date=date,
                prices={self.strategy.ticker: bar.close},
            )

        return self.portfolio

    def _execute_pending(self, bar: Bar) -> None:
        for signal in self._pending_signals:
            order = Order(
                date=bar.date,
                ticker=signal.ticker,
                side=signal.side,
                quantity=signal.quantity,
                order_type=OrderType.MARKET,
            )
            fill = self.broker.execute(order, bar)
            self.portfolio.process_fill(fill)

        self._pending_signals.clear()
