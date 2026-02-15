from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from backtester.models import Bar, Signal


class Strategy(ABC):
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self._bars: list[Bar] = []

    @property
    def bars(self) -> list[Bar]:
        return self._bars

    def _push_bar(self, bar: Bar) -> None:
        self._bars.append(bar)

    @abstractmethod
    def on_bar(self, bar: Bar) -> Signal | None:
        ...

    def closes(self) -> pd.Series:
        return pd.Series(
            [b.close for b in self._bars],
            index=[b.date for b in self._bars],
        )
