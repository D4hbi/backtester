from __future__ import annotations

from backtester.indicators import bbands, zscore
from backtester.models import Bar, Signal, Side
from backtester.strategy import Strategy


class BollingerMeanReversion(Strategy):
    def __init__(
        self,
        ticker: str,
        bb_period: int = 20,
        bb_std: float = 2.0,
        zscore_threshold: float = -2.0,
        quantity: int = 100,
    ) -> None:
        super().__init__(ticker)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.zscore_threshold = zscore_threshold
        self.quantity = quantity
        self._in_position = False

    def on_bar(self, bar: Bar) -> Signal | None:
        if len(self._bars) < self.bb_period + 1:
            return None

        closes = self.closes()
        bands = bbands(closes, period=self.bb_period, std_dev=self.bb_std)
        z = zscore(closes, period=self.bb_period)

        lower_band = bands.iloc[-1, 0]  # BBL
        upper_band = bands.iloc[-1, 2]  # BBU
        current_z = z.iloc[-1]
        price = bar.close

        # Buy when price drops below lower band and z-score confirms oversold
        if price <= lower_band and current_z <= self.zscore_threshold and not self._in_position:
            self._in_position = True
            return Signal(
                date=bar.date,
                ticker=self.ticker,
                side=Side.BUY,
                quantity=self.quantity,
            )

        # Sell when price reaches upper band (mean reverted past the mean)
        if price >= upper_band and self._in_position:
            self._in_position = False
            return Signal(
                date=bar.date,
                ticker=self.ticker,
                side=Side.SELL,
                quantity=self.quantity,
            )

        return None
