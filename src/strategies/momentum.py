from __future__ import annotations

from backtester.indicators import rsi, macd
from backtester.models import Bar, Signal, Side
from backtester.strategy import Strategy


class RsiMacdMomentum(Strategy):
    def __init__(
        self,
        ticker: str,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        quantity: int = 100,
    ) -> None:
        super().__init__(ticker)
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.quantity = quantity
        self._in_position = False

    def on_bar(self, bar: Bar) -> Signal | None:
        min_bars = self.macd_slow + self.macd_signal + 1
        if len(self._bars) < min_bars:
            return None

        closes = self.closes()
        current_rsi = rsi(closes, self.rsi_period).iloc[-1]
        macd_df = macd(closes, self.macd_fast, self.macd_slow, self.macd_signal)
        macd_line = macd_df.iloc[-1, 0]
        signal_line = macd_df.iloc[-1, 2]
        prev_macd = macd_df.iloc[-2, 0]
        prev_signal = macd_df.iloc[-2, 2]

        # Buy: RSI was oversold and MACD crosses above signal line
        macd_cross_up = prev_macd <= prev_signal and macd_line > signal_line
        if macd_cross_up and current_rsi < 50 and not self._in_position:
            self._in_position = True
            return Signal(
                date=bar.date,
                ticker=self.ticker,
                side=Side.BUY,
                quantity=self.quantity,
            )

        # Sell: RSI overbought or MACD crosses below signal line
        macd_cross_down = prev_macd >= prev_signal and macd_line < signal_line
        if (macd_cross_down or current_rsi > self.rsi_overbought) and self._in_position:
            self._in_position = False
            return Signal(
                date=bar.date,
                ticker=self.ticker,
                side=Side.SELL,
                quantity=self.quantity,
            )

        return None
