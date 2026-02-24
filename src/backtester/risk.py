from __future__ import annotations

from backtester.models import Signal


class RiskManager:
    """Enforces risk limits before orders are placed."""

    def __init__(
        self,
        max_position_pct: float = 0.5,
        max_drawdown_pct: float = 0.20,
    ) -> None:
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self._peak_equity = 0.0
        self._stopped_out = False

    def check(self, signal: Signal, equity: float, price: float) -> Signal | None:
        """Returns the signal if it passes risk checks, or None if blocked."""
        self._peak_equity = max(self._peak_equity, equity)

        # Check drawdown stop
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.max_drawdown_pct:
                if not self._stopped_out:
                    print(f"[Risk] Drawdown limit hit ({drawdown:.1%}). Blocking new buys.")
                    self._stopped_out = True
                return None

        if self._stopped_out and equity >= self._peak_equity * 0.95:
            self._stopped_out = False

        # Check max position size
        position_value = signal.quantity * price
        if position_value > equity * self.max_position_pct:
            max_shares = int((equity * self.max_position_pct) // price)
            if max_shares <= 0:
                return None
            return Signal(
                date=signal.date,
                ticker=signal.ticker,
                side=signal.side,
                quantity=max_shares,
            )

        return signal
