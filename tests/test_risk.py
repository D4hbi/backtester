from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import Engine
from backtester.models import Signal, Side
from backtester.risk import RiskManager
from backtester.sizing import fixed_fractional, volatility_targeted
from strategies.sma_cross import SmaCrossover


def _make_ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start="2022-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)

    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.3, n),
            "High": close + np.abs(rng.normal(0, 1, n)),
            "Low": close - np.abs(rng.normal(0, 1, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 50_000_000, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


class TestFixedFractional:
    def test_basic(self):
        shares = fixed_fractional(equity=100_000, price=50.0, risk_pct=0.02)
        # 100k * 0.02 = $2000, at $50/share = 40 shares
        assert shares == 40

    def test_zero_price(self):
        shares = fixed_fractional(equity=100_000, price=0.0, risk_pct=0.02)
        assert shares == 0

    def test_expensive_stock(self):
        shares = fixed_fractional(equity=10_000, price=5000.0, risk_pct=0.02)
        # 10k * 0.02 = $200, can't buy a single share at $5000
        assert shares == 0


class TestVolatilityTargeted:
    def test_returns_int(self):
        closes = pd.Series(100.0 + np.cumsum(np.random.default_rng(42).normal(0, 1, 50)))
        shares = volatility_targeted(equity=100_000, price=100.0, closes=closes)
        assert isinstance(shares, int)
        assert shares >= 0

    def test_not_enough_data(self):
        closes = pd.Series([100.0, 101.0, 102.0])
        shares = volatility_targeted(equity=100_000, price=100.0, closes=closes, lookback=20)
        assert shares == 0


class TestRiskManager:
    def test_passes_normal_signal(self):
        rm = RiskManager(max_position_pct=0.5, max_drawdown_pct=0.20)
        signal = Signal(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.BUY,
            quantity=100,
        )
        result = rm.check(signal, equity=100_000, price=150.0)
        assert result is not None
        assert result.quantity == 100

    def test_caps_oversized_position(self):
        rm = RiskManager(max_position_pct=0.1)
        signal = Signal(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.BUY,
            quantity=1000,
        )
        # 1000 * $150 = $150k, but max is 10% of $100k = $10k
        result = rm.check(signal, equity=100_000, price=150.0)
        assert result is not None
        assert result.quantity == 66  # int(10000 // 150)

    def test_blocks_after_drawdown(self):
        rm = RiskManager(max_drawdown_pct=0.10)
        signal = Signal(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.BUY,
            quantity=100,
        )
        # Set peak equity high, then check with much lower equity
        rm.check(signal, equity=100_000, price=150.0)
        result = rm.check(signal, equity=85_000, price=150.0)  # 15% drawdown
        assert result is None


class TestEngineWithRisk:
    def test_runs_with_risk_manager(self):
        df = _make_ohlcv(200)
        strategy = SmaCrossover("TEST", fast_period=10, slow_period=50, quantity=100)
        rm = RiskManager(max_position_pct=0.3, max_drawdown_pct=0.15)
        engine = Engine(strategy, df, initial_cash=100_000, risk_manager=rm)
        portfolio = engine.run()

        assert len(portfolio.equity_curve) == 200

    def test_runs_without_risk_manager(self):
        df = _make_ohlcv(200)
        strategy = SmaCrossover("TEST", fast_period=10, slow_period=50, quantity=100)
        engine = Engine(strategy, df, initial_cash=100_000)
        portfolio = engine.run()

        assert len(portfolio.equity_curve) == 200
