from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import Engine
from backtester import analytics
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


def _run_backtest(n: int = 200) -> analytics.Portfolio:
    df = _make_ohlcv(n)
    strategy = SmaCrossover("TEST", fast_period=10, slow_period=50, quantity=100)
    engine = Engine(strategy, df, initial_cash=100_000)
    return engine.run()


class TestEquitySeries:
    def test_length(self):
        portfolio = _run_backtest(200)
        eq = analytics.equity_series(portfolio)
        assert len(eq) == 200

    def test_starts_at_initial_cash(self):
        portfolio = _run_backtest(200)
        eq = analytics.equity_series(portfolio)
        assert eq.iloc[0] == 100_000.0


class TestReturns:
    def test_daily_returns_length(self):
        portfolio = _run_backtest(200)
        returns = analytics.daily_returns(portfolio)
        assert len(returns) == 199  # one less due to pct_change


class TestSharpe:
    def test_returns_float(self):
        portfolio = _run_backtest(200)
        result = analytics.sharpe_ratio(portfolio)
        assert isinstance(result, float)

    def test_zero_with_no_variance(self):
        portfolio = analytics.Portfolio(initial_cash=100_000)
        for i in range(50):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i)
            portfolio.equity_curve.append((date, 100_000.0))
        assert analytics.sharpe_ratio(portfolio) == 0.0


class TestSortino:
    def test_returns_float(self):
        portfolio = _run_backtest(200)
        result = analytics.sortino_ratio(portfolio)
        assert isinstance(result, float)


class TestDrawdown:
    def test_max_drawdown_is_negative(self):
        portfolio = _run_backtest(200)
        dd = analytics.max_drawdown(portfolio)
        assert dd <= 0.0

    def test_drawdown_series_length(self):
        portfolio = _run_backtest(200)
        dd = analytics.drawdown_series(portfolio)
        assert len(dd) == 200

    def test_drawdown_series_starts_at_zero(self):
        portfolio = _run_backtest(200)
        dd = analytics.drawdown_series(portfolio)
        assert dd.iloc[0] == 0.0


class TestCalmar:
    def test_returns_float(self):
        portfolio = _run_backtest(200)
        result = analytics.calmar_ratio(portfolio)
        assert isinstance(result, float)


class TestWinRate:
    def test_between_zero_and_one(self):
        portfolio = _run_backtest(200)
        wr = analytics.win_rate(portfolio)
        assert 0.0 <= wr <= 1.0

    def test_zero_with_no_trades(self):
        portfolio = analytics.Portfolio(initial_cash=100_000)
        assert analytics.win_rate(portfolio) == 0.0


class TestProfitFactor:
    def test_returns_float(self):
        portfolio = _run_backtest(200)
        result = analytics.profit_factor(portfolio)
        assert isinstance(result, float)

    def test_zero_with_no_trades(self):
        portfolio = analytics.Portfolio(initial_cash=100_000)
        assert analytics.profit_factor(portfolio) == 0.0


class TestStatsmodels:
    def test_adf_test(self):
        portfolio = _run_backtest(200)
        result = analytics.adf_test(portfolio)
        assert "statistic" in result
        assert "p_value" in result
        assert "is_stationary" in result
        assert isinstance(result["is_stationary"], bool)

    def test_autocorrelation(self):
        portfolio = _run_backtest(200)
        result = analytics.autocorrelation(portfolio, nlags=5)
        assert len(result) == 6  # lag 0 through 5
        assert result.iloc[0] == 1.0  # lag 0 is always 1


class TestSummary:
    def test_has_all_keys(self):
        portfolio = _run_backtest(200)
        result = analytics.summary(portfolio)
        expected_keys = [
            "total_return", "annual_return", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
            "trade_count", "start_date", "end_date",
        ]
        for key in expected_keys:
            assert key in result
