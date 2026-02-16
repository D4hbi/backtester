from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import Engine
from backtester.indicators import bbands, ema, macd, rolling_percentile, rsi, sma, zscore
from strategies.mean_reversion import BollingerMeanReversion


def _make_close(n: int = 100) -> pd.Series:
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, n))
    prices = np.maximum(prices, 1.0)
    index = pd.bdate_range(start="2023-01-01", periods=n, freq="B")
    return pd.Series(prices, index=index, name="Close")


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


class TestIndicators:
    def test_sma_length(self):
        close = _make_close(50)
        result = sma(close, 10)
        assert len(result) == 50
        assert result.iloc[:9].isna().all()
        assert result.iloc[9:].notna().all()

    def test_sma_value(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(close, 3)
        assert result.iloc[2] == 2.0  # (1+2+3)/3
        assert abs(result.iloc[4] - 4.0) < 1e-10  # (3+4+5)/3

    def test_ema_length(self):
        close = _make_close(50)
        result = ema(close, 10)
        assert len(result) == 50
        assert result.iloc[9:].notna().all()

    def test_rsi_range(self):
        close = _make_close(100)
        result = rsi(close, 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_columns(self):
        close = _make_close(100)
        result = macd(close)
        assert result.shape[1] == 3  # MACD, histogram, signal

    def test_bbands_columns(self):
        close = _make_close(50)
        result = bbands(close, period=20)
        assert result.shape[1] >= 3  # lower, mid, upper (+ bandwidth, %b)

    def test_bbands_ordering(self):
        close = _make_close(50)
        result = bbands(close, period=20).dropna()
        # lower < mid < upper
        assert (result.iloc[:, 0] <= result.iloc[:, 1]).all()
        assert (result.iloc[:, 1] <= result.iloc[:, 2]).all()

    def test_zscore_mean_near_zero(self):
        close = _make_close(200)
        result = zscore(close, 20).dropna()
        assert abs(result.mean()) < 0.5

    def test_rolling_percentile_range(self):
        close = _make_close(100)
        result = rolling_percentile(close, 20).dropna()
        assert (result >= 0).all()
        assert (result <= 1).all()


class TestMeanReversion:
    def test_runs_without_error(self):
        df = _make_ohlcv(200)
        strategy = BollingerMeanReversion("TEST", quantity=100)
        engine = Engine(strategy, df)
        portfolio = engine.run()

        assert len(portfolio.equity_curve) == 200

    def test_no_trades_without_enough_data(self):
        df = _make_ohlcv(15)
        strategy = BollingerMeanReversion("TEST", bb_period=20, quantity=100)
        engine = Engine(strategy, df)
        portfolio = engine.run()

        assert portfolio.trade_count == 0
