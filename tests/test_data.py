from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.data import DataFeed


def _make_ohlcv(
    n_days: int = 100,
    start: str = "2023-01-01",
    base_price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(start=start, periods=n_days, freq="B")
    close = base_price + np.cumsum(rng.normal(0, 1, n_days))
    close = np.maximum(close, 1.0)

    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.5, n_days),
            "High": close + np.abs(rng.normal(0, 1, n_days)),
            "Low": close - np.abs(rng.normal(0, 1, n_days)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 50_000_000, n_days),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


class TestValidation:
    def setup_method(self):
        self.feed = DataFeed(cache_dir=None)

    def test_sorts_by_date(self):
        df = _make_ohlcv(50)
        shuffled = df.sample(frac=1)
        result = self.feed._validate(shuffled, "TEST")
        assert result.index.is_monotonic_increasing

    def test_removes_duplicate_dates(self):
        df = _make_ohlcv(50)
        duped = pd.concat([df, df.iloc[:5]])
        result = self.feed._validate(duped, "TEST")
        assert not result.index.duplicated().any()
        assert len(result) == 50

    def test_forward_fills_small_nan_gaps(self):
        df = _make_ohlcv(50)
        df.iloc[10:13, df.columns.get_loc("Close")] = np.nan
        result = self.feed._validate(df, "TEST")
        assert not result["Close"].isna().any()

    def test_drops_rows_with_large_nan_gaps(self):
        df = _make_ohlcv(50)
        df.iloc[10:20, df.columns.get_loc("Close")] = np.nan
        result = self.feed._validate(df, "TEST")
        assert not result["Close"].isna().any()
        assert len(result) < 50

    def test_removes_non_positive_prices(self):
        df = _make_ohlcv(50)
        df.iloc[5, df.columns.get_loc("Close")] = -1.0
        result = self.feed._validate(df, "TEST")
        assert (result["Close"] > 0).all()

    def test_volume_is_int64(self):
        df = _make_ohlcv(50)
        df["Volume"] = df["Volume"].astype(float)
        result = self.feed._validate(df, "TEST")
        assert result["Volume"].dtype == np.int64


class TestCache:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feed = DataFeed(cache_dir=tmpdir)
            df = _make_ohlcv(30)

            feed._save_cache(df, "TEST", "2023-01-01", "2023-03-01")
            loaded = feed._load_cache("TEST", "2023-01-01", "2023-03-01")

            assert loaded is not None
            pd.testing.assert_frame_equal(df, loaded, check_freq=False)

    def test_cache_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feed = DataFeed(cache_dir=tmpdir)
            result = feed._load_cache("NOPE", "2023-01-01", "2023-12-01")
            assert result is None

    def test_clear_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feed = DataFeed(cache_dir=tmpdir)
            df = _make_ohlcv(10)
            feed._save_cache(df, "A", "2023-01-01", "2023-02-01")
            feed._save_cache(df, "B", "2023-01-01", "2023-02-01")
            deleted = feed.clear_cache()
            assert deleted == 2
            assert list(Path(tmpdir).glob("*.parquet")) == []

    def test_no_cache_dir(self):
        feed = DataFeed(cache_dir=None)
        assert feed.cache_dir is None
        assert feed.clear_cache() == 0


@pytest.mark.slow
class TestDownload:
    def test_download_aapl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feed = DataFeed(cache_dir=tmpdir)
            df = feed.get("AAPL", start="2023-01-01", end="2023-02-01")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
            assert df.index.name == "Date"
            assert (df["Close"] > 0).all()
