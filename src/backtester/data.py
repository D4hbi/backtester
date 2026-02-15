from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


class DataFeed:
    """Fetches, caches, and validates OHLCV equity data."""

    def __init__(self, cache_dir: str | Path | None = _DEFAULT_CACHE_DIR) -> None:
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def get(
        self,
        ticker: str,
        start: str,
        end: str,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Return validated OHLCV DataFrame for a ticker between start and end dates."""
        ticker = ticker.upper().strip()

        if use_cache and self.cache_dir is not None:
            cached = self._load_cache(ticker, start, end)
            if cached is not None:
                return cached

        df = self._download(ticker, start, end)
        df = self._validate(df, ticker)

        if use_cache and self.cache_dir is not None:
            self._save_cache(df, ticker, start, end)

        return df

    def _download(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance."""
        print(f"[DataFeed] Downloading {ticker} from {start} to {end} ...")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            raise ValueError(
                f"No data returned for {ticker} ({start} -> {end}). "
                "Check the ticker symbol and date range."
            )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        keep = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in keep if c in df.columns]]

        return df

    def _validate(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and validate the OHLCV DataFrame."""
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        nan_before = int(df.isna().sum().sum())
        df = df.ffill(limit=5)
        nan_after = int(df.isna().sum().sum())
        if nan_before > 0:
            filled = nan_before - nan_after
            print(f"[DataFeed] {ticker}: forward-filled {filled} NaN values")

        if df.isna().any().any():
            dropped = int(df.isna().any(axis=1).sum())
            print(f"[DataFeed] {ticker}: dropping {dropped} rows with remaining NaNs")
            df = df.dropna()

        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in df.columns:
                bad = (df[col] <= 0).sum()
                if bad > 0:
                    print(f"[DataFeed] WARNING: {ticker} has {bad} non-positive {col} values")
                    df = df[df[col] > 0]

        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(np.int64)

        return df

    def _cache_path(self, ticker: str, start: str, end: str) -> Path:
        return self.cache_dir / f"{ticker}_{start}_{end}.parquet"

    def _save_cache(self, df: pd.DataFrame, ticker: str, start: str, end: str) -> None:
        path = self._cache_path(ticker, start, end)
        df.to_parquet(path, engine="pyarrow")
        print(f"[DataFeed] Cached -> {path}")

    def _load_cache(self, ticker: str, start: str, end: str) -> pd.DataFrame | None:
        path = self._cache_path(ticker, start, end)
        if not path.exists():
            return None
        print(f"[DataFeed] Loading from cache -> {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df

    def clear_cache(self) -> int:
        """Delete all cached Parquet files. Returns number of files deleted."""
        if self.cache_dir is None:
            return 0
        files = list(self.cache_dir.glob("*.parquet"))
        for f in files:
            f.unlink()
        return len(files)
