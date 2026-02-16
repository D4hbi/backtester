from __future__ import annotations

import pandas as pd
import pandas_ta as ta
from scipy import stats


def sma(close: pd.Series, period: int) -> pd.Series:
    return ta.sma(close, length=period)


def ema(close: pd.Series, period: int) -> pd.Series:
    return ta.ema(close, length=period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return ta.rsi(close, length=period)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    return ta.macd(close, fast=fast, slow=slow, signal=signal)


def bbands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    return ta.bbands(close, length=period, std=std_dev)


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    rolling_mean = series.rolling(period).mean()
    rolling_std = series.rolling(period).std()
    return (series - rolling_mean) / rolling_std


def rolling_percentile(series: pd.Series, period: int = 20) -> pd.Series:
    def _pct(window):
        return stats.percentileofscore(window[:-1], window.iloc[-1]) / 100.0

    return series.rolling(period).apply(_pct, raw=False)
