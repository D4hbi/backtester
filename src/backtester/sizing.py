from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_fractional(
    equity: float,
    price: float,
    risk_pct: float = 0.02,
) -> int:
    """Risk a fixed percentage of equity per trade.

    For example, with $100k equity and 2% risk, you'd allocate
    $2,000 worth of shares.
    """
    if price <= 0:
        return 0
    dollar_amount = equity * risk_pct
    shares = int(dollar_amount // price)
    return max(shares, 0)


def volatility_targeted(
    equity: float,
    price: float,
    closes: pd.Series,
    target_vol: float = 0.02,
    lookback: int = 20,
) -> int:
    """Size positions based on recent volatility.

    In volatile markets, take smaller positions.
    In calm markets, take larger ones.
    """
    if len(closes) < lookback:
        return 0

    daily_returns = closes.pct_change().dropna().iloc[-lookback:]
    vol = daily_returns.std()

    if vol == 0 or np.isnan(vol):
        return 0

    dollar_amount = equity * (target_vol / vol)
    # Cap at 50% of equity to avoid going all-in
    dollar_amount = min(dollar_amount, equity * 0.5)
    shares = int(dollar_amount // price)
    return max(shares, 0)
