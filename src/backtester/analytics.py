from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf

from backtester.models import Side
from backtester.portfolio import Portfolio

TRADING_DAYS_PER_YEAR = 252


def equity_series(portfolio: Portfolio) -> pd.Series:
    dates, values = zip(*portfolio.equity_curve)
    return pd.Series(values, index=pd.DatetimeIndex(dates), name="Equity")


def daily_returns(portfolio: Portfolio) -> pd.Series:
    eq = equity_series(portfolio)
    return eq.pct_change().dropna()


def sharpe_ratio(portfolio: Portfolio, risk_free_rate: float = 0.0) -> float:
    returns = daily_returns(portfolio)
    if returns.std() == 0:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(portfolio: Portfolio, risk_free_rate: float = 0.0) -> float:
    returns = daily_returns(portfolio)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf")
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(portfolio: Portfolio) -> float:
    eq = equity_series(portfolio)
    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    return float(drawdown.min())


def drawdown_series(portfolio: Portfolio) -> pd.Series:
    eq = equity_series(portfolio)
    peak = eq.cummax()
    return (eq - peak) / peak


def calmar_ratio(portfolio: Portfolio) -> float:
    dd = max_drawdown(portfolio)
    if dd == 0:
        return float("inf")
    eq = equity_series(portfolio)
    total_days = (eq.index[-1] - eq.index[0]).days
    annual_return = (eq.iloc[-1] / eq.iloc[0]) ** (365 / total_days) - 1
    return float(annual_return / abs(dd))


def win_rate(portfolio: Portfolio) -> float:
    trades = _pair_trades(portfolio)
    if not trades:
        return 0.0
    wins = sum(1 for pnl in trades if pnl > 0)
    return wins / len(trades)


def profit_factor(portfolio: Portfolio) -> float:
    trades = _pair_trades(portfolio)
    if not trades:
        return 0.0
    gross_profit = sum(pnl for pnl in trades if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trades if pnl < 0))
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def adf_test(portfolio: Portfolio) -> dict:
    returns = daily_returns(portfolio)
    stat, pvalue, usedlag, nobs, critical, icbest = adfuller(returns, autolag="AIC")
    return {
        "statistic": float(stat),
        "p_value": float(pvalue),
        "is_stationary": bool(pvalue < 0.05),
    }


def autocorrelation(portfolio: Portfolio, nlags: int = 10) -> pd.Series:
    returns = daily_returns(portfolio)
    acf_values = acf(returns, nlags=nlags, fft=True)
    return pd.Series(acf_values, index=range(nlags + 1), name="ACF")


def summary(portfolio: Portfolio) -> dict:
    eq = equity_series(portfolio)
    return {
        "total_return": portfolio.total_return,
        "annual_return": _annual_return(eq),
        "sharpe_ratio": sharpe_ratio(portfolio),
        "sortino_ratio": sortino_ratio(portfolio),
        "max_drawdown": max_drawdown(portfolio),
        "calmar_ratio": calmar_ratio(portfolio),
        "win_rate": win_rate(portfolio),
        "profit_factor": profit_factor(portfolio),
        "trade_count": portfolio.trade_count,
        "start_date": str(eq.index[0].date()),
        "end_date": str(eq.index[-1].date()),
    }


def _annual_return(eq: pd.Series) -> float:
    total_days = (eq.index[-1] - eq.index[0]).days
    if total_days == 0:
        return 0.0
    return float((eq.iloc[-1] / eq.iloc[0]) ** (365 / total_days) - 1)


def _pair_trades(portfolio: Portfolio) -> list[float]:
    pnls = []
    buy_cost = 0.0
    for fill in portfolio.fills:
        if fill.side == Side.BUY:
            buy_cost = fill.fill_price * fill.quantity + fill.commission
        elif fill.side == Side.SELL:
            sell_proceeds = fill.fill_price * fill.quantity - fill.commission
            pnls.append(sell_proceeds - buy_cost)
    return pnls
