from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


def plot_price(
    df: pd.DataFrame,
    ticker: str = "",
    *,
    show_volume: bool = True,
    figsize: tuple[int, int] = (14, 7),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot OHLC price with optional volume bars."""
    has_volume = show_volume and "Volume" in df.columns

    if has_volume:
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
    else:
        fig, ax_price = plt.subplots(1, 1, figsize=figsize)
        ax_vol = None

    ax_price.plot(df.index, df["Close"], color="#2962FF", linewidth=1.2, label="Close")

    if all(c in df.columns for c in ["High", "Low"]):
        ax_price.fill_between(
            df.index, df["Low"], df["High"],
            alpha=0.08, color="#2962FF", label="High-Low range",
        )

    ax_price.set_ylabel("Price ($)")
    ax_price.set_title(f"{ticker} Price Chart" if ticker else "Price Chart")
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(True, alpha=0.3)

    if ax_vol is not None:
        colors = _volume_colors(df["Close"])
        ax_vol.bar(df.index, df["Volume"], color=colors, width=0.8, alpha=0.7)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, alpha=0.3)
        ax_vol.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: _fmt_volume(x))
        )

    bottom_ax = ax_vol if ax_vol is not None else ax_price
    bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    bottom_ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    return fig, fig.axes


def plot_returns(
    df: pd.DataFrame,
    ticker: str = "",
    *,
    figsize: tuple[int, int] = (14, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot daily returns distribution as a histogram."""
    returns = df["Close"].pct_change().dropna()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(returns, bins=80, color="#2962FF", alpha=0.7, edgecolor="white")
    ax.axvline(returns.mean(), color="red", linestyle="--", label=f"Mean: {returns.mean():.4f}")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{ticker} Daily Returns Distribution" if ticker else "Daily Returns Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def _volume_colors(close: pd.Series) -> list[str]:
    diff = close.diff()
    return ["#26A69A" if d >= 0 else "#EF5350" for d in diff]


def _fmt_volume(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.1f}B"
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return str(int(x))
