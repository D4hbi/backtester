from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from backtester import analytics
from backtester.portfolio import Portfolio


def generate(
    portfolio: Portfolio,
    strategy_name: str = "Strategy",
    benchmark: Portfolio | None = None,
    output_path: str | Path = "tearsheet.html",
) -> Path:
    output_path = Path(output_path)
    stats = analytics.summary(portfolio)

    equity_chart = _plot_equity(portfolio, benchmark, strategy_name)
    drawdown_chart = _plot_drawdown(portfolio, strategy_name)
    monthly_heatmap = _plot_monthly_returns(portfolio, strategy_name)
    rolling_sharpe_chart = _plot_rolling_sharpe(portfolio, strategy_name)

    html = _build_html(
        strategy_name=strategy_name,
        stats=stats,
        equity_chart=equity_chart,
        drawdown_chart=drawdown_chart,
        monthly_heatmap=monthly_heatmap,
        rolling_sharpe_chart=rolling_sharpe_chart,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[Report] Tearsheet saved to {output_path}")
    return output_path


def _fig_to_base64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plot_equity(
    portfolio: Portfolio,
    benchmark: Portfolio | None,
    strategy_name: str,
) -> str:
    fig, ax = plt.subplots(figsize=(12, 5))

    eq = analytics.equity_series(portfolio)
    ax.plot(eq.index, eq.values, color="#2962FF", linewidth=1.2, label=strategy_name)

    if benchmark is not None:
        bench_eq = analytics.equity_series(benchmark)
        ax.plot(bench_eq.index, bench_eq.values, color="#999999", linewidth=1.0, label="Benchmark (SPY)", linestyle="--")

    ax.set_title("Equity Curve")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_drawdown(portfolio: Portfolio, strategy_name: str) -> str:
    fig, ax = plt.subplots(figsize=(12, 3))

    dd = analytics.drawdown_series(portfolio)
    ax.fill_between(dd.index, dd.values, 0, color="#EF5350", alpha=0.5)
    ax.plot(dd.index, dd.values, color="#EF5350", linewidth=0.8)

    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_monthly_returns(portfolio: Portfolio, strategy_name: str) -> str:
    returns = analytics.daily_returns(portfolio)
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    table = pivot.pivot_table(index="year", columns="month", values="return", aggfunc="sum")
    table.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(table.columns)]

    fig, ax = plt.subplots(figsize=(12, max(3, len(table) * 0.6)))
    im = ax.imshow(table.values * 100, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)

    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            val = table.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8,
                        color="black" if abs(val) < 0.05 else "white")

    ax.set_title("Monthly Returns")
    fig.colorbar(im, ax=ax, label="Return %", shrink=0.8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_rolling_sharpe(portfolio: Portfolio, strategy_name: str, window: int = 63) -> str:
    fig, ax = plt.subplots(figsize=(12, 3))

    returns = analytics.daily_returns(portfolio)
    rolling = returns.rolling(window).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )

    ax.plot(rolling.index, rolling.values, color="#2962FF", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="#26A69A", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(-1, color="#EF5350", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_title(f"Rolling Sharpe Ratio ({window}-day)")
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _build_html(
    strategy_name: str,
    stats: dict,
    equity_chart: str,
    drawdown_chart: str,
    monthly_heatmap: str,
    rolling_sharpe_chart: str,
) -> str:
    def fmt(key: str) -> str:
        val = stats[key]
        if key in ("total_return", "annual_return", "max_drawdown", "win_rate"):
            return f"{val:.2%}"
        if isinstance(val, float):
            return f"{val:.2f}"
        return str(val)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{strategy_name} - Backtest Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #fafafa; color: #333; }}
    h1 {{ color: #1a1a1a; border-bottom: 2px solid #2962FF; padding-bottom: 10px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
    .metric {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
    .metric-value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
    .chart {{ margin: 24px 0; }}
    .chart img {{ width: 100%; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .positive {{ color: #26A69A; }}
    .negative {{ color: #EF5350; }}
</style>
</head>
<body>
<h1>{strategy_name} - Backtest Report</h1>
<p>{stats['start_date']} to {stats['end_date']} | {stats['trade_count']} trades</p>

<div class="metrics">
    <div class="metric">
        <div class="metric-label">Total Return</div>
        <div class="metric-value {'positive' if stats['total_return'] >= 0 else 'negative'}">{fmt('total_return')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Annual Return</div>
        <div class="metric-value {'positive' if stats['annual_return'] >= 0 else 'negative'}">{fmt('annual_return')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Sharpe Ratio</div>
        <div class="metric-value">{fmt('sharpe_ratio')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Sortino Ratio</div>
        <div class="metric-value">{fmt('sortino_ratio')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Max Drawdown</div>
        <div class="metric-value negative">{fmt('max_drawdown')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Calmar Ratio</div>
        <div class="metric-value">{fmt('calmar_ratio')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Win Rate</div>
        <div class="metric-value">{fmt('win_rate')}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Profit Factor</div>
        <div class="metric-value">{fmt('profit_factor')}</div>
    </div>
</div>

<div class="chart"><img src="data:image/png;base64,{equity_chart}"></div>
<div class="chart"><img src="data:image/png;base64,{drawdown_chart}"></div>
<div class="chart"><img src="data:image/png;base64,{monthly_heatmap}"></div>
<div class="chart"><img src="data:image/png;base64,{rolling_sharpe_chart}"></div>

</body>
</html>"""
