from backtester.data import DataFeed
from backtester.engine import Engine
from backtester import analytics
from backtester.report import generate
from strategies.sma_cross import SmaCrossover
from strategies.mean_reversion import BollingerMeanReversion
from strategies.momentum import RsiMacdMomentum

feed = DataFeed()
df = feed.get("AAPL", start="2020-01-01", end="2024-01-01")

strategies = [
    ("SMA Crossover", SmaCrossover("AAPL", fast_period=10, slow_period=50, quantity=100)),
    ("Mean Reversion", BollingerMeanReversion("AAPL", bb_period=20, bb_std=2.0, quantity=100)),
    ("RSI+MACD Momentum", RsiMacdMomentum("AAPL", quantity=100)),
]

header = (
    f"{'Strategy':<20} {'Trades':>7} {'Return':>8} {'Sharpe':>7} "
    f"{'Sortino':>8} {'MaxDD':>7} {'WinRate':>8} {'PF':>6}"
)
print(header)
print("-" * len(header))

for name, strategy in strategies:
    engine = Engine(strategy, df, initial_cash=100_000)
    portfolio = engine.run()

    stats = analytics.summary(portfolio)
    print(
        f"{name:<20} {stats['trade_count']:>7} "
        f"{stats['total_return']:>7.2%} "
        f"{stats['sharpe_ratio']:>7.2f} "
        f"{stats['sortino_ratio']:>8.2f} "
        f"{stats['max_drawdown']:>7.2%} "
        f"{stats['win_rate']:>7.2%} "
        f"{stats['profit_factor']:>6.2f}"
    )

    # Generate HTML tearsheet for each strategy
    generate(portfolio, strategy_name=name, output_path=f"reports/{name.lower().replace(' ', '_')}.html")

print("\nReports saved to reports/")
