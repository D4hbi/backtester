from backtester.data import DataFeed
from backtester.engine import Engine
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

print(f"{'Strategy':<20} {'Trades':>7} {'Final Equity':>14} {'Return':>8}")
print("-" * 53)

for name, strategy in strategies:
    engine = Engine(strategy, df, initial_cash=100_000)
    portfolio = engine.run()

    print(
        f"{name:<20} {portfolio.trade_count:>7} "
        f"${portfolio.equity_curve[-1][1]:>12,.2f} "
        f"{portfolio.total_return:>7.2%}"
    )
