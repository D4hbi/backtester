from backtester.data import DataFeed
from backtester.engine import Engine
from strategies.sma_cross import SmaCrossover

feed = DataFeed()
df = feed.get("AAPL", start="2020-01-01", end="2024-01-01")

strategy = SmaCrossover("AAPL", fast_period=10, slow_period=50, quantity=100)
engine = Engine(strategy, df, initial_cash=100_000)
portfolio = engine.run()

print(f"Trades: {portfolio.trade_count}")
print(f"Final equity: ${portfolio.equity_curve[-1][1]:,.2f}")
print(f"Return: {portfolio.total_return:.2%}")
