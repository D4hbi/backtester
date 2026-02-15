from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.broker import Broker
from backtester.engine import Engine
from backtester.models import Bar, Fill, Order, OrderType, Position, Side, Signal
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy
from strategies.sma_cross import SmaCrossover


def _make_ohlcv(n_days: int = 200, start: str = "2022-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, periods=n_days, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n_days))
    close = np.maximum(close, 1.0)

    df = pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.3, n_days),
            "High": close + np.abs(rng.normal(0, 1, n_days)),
            "Low": close - np.abs(rng.normal(0, 1, n_days)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 50_000_000, n_days),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def _make_trending_up(n_days: int = 200) -> pd.DataFrame:
    dates = pd.bdate_range(start="2022-01-01", periods=n_days, freq="B")
    # Dip first, then rally — guarantees a crossover happens
    dip = np.linspace(0, -10, n_days // 3)
    rally = np.linspace(-10, 40, n_days - n_days // 3)
    close = 100.0 + np.concatenate([dip, rally])

    df = pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(n_days, 10_000_000),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


class TestModels:
    def test_position_buy(self):
        pos = Position(ticker="AAPL")
        fill = Fill(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.BUY,
            quantity=100,
            fill_price=150.0,
            commission=0.50,
        )
        cash_delta = pos.update(fill)
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0
        assert cash_delta == -(150.0 * 100 + 0.50)

    def test_position_sell(self):
        pos = Position(ticker="AAPL", quantity=100, avg_cost=150.0)
        fill = Fill(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.SELL,
            quantity=100,
            fill_price=160.0,
            commission=0.50,
        )
        cash_delta = pos.update(fill)
        assert pos.quantity == 0
        assert pos.avg_cost == 0.0
        assert cash_delta == 160.0 * 100 - 0.50


class TestBroker:
    def test_buy_slippage(self):
        broker = Broker(slippage_pct=0.01, commission_per_share=0.0)
        order = Order(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.BUY,
            quantity=100,
        )
        bar = Bar(
            date=pd.Timestamp("2023-01-01"),
            open=100.0, high=105.0, low=99.0, close=102.0, volume=1_000_000,
        )
        fill = broker.execute(order, bar)
        assert fill.fill_price == 101.0  # 100 * 1.01
        assert fill.quantity == 100

    def test_sell_slippage(self):
        broker = Broker(slippage_pct=0.01, commission_per_share=0.0)
        order = Order(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.SELL,
            quantity=100,
        )
        bar = Bar(
            date=pd.Timestamp("2023-01-01"),
            open=100.0, high=105.0, low=99.0, close=102.0, volume=1_000_000,
        )
        fill = broker.execute(order, bar)
        assert fill.fill_price == 99.0  # 100 * 0.99

    def test_commission(self):
        broker = Broker(slippage_pct=0.0, commission_per_share=0.01)
        order = Order(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.BUY,
            quantity=200,
        )
        bar = Bar(
            date=pd.Timestamp("2023-01-01"),
            open=100.0, high=105.0, low=99.0, close=102.0, volume=1_000_000,
        )
        fill = broker.execute(order, bar)
        assert fill.commission == 2.0  # 200 * 0.01


class TestPortfolio:
    def test_initial_state(self):
        portfolio = Portfolio(initial_cash=50_000.0)
        assert portfolio.cash == 50_000.0
        assert portfolio.total_return == 0.0
        assert portfolio.trade_count == 0

    def test_buy_reduces_cash(self):
        portfolio = Portfolio(initial_cash=100_000.0)
        fill = Fill(
            date=pd.Timestamp("2023-01-01"),
            ticker="AAPL",
            side=Side.BUY,
            quantity=100,
            fill_price=150.0,
            commission=0.50,
        )
        portfolio.process_fill(fill)
        assert portfolio.cash == 100_000.0 - (150.0 * 100 + 0.50)
        assert portfolio.positions["AAPL"].quantity == 100

    def test_equity_curve(self):
        portfolio = Portfolio(initial_cash=100_000.0)
        portfolio.record_equity(pd.Timestamp("2023-01-01"), {})
        assert len(portfolio.equity_curve) == 1
        assert portfolio.equity_curve[0][1] == 100_000.0


class TestEngine:
    def test_runs_without_error(self):
        df = _make_ohlcv(200)
        strategy = SmaCrossover("TEST", fast_period=10, slow_period=50, quantity=100)
        engine = Engine(strategy, df)
        portfolio = engine.run()

        assert len(portfolio.equity_curve) == 200
        assert portfolio.equity_curve[0][1] == 100_000.0

    def test_trending_market_produces_trades(self):
        df = _make_trending_up(200)
        strategy = SmaCrossover("TEST", fast_period=10, slow_period=50, quantity=100)
        engine = Engine(strategy, df)
        portfolio = engine.run()

        assert portfolio.trade_count > 0

    def test_no_trades_without_enough_data(self):
        df = _make_ohlcv(30)  # less than slow_period
        strategy = SmaCrossover("TEST", fast_period=10, slow_period=50, quantity=100)
        engine = Engine(strategy, df)
        portfolio = engine.run()

        assert portfolio.trade_count == 0

    def test_custom_parameters(self):
        df = _make_ohlcv(200)
        strategy = SmaCrossover("TEST", fast_period=10, slow_period=50, quantity=100)
        engine = Engine(
            strategy, df,
            initial_cash=50_000.0,
            slippage_pct=0.0,
            commission_per_share=0.0,
        )
        portfolio = engine.run()

        assert portfolio.initial_cash == 50_000.0
