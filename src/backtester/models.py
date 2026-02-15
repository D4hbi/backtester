from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"


@dataclass(frozen=True)
class Bar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class Signal:
    date: datetime
    ticker: str
    side: Side
    quantity: int


@dataclass(frozen=True)
class Order:
    date: datetime
    ticker: str
    side: Side
    quantity: int
    order_type: OrderType = OrderType.MARKET


@dataclass(frozen=True)
class Fill:
    date: datetime
    ticker: str
    side: Side
    quantity: int
    fill_price: float
    commission: float


@dataclass
class Position:
    ticker: str
    quantity: int = 0
    avg_cost: float = 0.0

    @property
    def market_value(self) -> float:
        return 0.0

    def update(self, fill: Fill) -> float:
        if fill.side == Side.BUY:
            total_cost = self.avg_cost * self.quantity + fill.fill_price * fill.quantity
            self.quantity += fill.quantity
            self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0.0
            return -(fill.fill_price * fill.quantity + fill.commission)
        else:
            self.quantity -= fill.quantity
            if self.quantity == 0:
                self.avg_cost = 0.0
            return fill.fill_price * fill.quantity - fill.commission
