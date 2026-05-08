"""src/backtest/costs.py

A-share transaction cost model (as of 2024, post stamp-duty reduction).

Round-trip cost breakdown (CSI 300 example):
  Buy:  commission 0.025% + exchange fee 0.001% + slippage 0.05% ≈ 0.076%
  Sell: commission 0.025% + stamp duty 0.05% + exchange fee 0.001% + slippage 0.05% ≈ 0.126%
  Total round-trip: ≈ 0.25%

CSI 500 uses higher slippage (0.10%) → round-trip ≈ 0.30%.
"""
from __future__ import annotations

from src.utils import load_config


def _cfg():
    return load_config()["backtest"]


def buy_cost(price: float, shares: float, is_shanghai: bool = True, run_id: str = "csi300") -> float:
    """
    Total cash outflow for a buy order.
    = price × shares × (1 + commission + exchange_fee + slippage)
    """
    c = _cfg()
    slip = _slippage(run_id)
    rate = c["commission_rate"] + (c["exchange_fee_rate"] if is_shanghai else 0) + slip
    return price * shares * (1 + rate)


def sell_proceeds(price: float, shares: float, is_shanghai: bool = True, run_id: str = "csi300") -> float:
    """
    Net cash received from a sell order.
    = price × shares × (1 - commission - stamp_duty - exchange_fee - slippage)
    """
    c = _cfg()
    slip = _slippage(run_id)
    rate = c["commission_rate"] + c["stamp_duty_rate"] + (c["exchange_fee_rate"] if is_shanghai else 0) + slip
    return price * shares * (1 - rate)


def _slippage(run_id: str) -> float:
    c = _cfg()
    if run_id == "csi300":
        return c["slippage_csi300"]
    elif run_id == "csi500":
        return c["slippage_csi500"]
    else:
        # combined: average; actual per-stock lookup done in engine via membership tag
        return (c["slippage_csi300"] + c["slippage_csi500"]) / 2


def round_trip_cost_rate(run_id: str = "csi300") -> float:
    """Return approximate round-trip cost rate for documentation/stress-tests."""
    c = _cfg()
    slip = _slippage(run_id)
    buy_r  = c["commission_rate"] + c["exchange_fee_rate"] + slip
    sell_r = c["commission_rate"] + c["stamp_duty_rate"] + c["exchange_fee_rate"] + slip
    return buy_r + sell_r


def is_shanghai_stock(ts_code: str) -> bool:
    """Shanghai-listed stocks end with '.SH'."""
    return ts_code.endswith(".SH")
