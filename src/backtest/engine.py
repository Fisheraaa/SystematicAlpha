"""src/backtest/engine.py

Core backtest engine. Executes one full run (csi300, csi500, or combined).

State machine per trading day:
  Step 1: Mark-to-market (update equity)
  Step 2: Daily risk check (drawdown → scale invested fraction)
  Step 3: On rebalance dates — generate target weights → compute required trades
  Step 4: Execute sells (T+1-eligible only), then buys
  Step 5: Log daily state and trades

Public API
----------
run_backtest(run_id, start, end, ic_weights) -> (daily_state_df, trades_df)
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, write_parquet, PROCESSED, RESULTS
from src.data.universe import get_universe, get_combined_universe, rebalance_dates
from src.factors import load_all_factors
from src.strategy.regime import build_regime_series
from src.strategy.composite import compute_composite
from src.strategy.portfolio import compute_target_weights
from src.backtest.costs import buy_cost, sell_proceeds, is_shanghai_stock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ts_code:    str
    shares:     float
    entry_date: pd.Timestamp
    avg_cost:   float        # average cost per share (incl. fees)
    membership: str = "csi300"   # 'csi300' | 'csi500' | 'both' (for combined run)


@dataclass
class DailyRecord:
    date:      str
    cash:      float
    equity:    float
    drawdown:  float
    n_stocks:  int
    turnover:  float         # fraction of portfolio value traded today


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:

    def __init__(self, run_id: str, ic_weights: dict[str, float]):
        self.run_id    = run_id
        self.ic_weights = ic_weights
        self.cfg       = load_config()
        self.cfg_bt    = self.cfg["backtest"]
        self.cfg_r     = self.cfg["risk"]

        self.cash:     float = float(self.cfg_bt["initial_capital"])
        self.positions: dict[str, Position] = {}
        self.peak_equity: float = self.cash
        self.daily_records: list[DailyRecord] = []
        self.trades: list[dict] = []

        # Preload all required data
        logger.info("[%s] Loading price data …", run_id)
        self.prices  = pd.read_parquet(PROCESSED / "prices.parquet")
        self.opens   = pd.read_parquet(PROCESSED / "opens.parquet")
        self.returns = pd.read_parquet(PROCESSED / "returns.parquet")
        self.factors = load_all_factors()
        self.regime_series = build_regime_series(self.prices)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def market_value(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            price = self._latest_price(pos.ts_code)
            total += pos.shares * price
        return total

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    @property
    def drawdown(self) -> float:
        eq = self.equity
        if self.peak_equity > 0:
            return max(0.0, 1.0 - eq / self.peak_equity)
        return 0.0

    def _latest_price(self, ts_code: str, date: pd.Timestamp | None = None) -> float:
        if date is None:
            date = self._current_date
        col = self.prices.get(ts_code)
        if col is None:
            return 0.0
        try:
            return float(col.loc[:date].dropna().iloc[-1])
        except IndexError:
            return 0.0

    def _open_price(self, ts_code: str, date: pd.Timestamp) -> float:
        col = self.opens.get(ts_code)
        if col is None or date not in col.index:
            return self._latest_price(ts_code, date)
        val = col.loc[date]
        return float(val) if not np.isnan(val) else self._latest_price(ts_code, date)

    # ------------------------------------------------------------------
    # Universe helpers
    # ------------------------------------------------------------------

    def _get_universe(self, date: pd.Timestamp) -> list[str]:
        if self.run_id == "combined":
            df = get_combined_universe(date)
            return df["ts_code"].tolist()
        index_code = "000300.SH" if self.run_id == "csi300" else "000905.SH"
        return get_universe(date, index_code)

    def _membership(self, ts_code: str, date: pd.Timestamp) -> str:
        if self.run_id != "combined":
            return self.run_id
        df = get_combined_universe(date)
        row = df[df["ts_code"] == ts_code]
        return row["index_membership"].iloc[0] if not row.empty else "csi300"

    def _slippage_for(self, ts_code: str, date: pd.Timestamp) -> float:
        membership = self._membership(ts_code, date)
        if membership == "csi500":
            return self.cfg_bt["slippage_csi500"]
        return self.cfg_bt["slippage_csi300"]

    # ------------------------------------------------------------------
    # State machine steps
    # ------------------------------------------------------------------

    def _mark_to_market(self, date: pd.Timestamp) -> None:
        eq = self.equity
        if eq > self.peak_equity:
            self.peak_equity = eq

    def _execute_sells(
        self,
        target_codes: set[str],
        date: pd.Timestamp,
    ) -> float:
        """Sell positions not in target_codes. Respect T+1. Return proceeds."""
        proceeds = 0.0
        to_sell = [
            code for code in list(self.positions.keys())
            if code not in target_codes
        ]
        for code in to_sell:
            pos = self.positions[code]
            # T+1: must have held since at least the previous trading day
            if pos.entry_date >= date:
                logger.debug("T+1 constraint: cannot sell %s purchased today.", code)
                continue
            price = self._open_price(code, date)
            if price <= 0:
                continue
            net = sell_proceeds(
                price, pos.shares,
                is_shanghai=is_shanghai_stock(code),
                run_id=self.run_id,
            )
            proceeds += net
            self.trades.append({
                "date":      date.strftime("%Y%m%d"),
                "ts_code":   code,
                "direction": "sell",
                "shares":    pos.shares,
                "price":     price,
                "value":     net,
            })
            del self.positions[code]
        self.cash += proceeds
        return proceeds

    def _execute_buys(
        self,
        target_weights: pd.Series,
        date: pd.Timestamp,
    ) -> None:
        """Buy stocks in target_weights not currently held."""
        total_equity = self.equity
        for code, target_w in target_weights.items():
            if code in self.positions:
                continue   # already held; partial rebalancing not implemented
            target_value = target_w * total_equity
            price = self._open_price(code, date)
            if price <= 0 or target_value <= 0:
                continue
            shares = target_value / price
            cost = buy_cost(
                price, shares,
                is_shanghai=is_shanghai_stock(code),
                run_id=self.run_id,
            )
            if cost > self.cash:
                # Scale back to available cash
                shares = (self.cash * 0.99) / (price * (1 + self.cfg_bt["commission_rate"]
                                                         + self.cfg_bt["slippage_csi300"]))
                cost = buy_cost(price, shares, is_shanghai=is_shanghai_stock(code),
                                run_id=self.run_id)
            if shares <= 0 or cost > self.cash:
                continue

            self.cash -= cost
            self.positions[code] = Position(
                ts_code=code,
                shares=shares,
                entry_date=date,
                avg_cost=price,
                membership=self._membership(code, date),
            )
            self.trades.append({
                "date":      date.strftime("%Y%m%d"),
                "ts_code":   code,
                "direction": "buy",
                "shares":    shares,
                "price":     price,
                "value":     cost,
            })

    def _rebalance(self, date: pd.Timestamp) -> None:
        universe   = self._get_universe(date)
        regime     = str(self.regime_series.get(date, "momentum"))

        composite = compute_composite(
            date, universe, self.factors, regime, self.ic_weights
        )
        if composite.empty:
            return

        target_weights = compute_target_weights(
            date, composite, regime, drawdown=self.drawdown
        )
        if target_weights.empty:
            return

        target_codes = set(target_weights.index)
        eq_before = self.equity

        self._execute_sells(target_codes, date)
        self._execute_buys(target_weights, date)

        eq_after = self.equity
        turnover = abs(eq_after - eq_before) / max(eq_before, 1)
        self._log_day(date, turnover)

    def _log_day(self, date: pd.Timestamp, turnover: float = 0.0) -> None:
        self.daily_records.append(DailyRecord(
            date=date.strftime("%Y%m%d"),
            cash=self.cash,
            equity=self.equity,
            drawdown=self.drawdown,
            n_stocks=len(self.positions),
            turnover=turnover,
        ))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        start: str,
        end: str,
        output_stem: str = "daily_state",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the backtest from `start` to `end`.

        Args:
            start:       start date 'YYYY-MM-DD'
            end:         end date   'YYYY-MM-DD'
            output_stem: filename stem for saved parquet files.
                         Main backtest uses 'daily_state' (default).
                         Walk-forward windows use e.g. 'wf_window_03'
                         so they never overwrite the full-period result.

        Returns:
            (daily_state_df, trades_df)
        """
        cal = self.returns.loc[start:end].index
        reb_set = set(rebalance_dates(start, end))

        for date in cal:
            self._current_date = date
            self._mark_to_market(date)

            date_str = date.strftime("%Y%m%d")
            if date_str in reb_set:
                self._rebalance(date)
            else:
                self._log_day(date)

        daily_df = pd.DataFrame([
            {"date": r.date, "cash": r.cash, "equity": r.equity,
             "drawdown": r.drawdown, "n_stocks": r.n_stocks, "turnover": r.turnover}
            for r in self.daily_records
        ])
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        daily_df = daily_df.set_index("date")

        trades_df = pd.DataFrame(self.trades)

        # Persist — use output_stem so walk-forward windows don't overwrite
        # the main full-period daily_state.parquet.
        out_dir = RESULTS / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(daily_df, out_dir / f"{output_stem}.parquet")
        if not trades_df.empty:
            write_parquet(trades_df, out_dir / f"{output_stem}_trades.parquet")

        logger.info(
            "[%s] Backtest complete: %d days, final equity %.0f CNY.",
            self.run_id, len(daily_df), daily_df["equity"].iloc[-1],
        )
        return daily_df, trades_df


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_backtest(
    run_id: str,
    start: str,
    end: str,
    ic_weights: dict[str, float],
    output_stem: str = "daily_state",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Instantiate engine and run. Returns (daily_state_df, trades_df)."""
    engine = BacktestEngine(run_id, ic_weights)
    return engine.run(start, end, output_stem=output_stem)
