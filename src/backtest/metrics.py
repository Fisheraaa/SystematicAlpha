"""src/backtest/metrics.py — performance metrics from equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualised_return(equity: pd.Series) -> float:
    """Compound annual growth rate from equity curve."""
    if len(equity) < 2:
        return np.nan
    n_years = len(equity) / 252
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    return float((1 + total_ret) ** (1 / n_years) - 1)


def sharpe_ratio(daily_log_returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe Ratio. rf is daily risk-free rate (default 0)."""
    excess = daily_log_returns - rf
    if excess.std() == 0:
        return np.nan
    return float(np.sqrt(252) * excess.mean() / excess.std())


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(-dd.min())


def calmar_ratio(equity: pd.Series) -> float:
    ann_ret = annualised_return(equity)
    mdd = max_drawdown(equity)
    return float(ann_ret / mdd) if mdd > 0 else np.nan


def information_ratio(
    strategy_rets: pd.Series, benchmark_rets: pd.Series
) -> float:
    """Annualised Information Ratio vs. benchmark."""
    active = strategy_rets - benchmark_rets
    if active.std() == 0:
        return np.nan
    return float(np.sqrt(252) * active.mean() / active.std())


def win_rate(daily_log_returns: pd.Series) -> float:
    """Fraction of trading days with positive return."""
    return float((daily_log_returns > 0).mean())


def monthly_turnover(trades_df: pd.DataFrame) -> float:
    """
    Average monthly turnover fraction.

    trades_df must have columns: date, direction ('buy'|'sell'), value.
    Turnover = total_sell_value / avg_equity.
    """
    if trades_df.empty:
        return np.nan
    sells = trades_df[trades_df["direction"] == "sell"]
    if sells.empty:
        return np.nan
    sells = sells.copy()
    sells["month"] = pd.to_datetime(sells["date"]).dt.to_period("M")
    monthly_sell = sells.groupby("month")["value"].sum()
    return float(monthly_sell.mean())


def compute_all_metrics(
    equity: pd.Series,
    benchmark_equity: pd.Series,
    trades_df: pd.DataFrame,
) -> dict:
    """
    Compute the full performance metric suite.

    Args:
        equity:           daily equity curve (absolute value, CNY)
        benchmark_equity: benchmark equity curve, same index
        trades_df:        trade log DataFrame

    Returns:
        dict of metric name → value.
    """
    daily_ret = np.log(equity / equity.shift(1)).dropna()
    bmark_ret = np.log(benchmark_equity / benchmark_equity.shift(1)).dropna()

    # Align
    common = daily_ret.index.intersection(bmark_ret.index)
    daily_ret  = daily_ret.reindex(common)
    bmark_ret  = bmark_ret.reindex(common)
    eq_common  = equity.reindex(common)

    return {
        "annual_return":     round(annualised_return(eq_common), 4),
        "sharpe_ratio":      round(sharpe_ratio(daily_ret), 4),
        "max_drawdown":      round(max_drawdown(eq_common), 4),
        "calmar_ratio":      round(calmar_ratio(eq_common), 4),
        "information_ratio": round(information_ratio(daily_ret, bmark_ret), 4),
        "win_rate":          round(win_rate(daily_ret), 4),
        "monthly_turnover":  round(monthly_turnover(trades_df), 0),
    }
