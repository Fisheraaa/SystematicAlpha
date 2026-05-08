"""src/validation/quantile.py

Quintile (Q1–Q5) long-only backtest for factor validation.

For each rebalance date:
  1. Rank stocks by factor value within the eligible universe.
  2. Split into 5 equal-sized groups.
  3. Hold each group at equal weight until next rebalance.
  4. Record group returns.

Outputs per factor per universe:
  - Annualised return by quintile
  - Monotonicity check (is Q5 > Q4 > … > Q1?)
  - Q5 vs. benchmark Information Ratio
  - Q5 - Q1 spread (reference long-short alpha)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.utils import load_config, write_parquet, RESULTS, PROCESSED
from src.factors import load_all_factors
from src.data.universe import get_universe, rebalance_dates

logger = logging.getLogger(__name__)

N_QUINTILES = 5


def _quintile_returns(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    universe_fn,
    reb_dates: list[str],
) -> pd.DataFrame:
    """
    Compute equal-weighted quintile portfolio returns for one factor.

    Returns a DataFrame indexed by date with columns Q1…Q5.
    """
    reb_timestamps = [pd.Timestamp(d) for d in reb_dates]
    all_dates = returns_df.index

    quintile_rets: list[dict] = []

    for i, reb_dt in enumerate(reb_timestamps):
        # Determine the holding period: from reb_dt+1 to next reb_dt (inclusive)
        if i + 1 < len(reb_timestamps):
            next_reb = reb_timestamps[i + 1]
        else:
            break

        if reb_dt not in factor_df.index:
            continue

        universe = universe_fn(reb_dt)
        if len(universe) < N_QUINTILES * 4:
            continue

        factor_t = factor_df.loc[reb_dt, universe].dropna()
        if len(factor_t) < N_QUINTILES * 4:
            continue

        # Assign quintile labels (1=lowest, 5=highest)
        labels = pd.qcut(factor_t.rank(method="first"), N_QUINTILES, labels=False) + 1

        # Holding period dates: day after reb_dt up to (but not including) next_reb
        hold_mask = (all_dates > reb_dt) & (all_dates <= next_reb)
        hold_dates = all_dates[hold_mask]
        if len(hold_dates) == 0:
            continue

        for q in range(1, N_QUINTILES + 1):
            stocks_q = labels[labels == q].index.tolist()
            # Equal-weight daily return for this quintile
            period_rets = returns_df.loc[hold_dates, stocks_q].mean(axis=1)
            for dt in hold_dates:
                quintile_rets.append({"date": dt, f"Q{q}": period_rets.get(dt, np.nan)})

    if not quintile_rets:
        return pd.DataFrame()

    df = pd.DataFrame(quintile_rets)
    df = df.groupby("date").first().sort_index()
    return df


def _annualised(daily_log_ret: pd.Series) -> float:
    """Annualised return from a daily log-return series."""
    total = daily_log_ret.dropna().sum()
    n_years = len(daily_log_ret.dropna()) / 252
    return float(np.exp(total / n_years) - 1) if n_years > 0 else np.nan


def run_quantile_analysis(
    run_id: str,
    is_start: str,
    is_end: str,
) -> pd.DataFrame:
    """
    Run quintile analysis for all factors in one universe over IS period.

    Returns a DataFrame: factor × quintile → annualised return.
    Also persists results to data/results/{run_id}/quantile_returns.parquet.
    """
    assert run_id in ("csi300", "csi500")
    index_code = "000300.SH" if run_id == "csi300" else "000905.SH"

    returns_df = pd.read_parquet(PROCESSED / "returns.parquet")
    factors    = load_all_factors()
    reb_dates  = rebalance_dates(is_start, is_end)

    def universe_fn(dt):
        return get_universe(dt, index_code)

    result_rows = []

    for fname, fdf in factors.items():
        if fname in ("rvol_20d", "atr_14d"):
            continue

        logger.info("[%s] Quintile analysis for factor: %s", run_id, fname)
        q_rets = _quintile_returns(fdf, returns_df, universe_fn, reb_dates)

        if q_rets.empty:
            continue

        row = {"factor": fname}
        ann_rets = []
        for q in range(1, N_QUINTILES + 1):
            col = f"Q{q}"
            if col in q_rets.columns:
                ann = _annualised(q_rets[col])
                row[col] = round(ann, 4)
                ann_rets.append(ann)
            else:
                row[col] = np.nan
                ann_rets.append(np.nan)

        # Monotonicity check: is each quintile strictly greater than the previous?
        valid = [x for x in ann_rets if not np.isnan(x)]
        monotonic = all(valid[i] < valid[i + 1] for i in range(len(valid) - 1))
        row["monotonic"] = monotonic
        row["spread_Q5_Q1"] = round(ann_rets[4] - ann_rets[0], 4) if len(ann_rets) == 5 else np.nan

        result_rows.append(row)

    result_df = pd.DataFrame(result_rows).set_index("factor")

    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(result_df, out_dir / "quantile_returns.parquet")
    logger.info("[%s] Quintile analysis complete.", run_id)

    return result_df
