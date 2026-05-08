"""src/validation/ic_analysis.py

Information Coefficient analysis, per universe.

Functions
---------
compute_ic_series(factor_df, returns_df, universe_codes, dates) -> pd.Series
    Daily Spearman IC between factor_t and return_{t+1}.

ic_summary(ic_series) -> dict
    Mean IC, std, ICIR, t-stat, p-value, pct_positive.

ic_decay(factor_df, returns_df, universe_codes, dates, horizons) -> pd.DataFrame
    IC at multiple forward horizons (1, 2, 3, 5, 10 days).

run_validation(run_id, is_start, is_end) -> dict
    Full validation output for one universe over the IS period.
    Writes results to data/results/{run_id}/ic_summary.parquet.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import load_config, write_parquet, RESULTS, PROCESSED
from src.factors import load_all_factors
from src.data.universe import get_universe, rebalance_dates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core IC computation
# ---------------------------------------------------------------------------

def _rank_normalise(s: pd.Series) -> pd.Series:
    """Cross-sectionally rank-normalise: maps values to (0, 1)."""
    r = s.rank(method="average")
    return (r - 0.5) / len(r.dropna())


def compute_ic_series(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    universe_fn,           # callable(date) -> list[str]
    dates: Iterable[pd.Timestamp],
    horizon: int = 1,
) -> pd.Series:
    """
    Compute daily Spearman IC between factor_t and forward return_{t+horizon}.

    Args:
        factor_df:   wide factor DataFrame (date × ts_code)
        returns_df:  wide returns DataFrame (date × ts_code), daily log-returns
        universe_fn: function that returns eligible ts_codes for a given date
        dates:       iterable of dates to compute IC on
        horizon:     forward return horizon in trading days

    Returns:
        pd.Series indexed by date, values = Spearman IC.
    """
    dates = list(dates)
    ic_values: dict[pd.Timestamp, float] = {}

    for dt in dates:
        # Forward return: sum of log-returns over the next `horizon` days
        # We need to find the horizon-th date after dt in the returns index.
        try:
            pos = returns_df.index.get_loc(dt)
        except KeyError:
            continue
        if pos + horizon >= len(returns_df):
            continue

        fwd_ret = returns_df.iloc[pos + 1 : pos + 1 + horizon].sum(axis=0)

        # Get factor value at dt
        if dt not in factor_df.index:
            continue
        factor_t = factor_df.loc[dt]

        # Restrict to universe
        universe = universe_fn(dt)
        if len(universe) < 20:
            continue

        # Align
        common = factor_t.dropna().index.intersection(fwd_ret.dropna().index)
        common = [c for c in common if c in universe]
        if len(common) < 20:
            continue

        f_vals = _rank_normalise(factor_t[common])
        r_vals = _rank_normalise(fwd_ret[common])

        # Spearman IC = Pearson on ranks (already ranked above)
        corr, _ = stats.pearsonr(f_vals.values, r_vals.values)
        ic_values[dt] = corr

    return pd.Series(ic_values, name="IC")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def ic_summary(ic_series: pd.Series) -> dict:
    """Compute summary statistics for an IC time series."""
    clean = ic_series.dropna()
    if len(clean) == 0:
        return {}
    mean_ic  = float(clean.mean())
    std_ic   = float(clean.std())
    icir     = mean_ic / std_ic if std_ic > 0 else np.nan
    t_stat   = mean_ic / (std_ic / np.sqrt(len(clean))) if std_ic > 0 else np.nan
    p_value  = float(2 * stats.t.sf(abs(t_stat), df=len(clean) - 1)) if not np.isnan(t_stat) else np.nan
    pct_pos  = float((clean > 0).mean())
    return {
        "n_obs":      len(clean),
        "mean_ic":    round(mean_ic,  4),
        "std_ic":     round(std_ic,   4),
        "icir":       round(icir,     4),
        "t_stat":     round(t_stat,   4),
        "p_value":    round(p_value,  4),
        "pct_positive": round(pct_pos, 4),
    }


# ---------------------------------------------------------------------------
# IC decay
# ---------------------------------------------------------------------------

def ic_decay(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    universe_fn,
    dates: Iterable[pd.Timestamp],
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute mean IC at multiple forward horizons.

    Returns a DataFrame with columns = horizons, rows = summary stats.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10]

    rows = []
    for h in horizons:
        ic = compute_ic_series(factor_df, returns_df, universe_fn, dates, horizon=h)
        summary = ic_summary(ic)
        summary["horizon"] = h
        rows.append(summary)

    return pd.DataFrame(rows).set_index("horizon")


# ---------------------------------------------------------------------------
# Full validation runner (per run_id)
# ---------------------------------------------------------------------------

def run_validation(
    run_id: str,
    is_start: str,
    is_end: str,
) -> dict[str, pd.DataFrame]:
    """
    Run full IC validation for a single universe over the IS period.

    Args:
        run_id:   'csi300' or 'csi500'
        is_start: IS start date 'YYYY-MM-DD'
        is_end:   IS end date   'YYYY-MM-DD'

    Returns:
        dict with keys:
          'ic_summary'  — factor × metric table
          'ic_decay'    — factor × horizon table
          'ic_series'   — date × factor IC time-series
    """
    assert run_id in ("csi300", "csi500"), "run_validation only supports single-index runs."
    index_code = "000300.SH" if run_id == "csi300" else "000905.SH"

    returns_df = pd.read_parquet(PROCESSED / "returns.parquet")
    factors    = load_all_factors()

    dates = pd.DatetimeIndex(
        [pd.Timestamp(d) for d in rebalance_dates(is_start, is_end)]
    )

    # Expand dates to all trading days in IS window for IC series
    all_dates = returns_df.loc[is_start:is_end].index

    def universe_fn(dt):
        return get_universe(dt, index_code)

    summary_rows = []
    decay_frames = {}
    ic_series_dict = {}

    for fname, fdf in factors.items():
        if fname in ("rvol_20d", "atr_14d"):
            continue   # volatility factors are not IC-validated as return predictors

        logger.info("[%s] Computing IC for factor: %s", run_id, fname)

        ic = compute_ic_series(fdf, returns_df, universe_fn, all_dates, horizon=1)
        ic_series_dict[fname] = ic

        summ = ic_summary(ic)
        summ["factor"] = fname
        summary_rows.append(summ)

        decay = ic_decay(fdf, returns_df, universe_fn, all_dates)
        decay_frames[fname] = decay

    summary_df = pd.DataFrame(summary_rows).set_index("factor")
    ic_series_df = pd.DataFrame(ic_series_dict)

    # Flatten decay into one table: factor | horizon | mean_ic | icir …
    decay_rows = []
    for fname, ddf in decay_frames.items():
        for h, row in ddf.iterrows():
            r = row.to_dict()
            r["factor"] = fname
            r["horizon"] = h
            decay_rows.append(r)
    decay_df = pd.DataFrame(decay_rows)

    # Persist
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(summary_df,  out_dir / "ic_summary.parquet")
    write_parquet(ic_series_df, out_dir / "ic_series.parquet")
    write_parquet(decay_df,    out_dir / "ic_decay.parquet")
    logger.info("[%s] Validation complete. Results saved to %s.", run_id, out_dir)

    return {"ic_summary": summary_df, "ic_decay": decay_df, "ic_series": ic_series_df}
