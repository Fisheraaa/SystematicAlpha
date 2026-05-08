"""src/backtest/walk_forward.py

Walk-forward validation: slides a training window across time, re-calibrates
IC weights on each training fold, runs the backtest on the held-out test fold,
and aggregates OOS performance statistics.

Configuration (from config.yaml):
    train_months: 24
    test_months:   6
    step_months:   3
    oos_start: '2021-01-01'
    oos_end:   '2024-12-31'

For 4 OOS years with step=3 months → 13 rolling windows.

Public API
----------
run_walk_forward(run_id) -> pd.DataFrame
    Returns a DataFrame with one row per test window:
    window_id, train_start, train_end, test_start, test_end,
    sharpe, max_drawdown, annual_return, calmar, ir, win_rate, turnover.
"""
from __future__ import annotations

import logging
from dateutil.relativedelta import relativedelta

import pandas as pd

from src.utils import load_config, write_parquet, RESULTS
from src.validation.ic_analysis import run_validation
from src.strategy.composite import build_composite_weights
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def _add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    return dt + relativedelta(months=months)


def _benchmark_equity(run_id: str, start: str, end: str) -> pd.Series:
    """
    Build a benchmark equity series (base 1.0) for the given period.
    Uses the equal-weight average return of all stocks in the universe
    as a proxy — same approximation used in regime detection.

    For a production system, load actual CSI 300 / CSI 500 TR index prices.
    """
    from src.utils import PROCESSED
    import numpy as np
    returns = pd.read_parquet(PROCESSED / "returns.parquet").loc[start:end]
    idx_ret = returns.mean(axis=1)
    return np.exp(idx_ret.cumsum())


def _window_dates(cfg_wf: dict) -> list[dict]:
    """Generate all train/test window date ranges."""
    oos_start = pd.Timestamp(cfg_wf["oos_start"])
    oos_end   = pd.Timestamp(cfg_wf["oos_end"])
    train_m   = cfg_wf["train_months"]
    test_m    = cfg_wf["test_months"]
    step_m    = cfg_wf["step_months"]

    windows = []
    test_start = oos_start
    window_id  = 1

    while True:
        test_end   = _add_months(test_start, test_m) - pd.Timedelta(days=1)
        train_end  = test_start - pd.Timedelta(days=1)
        train_start = _add_months(train_end, -train_m) + pd.Timedelta(days=1)

        if test_end > oos_end:
            break

        windows.append({
            "window_id":   window_id,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end":   train_end.strftime("%Y-%m-%d"),
            "test_start":  test_start.strftime("%Y-%m-%d"),
            "test_end":    test_end.strftime("%Y-%m-%d"),
        })

        test_start = _add_months(test_start, step_m)
        window_id += 1

    return windows


def run_walk_forward(run_id: str) -> pd.DataFrame:
    """
    Execute walk-forward validation for a given run_id.

    For each window:
      1. Run IC validation on the training fold to get IC weights.
      2. Run backtest on the test fold with those weights.
      3. Record performance metrics.

    Results are persisted to data/results/{run_id}/walk_forward.parquet.
    """
    cfg     = load_config()
    cfg_wf  = cfg["walk_forward"]
    windows = _window_dates(cfg_wf)

    logger.info("[%s] Starting walk-forward: %d windows.", run_id, len(windows))

    rows = []

    for w in windows:
        wid   = w["window_id"]
        t_s   = w["train_start"]
        t_e   = w["train_end"]
        te_s  = w["test_start"]
        te_e  = w["test_end"]

        logger.info(
            "[%s] Window %d/%d: train %s→%s, test %s→%s",
            run_id, wid, len(windows), t_s, t_e, te_s, te_e,
        )

        # ── Step 1: calibrate IC weights on training fold ──────────────────
        if run_id == "combined":
            # Calibrate on each sub-universe; blend 40/60 in build_composite_weights
            for sub_id in ("csi300", "csi500"):
                try:
                    run_validation(sub_id, t_s, t_e)
                except Exception as exc:
                    logger.warning("Validation failed for %s window %d: %s", sub_id, wid, exc)
            ic_weights = build_composite_weights("combined", t_s, t_e)
        else:
            try:
                run_validation(run_id, t_s, t_e)
            except Exception as exc:
                logger.warning("Validation failed window %d: %s", wid, exc)
            ic_weights = build_composite_weights(run_id, t_s, t_e)

        # ── Step 2: backtest on test fold ─────────────────────────────────
        try:
            daily_df, trades_df = run_backtest(run_id, te_s, te_e, ic_weights)
        except Exception as exc:
            logger.error("Backtest failed window %d: %s", wid, exc)
            continue

        equity = daily_df["equity"]
        benchmark = _benchmark_equity(run_id, te_s, te_e) * equity.iloc[0]

        metrics = compute_all_metrics(equity, benchmark, trades_df)
        row = {**w, **metrics}
        rows.append(row)

        logger.info(
            "[%s] Window %d done: Sharpe=%.3f, MaxDD=%.2f%%, AnnRet=%.2f%%",
            run_id, wid,
            metrics.get("sharpe_ratio", float("nan")),
            metrics.get("max_drawdown", float("nan")) * 100,
            metrics.get("annual_return", float("nan")) * 100,
        )

    if not rows:
        logger.error("[%s] Walk-forward produced no results.", run_id)
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(result_df, out_dir / "walk_forward.parquet")

    # Summary statistics
    numeric = result_df.select_dtypes(include="number").drop(
        columns=["window_id"], errors="ignore"
    )
    summary = numeric.agg(["mean", "std", "min", "max"])
    logger.info(
        "[%s] Walk-forward summary:\n%s",
        run_id,
        summary[["sharpe_ratio", "max_drawdown", "annual_return"]].to_string(),
    )

    return result_df
