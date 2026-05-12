"""src/backtest/walk_forward.py

Walk-forward validation with resume support and faster IC validation.

Resume logic: if wf_window_XX.parquet already exists for a window, that
window is skipped entirely and its saved results are loaded. This means
you can Ctrl-C at any point and re-run without losing completed windows.

IC validation is also skipped per-sub-index if wf_val_XX/ already exists
and contains a valid ic_summary.parquet.
"""
from __future__ import annotations

import logging
from dateutil.relativedelta import relativedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, write_parquet, read_parquet, RESULTS, PROCESSED
from src.validation.ic_analysis import run_validation
from src.strategy.composite import build_composite_weights
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def _add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    return dt + relativedelta(months=months)


def _benchmark_equity(run_id: str, start: str, end: str) -> pd.Series:
    returns = pd.read_parquet(PROCESSED / "returns.parquet").loc[start:end]
    idx_ret = returns.mean(axis=1)
    return np.exp(idx_ret.cumsum())


def _window_dates(cfg_wf: dict) -> list[dict]:
    oos_start = pd.Timestamp(cfg_wf["oos_start"])
    oos_end   = pd.Timestamp(cfg_wf["oos_end"])
    train_m   = cfg_wf["train_months"]
    test_m    = cfg_wf["test_months"]
    step_m    = cfg_wf["step_months"]

    windows = []
    test_start = oos_start
    window_id  = 1

    while True:
        test_end    = _add_months(test_start, test_m) - pd.Timedelta(days=1)
        train_end   = test_start - pd.Timedelta(days=1)
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


EXPECTED_FACTORS = {"mom_5d", "mom_20d", "zscore_20d", "rsi_14d"}

def _val_dir_complete(val_dir: Path) -> bool:
    """True only if ic_summary.parquet exists with all 4 expected factors."""
    summary = val_dir / "ic_summary.parquet"
    if not summary.exists():
        return False
    try:
        df = pd.read_parquet(summary)
        return EXPECTED_FACTORS.issubset(set(df.index))
    except Exception:
        return False


def _window_result_exists(run_id: str, wid: int, w: dict) -> tuple[bool, dict]:
    """
    Check if this window's backtest result exists and compute metrics from it.
    Returns (exists, metrics_dict).
    """
    daily_path = RESULTS / run_id / f"wf_window_{wid:02d}.parquet"
    if not daily_path.exists():
        return False, {}
    try:
        daily_df = pd.read_parquet(daily_path)
        if daily_df.empty or len(daily_df) < 2:
            return False, {}
        equity    = daily_df["equity"]
        benchmark = _benchmark_equity(run_id, w["test_start"], w["test_end"]) * equity.iloc[0]
        trades_path = RESULTS / run_id / f"wf_window_{wid:02d}_trades.parquet"
        trades_df   = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()
        metrics = compute_all_metrics(equity, benchmark, trades_df)
        return True, metrics
    except Exception as exc:
        logger.warning("Could not load window %d result: %s", wid, exc)
        return False, {}


def run_walk_forward(run_id: str) -> pd.DataFrame:
    """
    Execute walk-forward validation for a given run_id.

    Resume-safe: completed windows (wf_window_XX.parquet present) are
    skipped automatically. Re-running after a Ctrl-C continues from where
    it left off.
    """
    cfg    = load_config()
    cfg_wf = cfg["walk_forward"]
    windows = _window_dates(cfg_wf)

    # Count already-completed windows for the log message
    already_done = sum(
        1 for w in windows
        if _window_result_exists(run_id, w["window_id"], w)[0]
    )
    remaining = len(windows) - already_done
    logger.info(
        "[%s] Walk-forward: %d windows total, %d already done, %d to run.",
        run_id, len(windows), already_done, remaining,
    )

    rows = []

    for w in windows:
        wid  = w["window_id"]
        t_s  = w["train_start"]
        t_e  = w["train_end"]
        te_s = w["test_start"]
        te_e = w["test_end"]

        # ── Resume: load existing result and skip ─────────────────────────
        exists, saved_row = _window_result_exists(run_id, wid, w)
        if exists:
            logger.info(
                "[%s] Window %d/%d: SKIPPING (already done — Sharpe=%.3f)",
                run_id, wid, len(windows),
                saved_row.get("sharpe_ratio", float("nan")),
            )
            rows.append({**w, **saved_row})
            continue

        logger.info(
            "[%s] Window %d/%d: train %s→%s, test %s→%s",
            run_id, wid, len(windows), t_s, t_e, te_s, te_e,
        )

        # ── Step 1: IC validation (skip sub-index if already done) ────────
        wf_val_dir_300 = RESULTS / "csi300" / f"wf_val_{wid:02d}"
        wf_val_dir_500 = RESULTS / "csi500" / f"wf_val_{wid:02d}"

        if run_id == "combined":
            for sub_id, val_dir in (("csi300", wf_val_dir_300), ("csi500", wf_val_dir_500)):
                if _val_dir_complete(val_dir):
                    logger.info("[%s] Window %d: %s IC already cached — skipping.", run_id, wid, sub_id)
                else:
                    try:
                        run_validation(sub_id, t_s, t_e, output_dir=val_dir)
                    except Exception as exc:
                        logger.warning("Validation failed for %s window %d: %s", sub_id, wid, exc)
            ic_weights = build_composite_weights("combined", t_s, t_e)
        else:
            val_dir = wf_val_dir_300 if run_id == "csi300" else wf_val_dir_500
            if _val_dir_complete(val_dir):
                logger.info("[%s] Window %d: IC already cached — skipping validation.", run_id, wid)
            else:
                try:
                    run_validation(run_id, t_s, t_e, output_dir=val_dir)
                except Exception as exc:
                    logger.warning("Validation failed window %d: %s", wid, exc)
            ic_weights = build_composite_weights(run_id, t_s, t_e)

        # ── Step 2: backtest ──────────────────────────────────────────────
        wf_stem = f"wf_window_{wid:02d}"
        try:
            daily_df, trades_df = run_backtest(
                run_id, te_s, te_e, ic_weights, output_stem=wf_stem
            )
        except Exception as exc:
            logger.error("Backtest failed window %d: %s", wid, exc)
            continue

        equity    = daily_df["equity"]
        benchmark = _benchmark_equity(run_id, te_s, te_e) * equity.iloc[0]
        metrics   = compute_all_metrics(equity, benchmark, trades_df)
        row       = {**w, **metrics}
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

    numeric = result_df.select_dtypes(include="number").drop(columns=["window_id"], errors="ignore")
    summary = numeric.agg(["mean", "std", "min", "max"])
    logger.info(
        "[%s] Walk-forward summary:\n%s",
        run_id,
        summary[["sharpe_ratio", "max_drawdown", "annual_return"]].to_string(),
    )

    return result_df
