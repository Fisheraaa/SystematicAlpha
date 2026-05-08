"""main.py — CLI entry point for the Systematic Alpha Framework.

Usage
-----
  python main.py --fetch               # pull all data from Tushare
  python main.py --process             # build wide price/return panels
  python main.py --factors             # compute all 6 factors
  python main.py --validate            # IC analysis + quantile backtest (IS)
  python main.py --backtest            # full IS+OOS backtest for all 3 runs
  python main.py --walkforward         # walk-forward validation
  python main.py --report              # generate report.html

  python main.py --all                 # run everything end-to-end
  python main.py --all --skip-fetch    # skip fetch (data already on disk)

All steps are idempotent: re-running skips already-completed work.
"""
from __future__ import annotations

import logging
import sys
import time

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _step(name: str):
    """Context manager: log step start/end and elapsed time."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        logger.info("=" * 60)
        logger.info("STEP: %s", name)
        logger.info("=" * 60)
        t0 = time.time()
        yield
        logger.info("DONE: %s  (%.1fs)", name, time.time() - t0)

    return _ctx()


@click.command()
@click.option("--fetch",       is_flag=True, help="Pull raw data from Tushare Pro.")
@click.option("--process",     is_flag=True, help="Build processed price/return panels.")
@click.option("--factors",     is_flag=True, help="Compute all 6 factors.")
@click.option("--validate",    is_flag=True, help="Run IC analysis and quantile backtest.")
@click.option("--backtest",    is_flag=True, help="Run IS + OOS backtest for all 3 runs.")
@click.option("--walkforward", is_flag=True, help="Run walk-forward validation.")
@click.option("--report",      is_flag=True, help="Generate report.html.")
@click.option("--all",         "run_all", is_flag=True, help="Run all steps end-to-end.")
@click.option("--skip-fetch",  is_flag=True, help="Skip --fetch even when --all is set.")
def main(fetch, process, factors, validate, backtest, walkforward, report, run_all, skip_fetch):

    if run_all:
        fetch       = not skip_fetch
        process     = True
        factors     = True
        validate    = True
        backtest    = True
        walkforward = True
        report      = True

    if not any([fetch, process, factors, validate, backtest, walkforward, report]):
        click.echo(__doc__)
        sys.exit(0)

    cfg = _load_cfg()

    # ── 1. Fetch ──────────────────────────────────────────────────────────────
    if fetch:
        with _step("Data Fetch (Tushare Pro)"):
            from src.data.fetcher import run_all_fetches
            run_all_fetches()

    # ── 2. Process ────────────────────────────────────────────────────────────
    if process:
        with _step("Data Processing (wide panels)"):
            from src.data.processor import build_panels
            build_panels()

    # ── 3. Factors ────────────────────────────────────────────────────────────
    if factors:
        with _step("Factor Computation"):
            from src.factors import build_all_factors
            build_all_factors()

    # ── 4. Validation ─────────────────────────────────────────────────────────
    if validate:
        is_start = cfg["walk_forward"]["is_start"]
        is_end   = cfg["walk_forward"]["is_end"]
        for run_id in ("csi300", "csi500"):
            with _step(f"Factor Validation [{run_id}]"):
                from src.validation.ic_analysis import run_validation
                results = run_validation(run_id, is_start, is_end)
                _print_ic_summary(run_id, results["ic_summary"])

            with _step(f"Quantile Analysis [{run_id}]"):
                from src.validation.quantile import run_quantile_analysis
                q_results = run_quantile_analysis(run_id, is_start, is_end)
                _print_quantile_summary(run_id, q_results)

    # ── 5. Backtest ───────────────────────────────────────────────────────────
    if backtest:
        is_start  = cfg["walk_forward"]["is_start"]
        oos_end   = cfg["walk_forward"]["oos_end"]

        for run_cfg in cfg["backtest"]["runs"]:
            run_id = run_cfg["run_id"]
            with _step(f"Backtest [{run_id}] {is_start} → {oos_end}"):
                from src.strategy.composite import build_composite_weights
                from src.backtest.engine import run_backtest

                ic_weights = build_composite_weights(run_id, is_start, cfg["walk_forward"]["is_end"])
                daily_df, trades_df = run_backtest(run_id, is_start, oos_end, ic_weights)
                _print_backtest_summary(run_id, daily_df, trades_df)

    # ── 6. Walk-forward ───────────────────────────────────────────────────────
    if walkforward:
        for run_cfg in cfg["backtest"]["runs"]:
            run_id = run_cfg["run_id"]
            with _step(f"Walk-Forward Validation [{run_id}]"):
                from src.backtest.walk_forward import run_walk_forward
                wf_df = run_walk_forward(run_id)
                if not wf_df.empty:
                    _print_wf_summary(run_id, wf_df)

    # ── 7. Report ─────────────────────────────────────────────────────────────
    if report:
        with _step("HTML Report Generation"):
            from src.report.generate import generate_report
            generate_report()
            logger.info("Open report.html in your browser.")

    logger.info("All requested steps complete.")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    from src.utils import load_config
    return load_config()


def _print_ic_summary(run_id: str, df) -> None:
    import pandas as pd
    if df is None or df.empty:
        return
    logger.info("\n[%s] IC Summary:\n%s", run_id, df.to_string())


def _print_quantile_summary(run_id: str, df) -> None:
    if df is None or df.empty:
        return
    logger.info("\n[%s] Quantile Returns:\n%s", run_id, df.to_string())


def _print_backtest_summary(run_id: str, daily_df, trades_df) -> None:
    import numpy as np
    from src.backtest.metrics import (
        annualised_return, sharpe_ratio, max_drawdown
    )
    eq = daily_df["equity"]
    ret = eq.pct_change().dropna()
    logger.info(
        "[%s] Final equity: %.0f CNY | Ann.Return: %.2f%% | Sharpe: %.2f | MaxDD: %.2f%%",
        run_id,
        eq.iloc[-1],
        annualised_return(eq) * 100,
        sharpe_ratio(ret),
        max_drawdown(eq) * 100,
    )


def _print_wf_summary(run_id: str, df) -> None:
    import numpy as np
    sr  = df["sharpe_ratio"].dropna()
    mdd = df["max_drawdown"].dropna()
    ar  = df["annual_return"].dropna()
    logger.info(
        "[%s] Walk-Forward: Sharpe μ=%.2f σ=%.2f | "
        "MaxDD μ=%.1f%% | AnnRet μ=%.1f%% | Hit rate=%.0f%%",
        run_id,
        sr.mean(), sr.std(),
        mdd.mean() * 100,
        ar.mean() * 100,
        (sr > 0).mean() * 100,
    )


if __name__ == "__main__":
    main()
