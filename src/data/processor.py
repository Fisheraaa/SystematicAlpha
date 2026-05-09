"""src/data/processor.py

Assembles per-stock Parquet files into wide panel DataFrames and applies
preprocessing: forward-fill, winsorisation, suspension flags.

Outputs (written to data/processed/):
  prices.parquet   — wide: date × ts_code, value = adj_close
  opens.parquet    — wide: date × ts_code, value = adj_open  (execution prices)
  highs.parquet    — wide: date × ts_code, value = adj_high
  lows.parquet     — wide: date × ts_code, value = adj_low
  volumes.parquet  — wide: date × ts_code, value = vol
  returns.parquet  — wide: date × ts_code, value = daily log-return (winsorised)
  suspended.parquet — wide bool: True when stock was suspended that day
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

from src.utils import (
    load_config, read_parquet, write_parquet, trading_calendar,
    RAW, PROCESSED,
)

logger = logging.getLogger(__name__)

WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


def _load_all_stocks() -> dict[str, pd.DataFrame]:
    """Load every per-stock Parquet file; return dict ts_code → DataFrame."""
    daily_dir = RAW / "daily"
    stock_dfs: dict[str, pd.DataFrame] = {}
    for f in sorted(daily_dir.glob("*.parquet")):
        ts_code = f.stem
        df = pd.read_parquet(f).set_index("trade_date")
        stock_dfs[ts_code] = df
    logger.info("Loaded %d stocks from disk.", len(stock_dfs))
    return stock_dfs


def _build_wide(stock_dfs: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """Stack a single column from all stocks into a wide date × ts_code table."""
    series = {code: df[column] for code, df in stock_dfs.items() if column in df.columns}
    wide = pd.DataFrame(series)
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    return wide


def _winsorise_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """Clip each row to [1%, 99%] across columns (cross-sectional winsorisation)."""
    def clip_row(row: pd.Series) -> pd.Series:
        lo = row.quantile(WINSOR_LOWER)
        hi = row.quantile(WINSOR_UPPER)
        return row.clip(lo, hi)
    return df.apply(clip_row, axis=1)


def build_panels() -> None:
    """Build and save all wide panel DataFrames. Idempotent."""
    dest_prices = PROCESSED / "prices.parquet"
    if dest_prices.exists():
        logger.info("Processed panels already exist — skipping processor.")
        return

    PROCESSED.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cal = trading_calendar()

    stock_dfs = _load_all_stocks()

    # ── Wide panels ──────────────────────────────────────────────────────────
    prices  = _build_wide(stock_dfs, "adj_close")
    opens   = _build_wide(stock_dfs, "adj_open")
    highs   = _build_wide(stock_dfs, "adj_high")
    lows    = _build_wide(stock_dfs, "adj_low")
    volumes = _build_wide(stock_dfs, "vol")

    # Reindex to trading calendar and forward-fill (handles non-trading dates
    # introduced by merging stocks with different listing windows).
    start = pd.Timestamp(cfg["data"]["start_date"])
    end   = pd.Timestamp(cfg["data"]["end_date"])
    idx   = cal[(cal >= start) & (cal <= end)]

    idx = idx.sort_values()                      # guarantee ascending order
    prices  = prices.reindex(idx).sort_index().ffill()
    opens   = opens.reindex(idx).sort_index().ffill()
    highs   = highs.reindex(idx).sort_index().ffill()
    lows    = lows.reindex(idx).sort_index().ffill()
    volumes = volumes.reindex(idx).sort_index().fillna(0)

    # ── Suspension flag ───────────────────────────────────────────────────────
    # A stock is considered suspended when volume == 0 for that trading day.
    suspended = (volumes == 0)

    # Additionally flag stocks with >5 consecutive suspension days.
    cfg_u = load_config()["universe"]
    max_susp = cfg_u["max_suspension_consecutive_days"]
    for col in suspended.columns:
        s = suspended[col]
        # Rolling sum: if sum of last (max_susp+1) days == (max_susp+1), mark True.
        consec = s.rolling(max_susp + 1).sum() == (max_susp + 1)
        suspended[col] = suspended[col] | consec

    # ── Daily log returns (winsorised) ────────────────────────────────────────
    log_rets = np.log(prices / prices.shift(1))
    log_rets = _winsorise_cross_section(log_rets)

    # ── Persist ───────────────────────────────────────────────────────────────
    write_parquet(prices,    PROCESSED / "prices.parquet")
    write_parquet(opens,     PROCESSED / "opens.parquet")
    write_parquet(highs,     PROCESSED / "highs.parquet")
    write_parquet(lows,      PROCESSED / "lows.parquet")
    write_parquet(volumes,   PROCESSED / "volumes.parquet")
    write_parquet(suspended, PROCESSED / "suspended.parquet")
    write_parquet(log_rets,  PROCESSED / "returns.parquet")

    logger.info(
        "Panels built: %d trading days × %d stocks.",
        len(idx), len(prices.columns),
    )
