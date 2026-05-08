"""src/data/fetcher.py

Pulls all required raw data from Tushare Pro and persists to Parquet.
Design: idempotent — re-running skips files that already exist.
Call sequence: fetch_calendar → fetch_index_weights → fetch_daily_bars.
"""
from __future__ import annotations

import time
import logging
import pathlib

import pandas as pd
import tushare as ts

from src.utils import load_config, write_parquet, RAW, CACHE

logger = logging.getLogger(__name__)


def _pro() -> ts.pro.client.DataApi:
    cfg = load_config()
    ts.set_token(cfg["data"]["token"])
    return ts.pro_api()


def _rate_limited_call(func, *args, retries: int = 5, **kwargs) -> pd.DataFrame:
    """Wraps a Tushare API call with exponential back-off on rate-limit errors."""
    for attempt in range(retries):
        try:
            result = func(*args, **kwargs)
            if result is not None and not result.empty:
                return result
            time.sleep(0.4)
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("API error (%s), retrying in %ds…", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Tushare call failed after {retries} attempts: {func.__name__}")


# ---------------------------------------------------------------------------
# 1. Trading calendar
# ---------------------------------------------------------------------------

def fetch_calendar() -> None:
    """Fetch A-share trading calendar (2010–2025) and save to cache."""
    dest = CACHE / "trading_calendar.parquet"
    if dest.exists():
        logger.info("Calendar already cached — skipping.")
        return

    pro = _pro()
    df = _rate_limited_call(
        pro.trade_cal, exchange="SSE", start_date="20100101", end_date="20251231"
    )
    df["cal_date"] = pd.to_datetime(df["cal_date"])
    write_parquet(df[["cal_date", "is_open"]], dest)
    logger.info("Calendar saved (%d rows).", len(df))


# ---------------------------------------------------------------------------
# 2. Index constituent weights (point-in-time snapshots)
# ---------------------------------------------------------------------------

def fetch_index_weights() -> None:
    """
    Fetch historical constituent weights for CSI 300 and CSI 500.
    Saves to data/raw/index_weight/{index_code}.parquet.
    Each row: (trade_date, con_code, i_weight).
    """
    cfg = load_config()
    pro = _pro()

    for index_code in cfg["data"]["indices"]:
        dest = RAW / "index_weight" / f"{index_code}.parquet"
        if dest.exists():
            logger.info("Index weights for %s already cached — skipping.", index_code)
            continue

        chunks = []
        # Tushare limits single calls to ~5000 rows; iterate by year.
        for year in range(2014, 2025):
            start = f"{year}0101"
            end   = f"{year}1231"
            chunk = _rate_limited_call(
                pro.index_weight,
                index_code=index_code,
                start_date=start,
                end_date=end,
            )
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
            time.sleep(0.5)

        if not chunks:
            logger.error("No data returned for %s.", index_code)
            continue

        df = pd.concat(chunks, ignore_index=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["trade_date", "con_code"]).drop_duplicates()
        write_parquet(df[["trade_date", "con_code", "i_weight"]], dest)
        logger.info("Index weights for %s saved (%d rows).", index_code, len(df))


# ---------------------------------------------------------------------------
# 3. Daily OHLCV bars + adjustment factors
# ---------------------------------------------------------------------------

def _all_ts_codes() -> list[str]:
    """Return the union of all ts_codes that ever appeared in either index."""
    cfg = load_config()
    codes: set[str] = set()
    for index_code in cfg["data"]["indices"]:
        path = RAW / "index_weight" / f"{index_code}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run fetch_index_weights() first."
            )
        df = pd.read_parquet(path)
        codes.update(df["con_code"].unique())
    return sorted(codes)


def fetch_daily_bars() -> None:
    """
    For each stock that ever appeared in CSI 300 or CSI 500, fetch:
      - unadjusted daily OHLCV  (pro.daily)
      - adjustment factor       (pro.adj_factor)
    Merge into a single file: data/raw/daily/{ts_code}.parquet.
    Columns: trade_date, open, high, low, close, vol, adj_factor, adj_close.
    """
    cfg = load_config()
    start = cfg["data"]["start_date"].replace("-", "")
    end   = cfg["data"]["end_date"].replace("-", "")
    pro   = _pro()

    codes = _all_ts_codes()
    logger.info("Fetching daily bars for %d stocks.", len(codes))

    dest_dir = RAW / "daily"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for ts_code in codes:
        dest = dest_dir / f"{ts_code}.parquet"
        if dest.exists():
            continue

        try:
            ohlcv = _rate_limited_call(
                pro.daily, ts_code=ts_code, start_date=start, end_date=end
            )
            adj = _rate_limited_call(
                pro.adj_factor, ts_code=ts_code, start_date=start, end_date=end
            )
            time.sleep(0.3)
        except RuntimeError as exc:
            logger.error("Skipping %s: %s", ts_code, exc)
            continue

        if ohlcv is None or ohlcv.empty:
            logger.warning("No OHLCV data for %s.", ts_code)
            continue

        ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])

        if adj is not None and not adj.empty:
            adj["trade_date"] = pd.to_datetime(adj["trade_date"])
            merged = ohlcv.merge(
                adj[["trade_date", "adj_factor"]], on="trade_date", how="left"
            )
        else:
            merged = ohlcv.copy()
            merged["adj_factor"] = 1.0

        merged["adj_factor"] = merged["adj_factor"].fillna(1.0)

        # Backward-adjusted (前复权): normalise so latest adj_factor = 1.
        latest_af = merged.sort_values("trade_date")["adj_factor"].iloc[-1]
        merged["adj_close"] = merged["close"] * merged["adj_factor"] / latest_af
        merged["adj_open"]  = merged["open"]  * merged["adj_factor"] / latest_af
        merged["adj_high"]  = merged["high"]  * merged["adj_factor"] / latest_af
        merged["adj_low"]   = merged["low"]   * merged["adj_factor"] / latest_af

        cols = [
            "trade_date", "ts_code",
            "open", "high", "low", "close", "vol",
            "adj_factor", "adj_close", "adj_open", "adj_high", "adj_low",
        ]
        merged = merged[[c for c in cols if c in merged.columns]]
        merged = merged.sort_values("trade_date").reset_index(drop=True)

        write_parquet(merged, dest)
        logger.debug("Saved %s (%d rows).", ts_code, len(merged))

    logger.info("Daily bar fetch complete.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_fetches() -> None:
    fetch_calendar()
    fetch_index_weights()
    fetch_daily_bars()
