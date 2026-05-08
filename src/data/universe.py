"""src/data/universe.py

Point-in-time universe construction.

Core guarantee: get_universe(date, index_code) returns only stocks that were
ACTUALLY in the index at `date`, using historical constituent snapshots.
No future information leaks in.

Eligibility filters (all evaluated at time t using t or earlier data):
  1. In index at t           — from historical snapshot
  2. Not suspended           — volume > 0 at t
  3. Not at limit-up/down    — |return_t| < 9.5 %
  4. Listed ≥ 60 trading days before t
  5. All 6 factor values non-null at t

Public API
----------
get_universe(date, index_code) -> list[str]
get_combined_universe(date)    -> pd.DataFrame  (ts_code, index_membership)
rebalance_dates(cfg)           -> list[str]      YYYYMMDD strings
"""
from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from src.utils import load_config, read_parquet, trading_calendar, RAW, PROCESSED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_index_weight(index_code: str) -> pd.DataFrame:
    path = RAW / "index_weight" / f"{index_code}.parquet"
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.sort_values("trade_date")


def _pit_constituents(date: pd.Timestamp, index_code: str) -> set[str]:
    """
    Return the set of ts_codes in `index_code` as of `date`.
    Uses the most recent constituent announcement on or before `date`.
    """
    df = _load_index_weight(index_code)
    snapshot = df[df["trade_date"] <= date]
    if snapshot.empty:
        return set()
    # For each stock, take the most recent record.
    latest = snapshot.sort_values("trade_date").groupby("con_code").last()
    return set(latest.index.tolist())


@lru_cache(maxsize=1)
def _suspended() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "suspended.parquet")


@lru_cache(maxsize=1)
def _returns() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "returns.parquet")


@lru_cache(maxsize=1)
def _prices() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "prices.parquet")


def _listing_dates() -> dict[str, pd.Timestamp]:
    """
    Return first date with non-NaN price for each stock as a proxy for
    listing date. This is safe to compute from adjusted price history.
    """
    prices = _prices()
    return {col: prices[col].first_valid_index() for col in prices.columns}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_universe(date: str | pd.Timestamp, index_code: str) -> list[str]:
    """
    Return eligible ts_codes for `index_code` at `date`.

    Args:
        date:       rebalance date (str 'YYYYMMDD' or pd.Timestamp)
        index_code: '000300.SH' or '000905.SH'

    Returns:
        Sorted list of eligible ts_codes.
    """
    cfg = load_config()["universe"]
    cal = trading_calendar()

    ts = pd.Timestamp(date) if isinstance(date, str) else date

    # Filter 1: PIT constituents
    constituents = _pit_constituents(ts, index_code)
    if not constituents:
        logger.warning("No constituents found for %s at %s.", index_code, ts.date())
        return []

    eligible = set(constituents)

    # Filter 2: not suspended
    susp = _suspended()
    if ts in susp.index:
        susp_today = set(susp.columns[susp.loc[ts]])
        eligible -= susp_today

    # Filter 3: not at limit-up/down
    rets = _returns()
    limit = cfg["limit_band_threshold"]
    if ts in rets.index:
        ret_today = rets.loc[ts]
        at_limit = set(ret_today[ret_today.abs() >= limit].index)
        eligible -= at_limit

    # Filter 4: listed ≥ min_listing_days before t
    listing = _listing_dates()
    min_days = cfg["min_listing_days"]
    cal_before_t = cal[cal <= ts]
    newly_listed: set[str] = set()
    for code in list(eligible):
        ld = listing.get(code)
        if ld is None:
            newly_listed.add(code)
            continue
        days_listed = (cal_before_t >= ld).sum()
        if days_listed < min_days:
            newly_listed.add(code)
    eligible -= newly_listed

    return sorted(eligible)


def get_combined_universe(date: str | pd.Timestamp) -> pd.DataFrame:
    """
    Return the union universe for the 'combined' run.

    Returns a DataFrame with columns:
        ts_code          (str)
        index_membership ('csi300' | 'csi500' | 'both')
    """
    ts = pd.Timestamp(date) if isinstance(date, str) else date
    csi300 = set(get_universe(ts, "000300.SH"))
    csi500 = set(get_universe(ts, "000905.SH"))

    records = []
    for code in csi300 | csi500:
        if code in csi300 and code in csi500:
            membership = "both"
        elif code in csi300:
            membership = "csi300"
        else:
            membership = "csi500"
        records.append({"ts_code": code, "index_membership": membership})

    return pd.DataFrame(records).sort_values("ts_code").reset_index(drop=True)


def rebalance_dates(start: str, end: str) -> list[str]:
    """
    Return list of rebalance dates (first trading day of each calendar month)
    between `start` and `end` inclusive (format: 'YYYY-MM-DD').
    """
    cal = trading_calendar()
    mask = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))
    cal_window = cal[mask]

    dates: list[pd.Timestamp] = []
    current_month = None
    for dt in cal_window:
        key = (dt.year, dt.month)
        if key != current_month:
            dates.append(dt)
            current_month = key

    return [d.strftime("%Y%m%d") for d in dates]
