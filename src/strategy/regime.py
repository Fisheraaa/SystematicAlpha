"""src/strategy/regime.py

Regime detection using CSI 300 index volatility percentile (primary)
and ADX trend strength (secondary).

Output: a daily pd.Series of regime labels:
    'momentum'     — low-vol, use momentum factors
    'transition'   — mid-vol, blend factors
    'reversion'    — high-vol + low-ADX, use mean-reversion factors
    'trend_reversion' — high-vol + high-ADX, retain some momentum

Factor weights are derived from the regime label in composite.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import load_config, PROCESSED


def _index_daily_return(prices: pd.DataFrame, index_code: str) -> pd.Series:
    """
    Approximate index daily return as the equal-weight mean of constituent
    adj_close changes. (We don't store index price series separately.)
    Uses the full price panel — this is valid because we're computing a
    market-level signal, not a per-stock signal.
    """
    # Use all stocks available; this approximation is fine for regime detection.
    return np.log(prices / prices.shift(1)).mean(axis=1)


def _compute_adx(
    index_ret: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Simplified ADX from an index return series.

    Since we don't have H/L for the index itself, we use the daily return
    magnitude as a proxy for directional movement:
      DM+ = max(ret, 0),  DM- = max(-ret, 0)
      ADX = EWM(|DM+ - DM-| / (DM+ + DM-))

    This is a structural approximation; the threshold (25) is calibrated
    to this definition.
    """
    alpha = 1.0 / period
    dm_plus  = index_ret.clip(lower=0)
    dm_minus = (-index_ret).clip(lower=0)

    adx_raw = (dm_plus - dm_minus).abs() / (dm_plus + dm_minus).replace(0, np.nan)
    adx = adx_raw.ewm(alpha=alpha, min_periods=period, adjust=False).mean() * 100
    return adx.shift(1)   # value at t uses data through t-1


def build_regime_series(prices: pd.DataFrame | None = None) -> pd.Series:
    """
    Compute daily regime labels over the full date range.

    Args:
        prices: adj_close wide DataFrame. Loaded from disk if None.

    Returns:
        pd.Series indexed by date, values ∈ {'momentum', 'transition',
        'reversion', 'trend_reversion'}.
    """
    cfg_r = load_config()["regime"]
    if prices is None:
        prices = pd.read_parquet(PROCESSED / "prices.parquet")

    # ── Index-level return proxy ───────────────────────────────────────────
    idx_ret = _index_daily_return(prices, "000300.SH")

    # ── Realised vol (20-day, annualised) of index return ─────────────────
    rvol_idx = (
        idx_ret.rolling(20, min_periods=10).std().shift(1) * np.sqrt(252)
    )

    # ── Volatility percentile (252-day rolling) ────────────────────────────
    vol_window = cfg_r["vol_percentile_window"]
    vol_pct = rvol_idx.rolling(vol_window, min_periods=vol_window // 2).rank(pct=True)

    # ── ADX ────────────────────────────────────────────────────────────────
    adx = _compute_adx(idx_ret, period=cfg_r["adx_period"])

    lo = cfg_r["low_vol_threshold"]
    hi = cfg_r["high_vol_threshold"]
    adx_thresh = cfg_r["adx_trending_threshold"]

    def _label(row) -> str:
        vp, ax = row["vol_pct"], row["adx"]
        if np.isnan(vp):
            return "momentum"           # default when insufficient history
        if vp <= lo:
            return "momentum"
        elif vp <= hi:
            return "transition"
        else:
            # High vol: distinguish directional vs. choppy
            if not np.isnan(ax) and ax > adx_thresh:
                return "trend_reversion"  # high-vol but trending
            return "reversion"

    combined = pd.DataFrame({"vol_pct": vol_pct, "adx": adx})
    regime = combined.apply(_label, axis=1)
    regime.name = "regime"
    return regime


# Pre-compute factor weight vectors for each regime
# Format: {factor_name: weight}   (weights sum to 1.0)

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "momentum": {
        "mom_5d":     0.60,
        "mom_20d":    0.40,
        "zscore_20d": 0.00,
        "rsi_14d":    0.00,
    },
    "transition": {      # filled dynamically via interpolation in composite.py
        "mom_5d":     0.30,
        "mom_20d":    0.20,
        "zscore_20d": 0.25,
        "rsi_14d":    0.25,
    },
    "reversion": {
        "mom_5d":     0.00,
        "mom_20d":    0.00,
        "zscore_20d": 0.50,
        "rsi_14d":    0.50,
    },
    "trend_reversion": {
        "mom_5d":     0.15,
        "mom_20d":    0.15,
        "zscore_20d": 0.35,
        "rsi_14d":    0.35,
    },
}
