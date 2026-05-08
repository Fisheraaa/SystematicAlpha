"""src/strategy/portfolio.py

Portfolio construction: selects top-N stocks and computes Risk Parity weights.

Steps:
  1. Rank stocks by composite signal.
  2. Select top N (regime-dependent: 40 in low-vol, 20 in high-vol).
  3. Compute inverse-volatility (Risk Parity) weights using RVol20d.
  4. Apply per-stock position bounds [w_min, w_max].
  5. Scale to invested fraction (1 - cash_reserve).
  6. Return target weight Series indexed by ts_code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import load_config, PROCESSED


def _rvol_at(date: pd.Timestamp, universe: list[str]) -> pd.Series:
    """Load RVol20d for the given universe at date."""
    rvol_df = pd.read_parquet(PROCESSED / "factors" / "rvol_20d.parquet")
    if date not in rvol_df.index:
        return pd.Series(dtype=float)
    return rvol_df.loc[date].reindex(universe).dropna()


def _n_holdings(regime: str, cfg_s: dict) -> int:
    """Number of stocks to hold based on regime."""
    if regime in ("reversion", "trend_reversion"):
        return cfg_s["min_holdings"]        # 20 — more defensive in high-vol
    return cfg_s["max_holdings"]            # 40 — more aggressive in low-vol


def compute_target_weights(
    date: pd.Timestamp,
    composite: pd.Series,
    regime: str,
    drawdown: float = 0.0,
) -> pd.Series:
    """
    Compute target portfolio weights for one rebalance date.

    Args:
        date:      rebalance date (used to load RVol)
        composite: ts_code → composite signal score (higher = more attractive)
        regime:    current regime label
        drawdown:  current portfolio drawdown (0–1); triggers position scaling

    Returns:
        pd.Series ts_code → target weight (sum ≤ 1 − cash_reserve).
    """
    cfg = load_config()
    cfg_s = cfg["strategy"]
    cfg_r = cfg["risk"]

    if composite.empty:
        return pd.Series(dtype=float)

    n = _n_holdings(regime, cfg_s)
    # Select top-N by composite score
    selected = composite.nlargest(n).index.tolist()

    # Inverse-volatility weights
    rvol = _rvol_at(date, selected)
    if rvol.empty or rvol.isna().all():
        # Fallback to equal weight
        weights = pd.Series(1.0 / len(selected), index=selected)
    else:
        inv_vol = 1.0 / rvol.replace(0, np.nan).fillna(rvol.mean())
        weights = inv_vol / inv_vol.sum()

    # Position bounds
    w_min = cfg_s["min_position_weight"]
    w_max = cfg_s["max_position_weight"]
    weights = weights.clip(w_min, w_max)
    weights = weights / weights.sum()       # re-normalise after clamping

    # Invested fraction: start at (1 - cash_reserve_min), reduce for drawdown
    invested = 1.0 - cfg_s["cash_reserve_min"]
    dd = drawdown
    if dd >= cfg_r["drawdown_extreme"]:
        invested = min(invested, 0.20)
    elif dd >= cfg_r["drawdown_hard_limit"]:
        invested = min(invested, 0.50)
    elif dd >= cfg_r["drawdown_soft_limit"]:
        invested = min(invested, 0.70)

    weights = weights * invested
    return weights
