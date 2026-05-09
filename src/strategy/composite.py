"""src/strategy/composite.py

Builds the composite signal from per-factor scores, IC weights, and regime.

Steps at each rebalance date t:
  1. Load factor values for the eligible universe.
  2. Cross-sectionally rank-normalise each factor → scores in (0, 1).
  3. Look up the regime at t → get regime-specific factor weights.
  4. Scale regime weights by IC weights (information-weighted).
  5. Compute composite = weighted sum of normalised factor scores.
  6. Return composite score series indexed by ts_code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.regime import REGIME_WEIGHTS
from src.utils import RESULTS


def _rank_normalise(s: pd.Series) -> pd.Series:
    r = s.rank(method="average", na_option="keep")
    n = r.notna().sum()
    return (r - 0.5) / n if n > 0 else r


def _load_ic_weights(run_id: str) -> dict[str, float]:
    """
    Load per-factor IC means from the IS validation results and
    convert to non-negative weights proportional to |mean_IC|.

    Falls back to equal weights if validation results don't exist yet.
    """
    result_path = RESULTS / run_id / "ic_summary.parquet"
    if not result_path.exists():
        # Equal weights for the 4 predictive factors
        return {f: 0.25 for f in ("mom_5d", "mom_20d", "zscore_20d", "rsi_14d")}

    summary = pd.read_parquet(result_path)
    ic_signed = summary["mean_ic"]
    ic_abs    = ic_signed.abs()
    total     = ic_abs.sum()
    if total == 0:
        return {f: 0.25 for f in ic_signed.index}
    # weight ∝ |IC|
    weights = ic_abs / total * ic_signed.apply(lambda x: 1.0 if x >= 0 else -1.0)
    return weights.to_dict()


def compute_composite(
    date: pd.Timestamp,
    universe: list[str],
    factors: dict[str, pd.DataFrame],
    regime: str,
    ic_weights: dict[str, float],
) -> pd.Series:
    """
    Compute composite signal for one rebalance date.

    Args:
        date:       rebalance date
        universe:   eligible ts_codes
        factors:    {factor_name: wide DataFrame}
        regime:     regime label for this date
        ic_weights: {factor_name: IC-derived weight}

    Returns:
        pd.Series (ts_code → composite score), higher = more attractive.
    """
    regime_w = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["momentum"])

    scores: dict[str, pd.Series] = {}
    for fname, w_regime in regime_w.items():
        if w_regime == 0:
            continue
        if fname not in factors or date not in factors[fname].index:
            continue

        f = factors[fname].loc[date].reindex(universe).dropna()
        if len(f) < 10:
            continue

        norm = _rank_normalise(f)
        w_ic = ic_weights.get(fname, 0.25)
        # Combined weight = regime weight × IC weight (then re-normalised below)
        scores[fname] = norm * (w_regime * w_ic)

    if not scores:
        return pd.Series(dtype=float)

    composite = pd.concat(scores.values(), axis=1).sum(axis=1)
    # Re-normalise so composite is on (0, 1) scale
    rng = composite.max() - composite.min()
    if rng > 0:
        composite = (composite - composite.min()) / rng
    return composite.reindex(universe)


def build_composite_weights(run_id: str, is_start: str, is_end: str) -> dict[str, float]:
    """Load IC weights calibrated on the IS period for a given run_id."""
    # For combined run, blend 40/60
    if run_id == "combined":
        w300 = _load_ic_weights("csi300")
        w500 = _load_ic_weights("csi500")
        blended: dict[str, float] = {}
        for fname in set(w300) | set(w500):
            blended[fname] = 0.40 * w300.get(fname, 0) + 0.60 * w500.get(fname, 0)
        return blended
    return _load_ic_weights(run_id)
