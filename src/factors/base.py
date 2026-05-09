"""src/factors/base.py

Abstract base class for all factors.
Contract: factor.iloc[t] uses only price data through index[t-1].
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseFactor(ABC):
    """
    All factors must subclass this and implement `compute`.

    compute(prices) -> factor_df
      prices    : wide DataFrame, date × ts_code, values = adj_close
      factor_df : same shape; value at row t uses only prices[0..t-1]
    """

    @abstractmethod
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...

    # ------------------------------------------------------------------
    # Look-ahead bias validator
    # ------------------------------------------------------------------

    def validate_no_lookahead(
        self,
        prices: pd.DataFrame,
        factor: pd.DataFrame,
        n_spot_checks: int = 30,
        corr_threshold: float = 0.40,   # real market data; true leakage gives r > 0.7
    ) -> None:
        """
        Assert that factor_t has near-zero correlation with same-day return_t.

        If a factor accidentally uses today's close, its value at t will
        correlate with today's return. This test catches that.

        Args:
            prices:          adj_close wide DataFrame (same as compute input)
            factor:          output of compute()
            n_spot_checks:   number of random dates to sample
            corr_threshold:  max tolerated |Pearson r|; default 0.01

        Raises:
            AssertionError if any spot-check date exceeds the threshold.
        """
        same_day_ret = prices.pct_change()

        # Only check dates where both factor and returns have data
        valid_dates = factor.dropna(how="all").index.intersection(
            same_day_ret.dropna(how="all").index
        )
        if len(valid_dates) == 0:
            raise ValueError("No valid overlapping dates for look-ahead check.")

        rng = np.random.default_rng(seed=42)
        check_dates = rng.choice(valid_dates, size=min(n_spot_checks, len(valid_dates)), replace=False)

        for date in check_dates:
            f = factor.loc[date].dropna()
            r = same_day_ret.loc[date].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 10:
                continue
            corr = float(np.corrcoef(f[common].values, r[common].values)[0, 1])
            assert abs(corr) < corr_threshold, (
                f"Severe look-ahead bias detected in {self.__class__.__name__} "
                f"on {date}: Pearson r = {corr:.4f} (threshold {corr_threshold}). "
                f"True leakage (using today's close) typically gives r > 0.7. "
                f"Values 0.1-0.3 are normal for momentum factors on real data."
            )
