"""src/factors/momentum.py — Mom5d and Mom20d."""
from __future__ import annotations

import pandas as pd
from src.factors.base import BaseFactor


class Mom5d(BaseFactor):
    """
    5-day price return, skipping the most recent day.

    Formula: Mom5d_t = P_{t-1} / P_{t-6} - 1

    The skip-1 design avoids the well-documented short-term reversal in A-shares:
    including the most recent day's return would cause the factor to partially
    predict reversals rather than continuation.

    Implementation: pct_change(5) computes P_{t}/P_{t-5} - 1; .shift(1) moves
    the result one day forward so the value at t uses prices only through t-1.
    """

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return prices.pct_change(5).shift(1)


class Mom20d(BaseFactor):
    """
    20-day price return, skipping the two most recent days.

    Formula: Mom20d_t = P_{t-2} / P_{t-22} - 1

    A-share literature shows a stronger short-term reversal than US markets;
    skipping two days rather than one yields more stable IC over the 20-day window.
    """

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return prices.pct_change(20).shift(2)
