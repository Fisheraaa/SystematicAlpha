"""src/factors/reversion.py — ZScore20d and RSI14d."""
from __future__ import annotations

import pandas as pd
import numpy as np
from src.factors.base import BaseFactor


class ZScore20d(BaseFactor):
    """
    20-day cross-sectional Z-score of price relative to its rolling mean.

    Formula: Z_t = (P_{t-1} - μ_{[t-21, t-1]}) / σ_{[t-21, t-1]}

    A large positive Z indicates the stock is extended above its recent average
    and is a mean-reversion SHORT signal (high Z → low expected return).
    Winsorised at ±clip to limit influence of corporate-event outliers.

    Look-ahead control: prices.shift(1) moves the entire series one day so that
    the rolling window never includes today's price; closed='left' is redundant
    here but added as an explicit safety net.
    """

    def __init__(self, window: int = 20, clip: float = 3.0):
        self.window = window
        self.clip = clip

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        p = prices.shift(1)                             # use prices through t-1
        mu    = p.rolling(self.window, min_periods=self.window // 2).mean()
        sigma = p.rolling(self.window, min_periods=self.window // 2).std()
        z = (p - mu) / sigma.replace(0, np.nan)
        return z.clip(-self.clip, self.clip)


class RSI14d(BaseFactor):
    """
    14-period Relative Strength Index, negated for use as a return predictor.

    Formula: RSI_{14,t} = 100 - 100 / (1 + AvgGain / AvgLoss)
    where gains/losses are computed over periods [t-15, t-1] (14 changes,
    fully lagged by 1 day so today's price is excluded).

    Convention: higher RSI = overbought = lower expected return.
    The factor is returned as NEGATIVE RSI so that a high factor value
    consistently means high expected return (consistent with other factors).
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # Daily changes, then shift so index t uses change at t-1
        delta = prices.diff(1).shift(1)

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        # Wilder's smoothing: EWM with alpha = 1/period
        alpha = 1.0 / self.period
        avg_gain = gain.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()

        rs  = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Negate: high RSI (overbought) → negative factor → low expected return
        return -rsi
