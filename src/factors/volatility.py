"""src/factors/volatility.py — RVol20d and ATR14d.

These factors are NOT direct return predictors.
Primary uses:
  RVol20d → regime detection input + risk-parity position sizing
  ATR14d  → supplementary position sizing (more robust to limit-up/down)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.factors.base import BaseFactor


class RVol20d(BaseFactor):
    """
    Annualised 20-day realised volatility of log returns.

    Formula: RVol_{20,t} = sqrt(252) * std(log(P_i/P_{i-1}))_{i in [t-20, t-1]}

    Uses log returns; the rolling window is shifted so value at t uses
    returns through t-1 only.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        log_ret = np.log(prices / prices.shift(1))
        # .shift(1) on the rolling result: std at t uses log returns [t-window-1, t-1]
        rvol = (
            log_ret
            .rolling(self.window, min_periods=self.window // 2)
            .std()
            .shift(1)
            * np.sqrt(252)
        )
        return rvol


class ATR14d(BaseFactor):
    """
    14-period Average True Range (Wilder's EWM), normalised by close price.

    TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
    ATR_{14,t} = EWM_{alpha=1/14}(TR_{t-1})  / C_{t-1}

    Normalised by close so the value is comparable across stocks.
    Shifted by 1 day so value at t uses data through t-1.

    Requires highs and lows DataFrames passed as kwargs:
        factor.compute(prices, highs=highs_df, lows=lows_df)
    """

    def __init__(self, period: int = 14):
        self.period = period

    def compute(
        self,
        prices: pd.DataFrame,
        highs: pd.DataFrame | None = None,
        lows: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        if highs is None or lows is None:
            raise ValueError("ATR14d.compute() requires `highs` and `lows` DataFrames.")

        prev_close = prices.shift(1)

        tr = pd.concat(
            [
                (highs - lows).abs(),
                (highs - prev_close).abs(),
                (lows  - prev_close).abs(),
            ],
            axis=0,
        ).groupby(level=0).max()

        # Shift TR by 1 so value at t uses TR_{t-1}
        tr_shifted = tr.shift(1)

        alpha = 1.0 / self.period
        atr = tr_shifted.ewm(alpha=alpha, min_periods=self.period, adjust=False).mean()

        # Normalise by previous close to make it scale-free
        return atr / prev_close.replace(0, np.nan)
