"""src/factors/__init__.py

Factor registry: computes all 6 factors over the full union price panel
and writes them to data/processed/factors/.

Call: build_all_factors()
"""
from __future__ import annotations

import logging

import pandas as pd

from src.utils import load_config, read_parquet, write_parquet, PROCESSED
from src.factors.momentum   import Mom5d, Mom20d
from src.factors.reversion  import ZScore20d, RSI14d
from src.factors.volatility import RVol20d, ATR14d

logger = logging.getLogger(__name__)

FACTOR_MAP = {
    "mom_5d":     Mom5d(),
    "mom_20d":    Mom20d(),
    "zscore_20d": ZScore20d(),
    "rsi_14d":    RSI14d(),
    "rvol_20d":   RVol20d(),
    "atr_14d":    ATR14d(),
}


def build_all_factors() -> None:
    """Compute all factors and persist. Idempotent."""
    factor_dir = PROCESSED / "factors"
    factor_dir.mkdir(parents=True, exist_ok=True)

    prices = read_parquet(PROCESSED / "prices.parquet")
    highs  = read_parquet(PROCESSED / "highs.parquet")
    lows   = read_parquet(PROCESSED / "lows.parquet")

    cfg = load_config()["factors"]

    for name, factor in FACTOR_MAP.items():
        dest = factor_dir / f"{name}.parquet"
        if dest.exists():
            logger.info("Factor %s already computed — skipping.", name)
            continue

        logger.info("Computing factor: %s …", name)

        # Re-instantiate with config hyperparams where needed
        if name == "zscore_20d":
            factor = ZScore20d(window=cfg["zscore_window"], clip=cfg["zscore_clip"])
        elif name == "rvol_20d":
            factor = RVol20d(window=cfg["rvol_window"])
        elif name == "atr_14d":
            factor = ATR14d(period=cfg["atr_period"])

        # ATR needs highs and lows; all others just need prices
        kwargs = {"highs": highs, "lows": lows} if name == "atr_14d" else {}
        result = factor.compute(prices, **kwargs)

        write_parquet(result, dest)
        logger.info("Factor %s saved: shape %s.", name, result.shape)

    logger.info("All factors built.")


def load_factor(name: str) -> pd.DataFrame:
    """Load a factor DataFrame from disk by name (e.g. 'mom_5d')."""
    return read_parquet(PROCESSED / "factors" / f"{name}.parquet")


def load_all_factors() -> dict[str, pd.DataFrame]:
    """Return all 6 factors as a dict {name: DataFrame}."""
    return {name: load_factor(name) for name in FACTOR_MAP}
