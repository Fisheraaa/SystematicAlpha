"""src/utils.py — shared helpers used across all modules."""
from __future__ import annotations

import pathlib
import yaml
import pandas as pd


ROOT = pathlib.Path(__file__).parent.parent          # repo root
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
RESULTS = DATA / "results"


def load_config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def read_parquet(path: pathlib.Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: pathlib.Path | str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def trading_calendar() -> pd.DatetimeIndex:
    """Return the full A-share trading calendar as a DatetimeIndex."""
    cal = read_parquet(CACHE / "trading_calendar.parquet")
    return pd.DatetimeIndex(cal[cal["is_open"] == 1]["cal_date"])
