# Systematic Alpha Research & Trading Framework
## Product Requirements Document — v1.1 (2026)

> **Tagline:** A bias-controlled, regime-aware long-only equity research platform
> targeting the A-share universe (CSI 300 / CSI 500), with independent per-universe
> analysis and a combined allocation strategy, built for reproducibility and
> institutional rigour.

---

## Table of Contents

1. [Project Motivation](#1-project-motivation)
2. [System Architecture](#2-system-architecture)
3. [Technical Constraints](#3-technical-constraints)
4. [Module 1 — Data Layer](#4-module-1--data-layer)
5. [Module 2 — Universe Construction](#5-module-2--universe-construction)
6. [Module 3 — Factor Library](#6-module-3--factor-library)
7. [Module 4 — Factor Validation](#7-module-4--factor-validation)
8. [Module 5 — Regime Detection](#8-module-5--regime-detection)
9. [Module 6 — Backtest Engine](#9-module-6--backtest-engine)
10. [Module 7 — Portfolio & Risk Management](#10-module-7--portfolio--risk-management)
11. [Module 8 — Walk-Forward Validation](#11-module-8--walk-forward-validation)
12. [Module 9 — Output & Reporting](#12-module-9--output--reporting)
13. [Repository Structure](#13-repository-structure)
14. [Acceptance Criteria](#14-acceptance-criteria)
15. [Roadmap & Milestones](#15-roadmap--milestones)

---

## 1. Project Motivation

Most retail backtests share three silent flaws: they are tested on today's index
constituents (survivorship bias), they mix training and test data chronologically
(look-ahead bias), and they ignore realistic transaction costs. The result is a
Sharpe ratio that is flattering in-sample and flat out-of-sample.

This framework treats each of these as a first-class engineering constraint, not
an afterthought. The goal is a **long-only equity strategy on Chinese A-shares**
(CSI 300 / CSI 500) that can answer two questions honestly:

> "Does this factor have statistically significant predictive power, and does
> that power survive realistic trading costs on unseen data?"

The project is intentionally scoped as a **research platform** rather than a
single script. Every module has a clean interface so that new factors, new
regime signals, and new cost assumptions can be plugged in without touching the
backtest engine.

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│                     config/config.yaml                  │
│          (single source of truth for all parameters)    │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────┐
         │         src/data/                │
         │  fetcher → processor → universe  │
         └───────────────┬──────────────────┘
                         │ Parquet (date × ts_code)
         ┌───────────────▼──────────────────┐
         │       src/factors/               │
         │  BaseFactor → 6 concrete factors │
         │  (computed once over full pool)  │
         └───────────────┬──────────────────┘
                         │
              ┌──────────┴──────────┐
              │  universe slice at  │
              │  each rebalance date│
              └──┬───────────┬──────┘
                 │           │
        CSI 300  │           │  CSI 500
                 ▼           ▼
         ┌─────────────────────────────────┐
         │      src/validation/            │
         │  IC · quantile (per universe)   │
         └──────────────┬──────────────────┘
                        │  per-universe IC weights
                        ▼
         ┌─────────────────────────────────────────────┐
         │  src/strategy/ + src/backtest/              │
         │                                             │
         │  run_id="csi300"  run_id="csi500"           │
         │  universe=300     universe=500              │
         │  benchmark=300TR  benchmark=500TR           │
         │        │                 │                  │
         │        └────────┬────────┘                  │
         │                 ▼                           │
         │         run_id="combined"                   │
         │         40% csi300 sub-portfolio            │
         │       + 60% csi500 sub-portfolio            │
         │         benchmark=blended TR                │
         └─────────────────┬───────────────────────────┘
                           │ 3 × daily PnL streams
         ┌─────────────────▼───────────────────────────┐
         │       src/report/                           │
         │  3 equity curves · comparison table         │
         │  per-universe IC tables · HTML report       │
         └─────────────────────────────────────────────┘
```

### 2.2 Three-Run Design

The framework executes **three independent backtest runs** from a single
data and factor computation pass:

| `run_id` | Universe | Benchmark | Slippage | Notes |
|---|---|---|---|---|
| `csi300` | CSI 300 PIT constituents | CSI 300 TR | 0.05% | Large-cap only |
| `csi500` | CSI 500 PIT constituents | CSI 500 TR | 0.10% | Mid-cap only |
| `combined` | Both (union) | 40/60 blended TR | By stock | Sub-allocation: 40% csi300 + 60% csi500 |

Factor computation runs **once** over the full union of all stocks that ever
appeared in either index. Each run then slices to its own point-in-time
universe at each rebalance date. This avoids redundant computation and
guarantees that the same factor values are used across all three runs —
differences in run performance are attributable to universe composition and
cost differences, not to different factor computations.

Factor IC weights are **calibrated independently** per universe. The
`combined` run uses a weighted average of the two per-universe weight vectors
(40/60) rather than a separate re-calibration, to maintain interpretability.

Data flows in one direction only. No module downstream of the factor layer
is ever allowed to reference raw price data — it receives only pre-computed,
already-shifted factor values.

---

## 3. Technical Constraints

### 3.1 Permitted Stack

| Layer | Libraries |
|---|---|
| Core logic | Python 3.11+, Pandas 2.x, NumPy |
| Performance | Polars (optional, for large cross-sections) |
| Statistics | SciPy (t-tests, Spearman correlation) |
| Visualisation | Plotly (interactive HTML), Matplotlib (static) |
| Storage | Parquet (via PyArrow), SQLite (trade log) |
| Config | PyYAML |
| Testing | pytest |

### 3.2 Hard Prohibitions

The following will be treated as fatal errors during code review:

| Prohibited Pattern | Reason |
|---|---|
| `df.sample(frac=0.8)` or random train/test split | Destroys temporal ordering; creates leakage |
| Any feature computed with `closed='right'` (default) in a rolling window without a subsequent `.shift(1)` | Look-ahead bias |
| Using today's CSI 300 / CSI 500 constituent list for historical backtests | Survivorship bias |
| `df['label'] = df['close'].pct_change()` without `.shift(-1)` on the label side | Label leakage |
| AutoML, deep learning, or any black-box model | Violates the explainability requirement |
| Ignoring stamp duty, commission, or slippage | Produces unrealistically optimistic returns |
| Short-selling | Outside long-only scope; A-share borrow constraints apply |

### 3.3 Data Split Convention

```
|<-------- In-Sample (IS) -------->|<----- Out-of-Sample (OOS) ----->|
|   Factor development & tuning    |  Walk-forward validation only    |
|   ~2015-01 to 2020-12            |  2021-01 to 2024-12              |
```

The OOS period is **never touched** until the strategy specification is frozen.
No parameter tuning of any kind may reference OOS data.

---

## 4. Module 1 — Data Layer

### 4.1 Data Source

**Primary: Tushare Pro** (`tushare.pro_api(token)`)

Tushare Pro is the de-facto standard data source for Chinese A-share
quantitative research. The free-tier token provides access to:

- Daily OHLCV bars (adjusted and unadjusted)
- Adjustment factors for dividend and split correction
- **Historical index constituent snapshots** (the key dataset for point-in-time
  universe construction, unavailable in most free alternatives)
- Trading calendar
- Listing/delisting dates

Fallback: AKShare (free, no registration), used only if Tushare daily call
limits are exhausted. AKShare does not provide historical constituent snapshots,
so the universe module must apply a conservative approximation for AKShare runs
(flag results accordingly in the report).

### 4.2 Data Fetching Design

The fetcher follows an **idempotent pull** pattern: before any API call, it
checks whether the local Parquet file for that ticker and date range already
exists. If it does, the call is skipped. This means the full dataset can be
rebuilt from scratch with a single `python main.py --fetch`, and subsequent
runs are instantaneous.

**Fetching sequence:**

1. Pull trading calendar → `data/cache/trading_calendar.parquet`
2. Pull CSI 300 and CSI 500 historical constituent weights →
   `data/raw/index_weight/{index_code}.parquet`
3. For each unique `ts_code` that ever appeared in either index:
   - Pull unadjusted daily OHLCV (`daily`)
   - Pull adjustment factors (`adj_factor`)
   - Merge and compute forward-adjusted close
   - Save to `data/raw/daily/{ts_code}.parquet`

### 4.3 Adjustment Convention

Use **backward (forward) price adjustment** (前复权):

```
adj_close_t = close_t × (adj_factor_t / adj_factor_latest)
```

The ratio is normalised to today's price level so that the most recent close
equals the unadjusted close. This is the standard convention for momentum
factor construction (price continuity is preserved across dividend events).

### 4.4 Data Preprocessing

Applied inside `processor.py` after fetching, before any factor computation:

**Missing values:** Forward-fill (`ffill`) for price and volume data to handle
non-trading days in merged DataFrames. If a stock has more than 5 consecutive
missing trading days, flag it as suspended and exclude it from the universe for
that period.

**Outlier truncation (Winsorisation):** For each cross-section date, clip
return values to the [1st, 99th] percentile. Applied to raw returns before they
are used as factor labels. Not applied to factor values themselves (those are
rank-transformed at the IC calculation stage).

**Delisting handling:** When a stock is delisted, its final return is typically
large and negative. Include the delisting return in the backtest (do not drop it)
to correctly account for the cost of holding a stock that gets removed from the
index.

### 4.5 Storage Schema

All processed data is stored as Parquet with a `(date, ts_code)` multi-index.
This enables efficient slicing by either dimension.

```
data/
├── raw/
│   ├── daily/
│   │   └── {ts_code}.parquet        # columns: open, high, low, close,
│   │                                #           volume, adj_close, adj_factor
│   └── index_weight/
│       ├── 000300.SH.parquet        # columns: ts_code, weight, date
│       └── 000905.SH.parquet
├── processed/
│   ├── prices.parquet               # wide: date × ts_code (full union pool)
│   ├── returns.parquet              # wide: date × ts_code (full union pool)
│   ├── universe/
│   │   ├── csi300/
│   │   │   └── {YYYYMMDD}.parquet   # PIT eligible ts_codes, CSI 300
│   │   └── csi500/
│   │       └── {YYYYMMDD}.parquet   # PIT eligible ts_codes, CSI 500
│   └── factors/                     # computed over full union pool;
│       ├── mom_5d.parquet           # sliced per run_id at usage time
│       ├── mom_20d.parquet
│       ├── zscore_20d.parquet
│       ├── rsi_14d.parquet
│       ├── rvol_20d.parquet
│       └── atr_14d.parquet
├── results/
│   ├── csi300/
│   │   ├── ic_summary.parquet       # IC stats per factor
│   │   ├── daily_state.parquet      # equity, cash, drawdown
│   │   └── trades.db                # SQLite trade log
│   ├── csi500/
│   │   └── ...                      # same structure
│   └── combined/
│       └── ...                      # same structure
└── cache/
    └── trading_calendar.parquet
```

---

## 5. Module 2 — Universe Construction

### 5.1 The Survivorship Bias Problem

The CSI 300 and CSI 500 indices rebalance semi-annually. A stock that is in
the index today may have been added precisely because it performed well over the
past several years. If we test a strategy using today's constituents on 2018
data, we are implicitly granting the strategy knowledge of future survivors —
an information advantage that no real investor could have had.

This is not a minor adjustment. Studies on US equities estimate that
survivorship bias inflates backtest Sharpe ratios by 0.3–0.8.

### 5.2 Point-in-Time Universe

At each rebalance date `t`, the eligible universe is defined strictly as the set
of stocks that were **actually in the target index at time `t`**, as recorded in
the historical constituent snapshot.

**Single-index implementation:**

```python
def get_universe(date: str, index_code: str) -> list[str]:
    """
    Return the point-in-time constituent list for `index_code` as of `date`.
    Uses the most recent constituent announcement on or before `date`.

    Args:
        date:       rebalance date, format 'YYYYMMDD'
        index_code: '000300.SH' or '000905.SH'

    Returns:
        List of ts_codes in the index at `date`.
    """
    constituents = load_parquet(f"data/raw/index_weight/{index_code}.parquet")
    snapshot = constituents[constituents["date"] <= date].sort_values("date")
    latest = snapshot.groupby("ts_code").last()
    return latest.index.tolist()
```

**Combined-universe implementation:**

For `run_id="combined"`, the eligible pool is the **union** of both indices'
PIT constituent lists. Each stock is tagged with its source index
(`index_membership ∈ {"csi300", "csi500", "both"}`). Stocks appearing in both
indices at date `t` are tagged "both" and use the CSI 300 slippage rate.

```python
def get_combined_universe(date: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [ts_code, index_membership].
    Used exclusively by run_id='combined'.
    """
    csi300 = set(get_universe(date, "000300.SH"))
    csi500 = set(get_universe(date, "000905.SH"))
    records = []
    for code in csi300 | csi500:
        if code in csi300 and code in csi500:
            membership = "both"
        elif code in csi300:
            membership = "csi300"
        else:
            membership = "csi500"
        records.append({"ts_code": code, "index_membership": membership})
    return pd.DataFrame(records)
```

### 5.3 Additional Eligibility Filters

Applied after constituent filtering, using only information available at time `t`.
Identical filters are applied across all three run modes:

| Filter | Criterion | Rationale |
|---|---|---|
| Suspension | `volume_t > 0` | Suspended stocks cannot be traded |
| Limit-up/down | `abs(return_t) < 0.095` | Stocks at limit cannot be reliably entered/exited |
| Newly listed | `listing_date < t − 60 trading days` | Insufficient price history for rolling factors |
| Data completeness | All 6 factors are non-null at `t` | Cannot rank stocks with missing signals |

### 5.4 Rebalancing Schedule

Monthly rebalancing on the **first trading day of each calendar month**.
This frequency balances turnover cost against signal staleness, based on the
IC decay analysis from Module 4.

The same calendar is used across all three runs so that performance differences
are attributable to universe composition, not to timing differences.

### 5.5 Expected Universe Sizes

At a typical rebalance date, after applying eligibility filters:

| Run | Raw constituents | Post-filter (approx.) |
|---|---|---|
| `csi300` | 300 | 270–290 |
| `csi500` | 500 | 440–470 |
| `combined` union | ~700 (overlap ~30) | 700–750 |

The ~30-stock overlap between CSI 300 and CSI 500 is negligible but must be
tracked to avoid double-counting in the `combined` run's position sizing.

---

## 6. Module 3 — Factor Library

### 6.1 Abstract Interface

```python
from abc import ABC, abstractmethod
import pandas as pd

class BaseFactor(ABC):
    """
    All factors must implement this interface.
    Input: wide-format adjusted close price DataFrame (date × ts_code).
    Output: wide-format factor value DataFrame (date × ts_code).
    Contract: output.index == input.index; no future data may appear in any cell.
    """
    @abstractmethod
    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...

    def validate_no_lookahead(self, prices: pd.DataFrame, factor: pd.DataFrame):
        """
        Regression test: factor_t must have zero correlation with return_t
        (same-day return). Any non-trivial correlation indicates look-ahead.
        Raises AssertionError if Pearson |r| > 0.01.
        """
        same_day_return = prices.pct_change()
        for date in factor.index[-20:]:   # spot-check last 20 dates
            f = factor.loc[date].dropna()
            r = same_day_return.loc[date].reindex(f.index).dropna()
            corr = f.corr(r)
            assert abs(corr) < 0.01, (
                f"Look-ahead detected in {self.__class__.__name__} "
                f"on {date}: corr={corr:.4f}"
            )
```

### 6.2 Factor Definitions

All factors use `closed='left'` rolling windows or an explicit `.shift(1)` on
the output to guarantee that the value at date `t` uses only prices through
`t-1`.

---

**Factor 1: Short-term Momentum (Mom5d)**

$$\text{Mom5d}_t = \frac{P_{t-1}}{P_{t-6}} - 1$$

Five-day price return, skipping the most recent day (`shift(1)`) to avoid the
short-term reversal that is well-documented in A-share markets.

```python
def compute(self, prices):
    return prices.pct_change(5).shift(1)
```

---

**Factor 2: Medium-term Momentum (Mom20d)**

$$\text{Mom20d}_t = \frac{P_{t-2}}{P_{t-22}} - 1$$

Twenty-day return with a two-day skip (`shift(2)`). The two-day skip is
deliberate: A-share empirical literature shows a stronger short-term reversal
effect than US markets, and skipping two days rather than one produces more
stable IC in the 20-day window.

```python
def compute(self, prices):
    return prices.pct_change(20).shift(2)
```

---

**Factor 3: Mean-Reversion Z-Score (ZScore20d)**

$$Z_t = \frac{P_{t-1} - \mu_{[t-21,\, t-1]}}{\sigma_{[t-21,\, t-1]}}$$

The rolling mean and standard deviation use a 20-day window with `closed='left'`
so that day `t`'s price is never included. A large positive Z-score indicates
the stock is extended above its recent mean and is a candidate for reversion
(negative expected return signal). Winsorise Z at ±3 before use to reduce the
influence of outliers driven by corporate events.

```python
def compute(self, prices):
    mu = prices.shift(1).rolling(20, closed='left').mean()
    sigma = prices.shift(1).rolling(20, closed='left').std()
    z = (prices.shift(1) - mu) / sigma
    return z.clip(-3, 3)
```

---

**Factor 4: Relative Strength Index (RSI14d)**

$$\text{RSI}_{14,t} = 100 - \frac{100}{1 + \frac{\overline{\text{Gain}}_{14}}{\overline{\text{Loss}}_{14}}}$$

where gains and losses are computed from price changes `t-15` through `t-1`
(14 periods, fully lagged by one day).

RSI > 70 flags overbought conditions (bearish signal for mean reversion);
RSI < 30 flags oversold conditions (bullish signal). The raw RSI value is used
as a continuous factor — higher RSI implies lower expected return — so the
factor is negated before entry into the signal composite.

---

**Factor 5: Realised Volatility (RVol20d)**

$$\text{RVol}_{20,t} = \sqrt{252} \cdot \sigma\!\left(\ln\frac{P_i}{P_{i-1}}\right)_{i \in [t-20,\, t-1]}$$

Annualised 20-day realised volatility of log returns. Primary use cases:

1. **Regime detection input** (Module 5): high vs. low volatility regime classification.
2. **Risk Parity position sizing** (Module 7): inverse-volatility weights.

The factor is not directly used as a return predictor in the signal composite.

---

**Factor 6: Average True Range (ATR14d)**

$$\text{TR}_t = \max(H_t - L_t,\; |H_t - C_{t-1}|,\; |L_t - C_{t-1}|)$$

$$\text{ATR}_{14,t} = \text{EWM}_{\alpha=1/14}(\text{TR}_{t-1})$$

True Range captures intraday volatility and gap risk more directly than
return-based volatility, making it a more robust risk proxy in environments
with frequent limit-up/down events. The EWM is computed on TR values shifted
by one day. Used as a supplementary position sizing input alongside RVol.

---

### 6.3 Factor Cross-Sectional Normalisation

Before any downstream use (validation, signal construction, or position sizing),
each factor is **cross-sectionally rank-normalised** within the daily universe:

$$\tilde{f}_{i,t} = \frac{\text{rank}(f_{i,t}) - 0.5}{N_t}$$

This maps factor values to `(0, 1)` regardless of the factor's raw distribution,
removing the influence of outliers and making IC calculations stable.
Rank normalisation is applied at validation time, not stored — the raw factor
values are preserved in Parquet.

---

## 7. Module 4 — Factor Validation

Factor validation is the **analytical core** of the project. The backtest result
is only as credible as the factors going into it.

### 7.1 Per-Universe Execution

All validation analyses (IC, decay, quantile) are executed **independently for
each universe** — `csi300` and `csi500` — using only the stocks eligible in
that universe at each date. The `combined` universe does not receive a separate
IC calibration; it uses the weighted average of the two sets of IC weights
(40/60).

This separation is intentional: CSI 300 (large-cap) and CSI 500 (mid-cap)
have different return distributions, liquidity profiles, and analyst coverage.
Factor effectiveness often diverges between the two segments, and the
per-universe analysis makes this divergence explicit and quantifiable.

### 7.2 Information Coefficient Analysis

The IC measures the cross-sectional Spearman rank correlation between the
factor value at date `t` and the forward one-day return at `t+1`:

$$\text{IC}_t = \rho_s\!\left(\tilde{f}_t,\; r_{t+1}\right)$$

Spearman rank correlation is preferred over Pearson because A-share return
distributions are leptokurtic and exhibit frequent extreme observations
(limit-up/down events), which Pearson is sensitive to.

**Output: per-factor, per-universe summary table**

| Factor | CSI300 Mean IC | CSI300 ICIR | CSI300 t-stat | CSI300 p-val | CSI500 Mean IC | CSI500 ICIR | CSI500 t-stat | CSI500 p-val |
|---|---|---|---|---|---|---|---|---|
| Mom5d | | | | | | | | |
| Mom20d | | | | | | | | |
| ZScore20d | | | | | | | | |
| RSI14d | | | | | | | | |

This table is the centrepiece of Section 3 of the HTML report. It makes the
research finding "factor X works better in mid-caps than large-caps" immediately
legible to a reader.

**Per-factor statistics:**

| Statistic | Formula | Interpretation |
|---|---|---|
| Mean IC | $\bar{IC}$ | Average predictive power |
| IC Std | $\sigma_{IC}$ | Stability of predictive power |
| ICIR | $\bar{IC} / \sigma_{IC}$ | Signal-to-noise ratio; threshold > 0.3 |
| t-statistic | $\bar{IC} / (\sigma_{IC} / \sqrt{T})$ | Hypothesis test: $H_0: \bar{IC} = 0$ |
| p-value | From t-distribution, df=T−1 | Statistical significance; threshold < 0.05 |
| IC > 0 (%) | Fraction of dates with positive IC | Directional consistency |

The t-statistic is the critical number for a research context: it is not enough
for the mean IC to be positive — it must be distinguishable from zero with
standard statistical confidence.

### 7.3 IC Decay Analysis

The IC is computed not just for the 1-day forward return but for forward
returns at horizons `k = 1, 2, 3, 5, 10` trading days, **separately for each
universe**. The resulting IC-decay curves are overlaid on the same chart to
reveal whether signal persistence differs between large-cap and mid-cap stocks.

A factor with IC that decays to zero by day 5 is a poor candidate for monthly
rebalancing (signal is stale by day 30). A factor with persistent IC at
day 10 can support monthly rebalancing without excessive signal decay.

Expected finding: momentum factors in CSI 500 tend to decay more slowly than
in CSI 300, because mid-cap stocks are followed by fewer analysts and
information diffuses more slowly.

### 7.4 Quantile Backtest (Long-Only Perspective)

At each rebalance date, stocks are sorted by factor rank **within each universe
independently** into five quintile groups (Q1 = lowest factor, Q5 = highest).
Each quintile is held at equal weight until the next rebalance.

**Key outputs (reported per universe):**

- Annualised return for each quintile (Q1 through Q5)
- Monotonicity check: does the return profile increase (or decrease) consistently
  from Q1 to Q5? Non-monotonic profiles indicate weak or noisy factor structure.
- Q5 − Q1 spread: the "pure" long-short alpha of the factor, used as a reference
  even though the live strategy is long-only.
- Q5 Information Ratio vs. respective index benchmark.

**Comparison output:** a side-by-side Q5 bar chart for CSI 300 vs CSI 500,
for each factor. Bars where CSI 500 Q5 significantly outperforms CSI 300 Q5
justify the 60% allocation weight to the mid-cap sub-portfolio in the
`combined` run.

### 7.5 Factor Correlation Matrix

Before combining factors into a composite signal, compute the pairwise
time-series correlation of IC values across all six factors, **per universe**.
High IC correlation (|r| > 0.7) between two factors indicates redundancy —
one should be dropped or the pair should be combined rather than used additively.

Report the correlation matrix for both universes. If the correlation structure
differs materially, the factor weighting scheme should differ between universes
(which it will, since IC-weighted composition is calibrated independently).

---

## 8. Module 5 — Regime Detection

### 8.1 Motivation

Momentum and mean-reversion strategies tend to work in different market
environments. Trend-following (momentum) works in persistent, low-volatility
bull markets; mean-reversion works in choppy, high-volatility environments.
A regime-aware signal composite adapts the factor weights to the current
market state rather than using a static blend.

### 8.2 Primary Regime Signal: Volatility Percentile

At each rebalance date `t`, compute the 252-day rolling percentile rank of
the CSI 300 index's 20-day realised volatility:

$$\text{VolPct}_t = \frac{\text{rank}(\text{RVol}_{20,t}^{\text{idx}},\; \text{window}=252)}{252}$$

| Regime | Condition | Active factors | Factor weights |
|---|---|---|---|
| Low-vol (trending) | VolPct ≤ 0.5 | Mom5d, Mom20d | 0.6, 0.4 |
| Transition | 0.5 < VolPct ≤ 0.7 | Blend | Linear interpolation |
| High-vol (choppy) | VolPct > 0.7 | ZScore20d, RSI14d | 0.5, 0.5 |

The transition zone uses linear interpolation between the two regime weight
vectors to avoid abrupt regime switches that generate unnecessary turnover.

### 8.3 Secondary Regime Indicator: Trend Strength (ADX)

Within the high-volatility regime, the **Average Directional Index (ADX)**
of the CSI 300 distinguishes between directional high-volatility (strong trend
despite noise) and non-directional high-volatility (genuinely choppy).

$$\text{ADX}_{14,t} = \text{EWM}_{14}\!\left(\frac{|DI^+ - DI^-|}{DI^+ + DI^-}\right)$$

If `VolPct > 0.7` **and** `ADX > 25`: the market is trending despite high
volatility → momentum factors retain 30% weight.

If `VolPct > 0.7` **and** `ADX ≤ 25`: genuinely choppy → pure mean-reversion.

### 8.4 Regime Signal Timing

The regime classification is determined on the **last trading day of the
current month** using data through that day, and applied to the **next month's**
portfolio. This ensures the regime signal is available before position changes
are executed and introduces no look-ahead bias.

---

## 9. Module 6 — Backtest Engine

### 9.1 A-Share Transaction Cost Model

A realistic cost model is non-negotiable for A-share research. The cost
structure as of 2024 (post stamp-duty reduction) is:

| Cost Component | Direction | Rate | Notes |
|---|---|---|---|
| Brokerage commission | Both | 0.025% | Minimum ¥5 per order |
| Stamp duty | Sell only | 0.05% | Reduced from 0.1% in Aug 2023 |
| Stock exchange fee | Both | 0.001% | Shanghai-listed stocks only |
| Market impact / slippage | Both | 0.05%–0.1% | Higher for CSI 500 (lower liquidity) |

**Total round-trip cost (buy + sell):**
- CSI 300 stocks: ≈ 0.25%
- CSI 500 stocks: ≈ 0.30% (higher slippage)

This becomes the baseline for robustness testing: the strategy must remain
viable when costs are doubled to 0.50% / 0.60%.

### 9.2 Execution Model

**T+1 constraint:** In A-share markets, stocks purchased on day `t` cannot be
sold until day `t+1`. The engine tracks `entry_date` for every position and
refuses any sell order where `current_date − entry_date < 1`.

**Signal-to-execution lag:**
- Rebalance signal is generated using data through day `t` close.
- Trades are executed at day `t+1` open.
- This one-day lag is mandatory and removes any possibility of using same-day
  closing prices to fill trades.

**Partial fills:** For simplicity, assume full fills at the execution price.
Record the theoretical share count as a float; convert to integer shares only
in the final position report (not during PnL calculation).

### 9.3 Backtest State Machine

Each trading day, the engine processes the following steps in order:

```
Step 1: Mark-to-market
  → Update market value of all open positions using today's close price.
  → equity_t = cash_t + Σ(shares_i × close_i)

Step 2: Risk checks
  → drawdown_t = 1 − equity_t / max(equity_0..t)
  → If drawdown_t > DRAWDOWN_THRESHOLD: trigger position reduction (Module 7).

Step 3: Rebalance (on rebalance dates only)
  → Compute target weights from signal composite (Module 5) and position
    sizer (Module 7).
  → Determine required trades: sells first, then buys (to free cash).
  → Apply T+1 constraint: only stocks held since at least yesterday can be sold.

Step 4: Execute trades
  → For each sell: proceeds = shares × price × (1 − commission − stamp_duty − slippage)
  → For each buy: cost = shares × price × (1 + commission + exchange_fee + slippage)
  → Update cash and positions.

Step 5: Log
  → Record (date, cash, equity, drawdown, turnover, trades_list) to SQLite.
```

### 9.4 Portfolio Tracking Data Structure

```python
@dataclass
class Position:
    ts_code: str
    shares: float
    entry_date: str
    avg_cost: float       # volume-weighted average cost basis

@dataclass
class DailyState:
    date: str
    cash: float
    positions: dict[str, Position]
    equity: float
    peak_equity: float
    drawdown: float
    turnover: float       # fraction of portfolio traded today
```

---

## 10. Module 7 — Portfolio & Risk Management

### 10.1 Target Stock Count

Hold between 20 and 40 stocks at any rebalance date. This range balances:
- Idiosyncratic diversification (≥20 stocks)
- Concentration to capture factor alpha (≤40 stocks; holding 100 stocks
  approximates the index)

Selection: the top N stocks by composite signal rank, where N is determined
by the regime (20 in high-vol regimes to reduce exposure, 40 in low-vol regimes).

### 10.2 Position Sizing: Risk Parity

Rather than equal-weighting the selected stocks, allocate capital inversely
proportional to each stock's realised volatility:

$$w_i = \frac{1/\sigma_i}{\sum_{j} 1/\sigma_j}, \quad \sigma_i = \text{RVol20d}_{i,t}$$

This ensures each stock contributes approximately equal risk to the portfolio,
preventing high-volatility small-caps from dominating the drawdown profile.

**Constraint:** Apply position bounds after computing raw weights:

$$w_i^{\text{final}} = \min\!\left(\max(w_i,\; w_{\min}),\; w_{\max}\right)$$

where $w_{\min} = 0.5\%$ and $w_{\max} = 8\%$.

After clamping, re-normalise so weights sum to the **invested fraction**
(see 10.4 below).

### 10.3 Drawdown Control

Define the high-water mark as the maximum equity value recorded since
inception. Drawdown is monitored daily (not just at rebalance dates).

| Drawdown Level | Action |
|---|---|
| < 10% | No action |
| 10%–15% | Reduce invested fraction to 70%; move 30% to cash |
| 15%–20% | Reduce invested fraction to 50%; move 50% to cash |
| > 20% | Reduce invested fraction to 20%; hard floor |

Recovery rule: when drawdown falls below 5%, restore invested fraction by
10% per month until fully invested again. This prevents a single bad
month from keeping the strategy in defensive mode indefinitely.

### 10.4 Cash Buffer

Maintain a minimum 5% cash reserve at all times for:
- Covering transaction costs without forced sells
- Absorbing intra-period dividend receipts
- Reducing the impact of T+1 execution lags

Maximum invested fraction (excluding the drawdown reduction rule) is
therefore 95%.

### 10.5 Benchmarks (Per Run)

Each run is evaluated against its own appropriate benchmark. Using a single
CSI 300 benchmark for all three runs would unfairly penalise the CSI 500 run,
whose universe has a different risk-return profile.

| `run_id` | Benchmark | Rationale |
|---|---|---|
| `csi300` | CSI 300 Total Return Index | Like-for-like large-cap comparison |
| `csi500` | CSI 500 Total Return Index | Like-for-like mid-cap comparison |
| `combined` | 40% CSI 300 TR + 60% CSI 500 TR (daily rebalanced) | Matches the sub-portfolio allocation weights |

The blended benchmark for `combined` is computed daily:

$$R_{\text{blend},t} = 0.40 \times R_{\text{CSI300},t} + 0.60 \times R_{\text{CSI500},t}$$

All Information Ratio, alpha, and tracking error calculations use the
run-specific benchmark. Cross-run comparisons use the blended benchmark as a
common reference to put all three equity curves on the same footing in
Chart 1 (Section 12.2).

---

## 11. Module 8 — Walk-Forward Validation

### 11.1 Why Walk-Forward

A single train/test split produces one out-of-sample observation. Walk-forward
validation produces multiple independent out-of-sample windows, yielding a
**distribution** of performance outcomes. This is far more informative:

- Mean OOS Sharpe > 1.0: strategy is likely robust
- Fraction of OOS windows with positive Sharpe > 70%: signal is consistent
- OOS Sharpe mean / IS Sharpe mean > 0.5: strategy is not badly overfit

### 11.2 Walk-Forward Configuration

```
Training window:   24 months (rolling, not expanding)
Test window:       6 months
Step size:         3 months (test windows overlap by 3 months)
Total OOS period:  2021-01 to 2024-12 (4 years → 8 non-overlapping half-years
                   = 13 rolling test windows with 3-month step)
```

At each step:
1. Re-estimate factor weights (IC-weighted composite) using only the training
   window data.
2. Re-run regime thresholds calibration on the training window.
3. Execute the backtest on the test window using the newly calibrated parameters.
4. Record: Sharpe, Max Drawdown, Annual Return, Turnover, Information Ratio.

### 11.3 Aggregated OOS Report

After all windows complete:

- Box plot of OOS Sharpe ratios across windows
- Bar chart of OOS Annual Return by window, with CSI 300 benchmark overlay
- Hit rate: percentage of windows where strategy outperforms benchmark
- Maximum consecutive losing windows

---

## 12. Module 9 — Output & Reporting

### 12.1 Performance Metrics

All metrics are computed for both IS and OOS periods, and for each
walk-forward window.

| Metric | Formula |
|---|---|
| Annual Return | $(1 + \bar{r})^{252} - 1$ |
| Sharpe Ratio | $\sqrt{252} \cdot \bar{r}_{\text{excess}} / \sigma_r$ |
| Max Drawdown | $\max_{t} (1 - \text{equity}_t / \text{peak}_t)$ |
| Calmar Ratio | Annual Return / Max Drawdown |
| Information Ratio | $(R_{\text{strategy}} - R_{\text{benchmark}}) / \sigma_{\text{tracking error}}$ |
| Monthly Turnover | Average fraction of portfolio replaced per month |
| Win Rate | Fraction of months with positive excess return |

### 12.2 Required Visualisations

All charts are produced as interactive Plotly HTML files and as static PNG
for the README.

**Chart 1: Three-Run Equity Curve Comparison**
All three strategy runs (`csi300`, `csi500`, `combined`) plotted on a single
chart with their respective benchmarks, normalised to 100 at the IS start date.
Use distinct colours for strategies vs. benchmarks (solid lines for strategies,
dashed for benchmarks). The lower panel shows the drawdown for each run
overlaid. Regime periods (high-vol / low-vol) marked as background colour bands
on the shared x-axis. This is the headline chart of the entire report.

**Chart 2: Per-Universe Factor IC Time Series**
Two panels (CSI 300 | CSI 500), side by side. For each factor: rolling 60-day
mean IC with ±1 standard deviation band. Periods where IC crosses zero and
stays negative for more than 30 days are flagged as "factor decay events."
The side-by-side layout directly visualises where factor effectiveness diverges
between large-cap and mid-cap stocks.

**Chart 3: Quantile Return Bar Chart**
Two sets of five bars (Q1–Q5) per factor — one set for CSI 300, one for CSI 500
— with error bars showing the standard error of quarterly quintile returns.
Bars are grouped by quintile so the CSI 300 vs. CSI 500 difference is visible
for each quintile simultaneously. This chart justifies the 40/60 allocation
split: if CSI 500 Q5 consistently outperforms CSI 300 Q5, the higher weight
is earned by the data.

**Chart 4: Walk-Forward Performance Distribution**
Box plots of Sharpe, Max Drawdown, and Annual Return across all OOS windows,
separately for each of the three runs. Six box plots per metric (3 runs × IS/OOS).
The IS vs. OOS degradation ratio is annotated on each pair.

**Chart 5: Turnover & Cost Analysis**
Monthly turnover rate on the primary axis; cumulative transaction cost drag on
the secondary axis. Plotted for all three runs on the same chart to make the
cost premium of the CSI 500 run (higher slippage) quantitatively visible.

**Chart 6: Universe Composition Over Time**
Stacked area chart showing the count of eligible stocks per run over the
backtest period. Marks index rebalancing events and periods of high suspension
rates. Demonstrates that the universe is genuinely dynamic (not static), and
that PIT filtering removes a non-trivial number of stocks at any given date.

### 12.3 Failure Case Analysis (Required)

The report must include a section explicitly documenting **when and why the
strategy fails**. This is not optional — a strategy report without failure
analysis is not credible.

Required failure cases to analyse:

1. **Regime misclassification lag:** During rapid market transitions (e.g.,
   February 2020 crash, October 2022 policy shift), the volatility percentile
   signal has a detection delay of ~10–20 trading days. Quantify the PnL impact
   of this lag.

2. **Factor crowding:** When many systematic strategies hold the same stocks,
   the unwinding of these positions causes correlated drawdowns. The strategy
   has no crowding detection; document periods where this was likely a factor.

3. **T+1 execution cost in fast-moving markets:** If the market gaps down
   significantly between signal generation and the next-day open, the
   realised fill price is materially worse than assumed. Quantify the
   worst-case gap risk in the dataset.

### 12.4 Robustness Checks (Required)

| Parameter | Baseline | Stress Test |
|---|---|---|
| Transaction cost | 0.25% round-trip | 0.50% round-trip (double) |
| Regime threshold | VolPct = 70th percentile | 60th and 80th percentile |
| Rebalance frequency | Monthly | Weekly and Quarterly |
| Portfolio size | 20–40 stocks | 10–20 stocks and 40–80 stocks |
| Training window | 24 months | 12 months and 36 months |

For each stress test, report the change in Sharpe ratio and Max Drawdown
relative to the baseline. A robust strategy should show Sharpe degradation
of less than 30% under any single stress test.

### 12.5 HTML Research Report

Generate a single self-contained `report.html` (all charts inlined as base64
or embedded Plotly JSON) that can be opened in any browser without a server.
This is the primary deliverable for sharing with interviewers.

Structure:

```
1. Executive Summary
   ├── Three-run results table: Sharpe · Drawdown · Annual Return · IR vs benchmark
   └── Headline chart: three equity curves vs. respective benchmarks (Chart 1)

2. Strategy Description
   ├── Factor composite & regime-switching logic
   └── Universe design: why CSI 300 + CSI 500 separately, and combined

3. Factor Validation Results
   ├── Per-universe IC summary table (CSI 300 columns | CSI 500 columns)
   ├── IC decay curves: CSI 300 vs CSI 500 side-by-side (Chart 2)
   └── Quintile analysis: per-factor, per-universe bar chart (Chart 3)

4. Backtest Results — CSI 300 Run
   ├── Equity curve vs. CSI 300 TR benchmark
   └── Performance metrics table (IS | OOS)

5. Backtest Results — CSI 500 Run
   ├── Equity curve vs. CSI 500 TR benchmark
   └── Performance metrics table (IS | OOS)

6. Backtest Results — Combined Run
   ├── Equity curve vs. blended benchmark
   ├── Performance metrics table (IS | OOS)
   └── Sub-portfolio attribution: CSI 300 contribution vs. CSI 500 contribution

7. Cross-Run Comparison
   ├── Three-run performance table (IS + OOS side by side)
   ├── Turnover & cost analysis (Chart 5)
   └── Universe composition over time (Chart 6)

8. Walk-Forward Validation
   ├── OOS performance distribution (Chart 4)
   └── Hit rate and consecutive-loss analysis per run

9. Robustness & Failure Analysis
   ├── Sensitivity table (cost double, regime threshold shift, rebalance freq)
   └── Failure case narratives with quantified PnL impact

10. Appendix
    └── Full trade log per run (downloadable CSV links)
```

---

## 13. Repository Structure

```
SystematicAlpha/
│
├── config/
│   └── config.yaml              # All tunable parameters (no magic numbers in code)
│
├── data/                        # Gitignored; populated by main.py --fetch
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetcher.py           # Tushare API calls; idempotent pull
│   │   ├── processor.py         # Adjustment, winsorisation, alignment
│   │   └── universe.py          # Point-in-time universe builder
│   │
│   ├── factors/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseFactor ABC + lookahead validator
│   │   ├── momentum.py          # Mom5d, Mom20d
│   │   ├── reversion.py         # ZScore20d, RSI14d
│   │   └── volatility.py        # RVol20d, ATR14d
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── ic_analysis.py       # IC, ICIR, t-test, IC decay
│   │   └── quantile.py          # Quintile sort backtest
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── regime.py            # VolPct + ADX regime classifier
│   │   ├── composite.py         # Factor weighting → composite signal
│   │   └── portfolio.py         # Risk parity weights, position bounds
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py            # Main backtest loop; state machine
│   │   ├── costs.py             # A-share cost model (commission, stamp, slippage)
│   │   ├── risk.py              # Drawdown control, cash management
│   │   └── metrics.py           # Performance metric calculations
│   │
│   └── report/
│       ├── __init__.py
│       ├── charts.py            # Plotly chart builders
│       └── generate.py          # HTML report assembly
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data quality checks, return distributions
│   └── 02_factor_research.ipynb     # IC analysis, quantile charts (exploratory)
│
├── tests/
│   ├── test_no_lookahead.py     # Automated lookahead bias detection
│   ├── test_universe_pit.py     # Verify point-in-time universe correctness
│   ├── test_costs.py            # Verify round-trip cost calculations
│   └── test_metrics.py          # Verify Sharpe, Drawdown formulae
│
├── main.py                      # CLI entry point
│                                # Usage: python main.py [--fetch] [--validate]
│                                #                       [--backtest] [--report]
│
├── requirements.txt
└── README.md
```

### 13.1 config.yaml Schema

```yaml
data:
  token: "YOUR_TUSHARE_TOKEN"
  indices: ["000300.SH", "000905.SH"]
  start_date: "2015-01-01"
  end_date: "2024-12-31"
  rebalance_day: "first_trading_day_of_month"

universe:
  min_listing_days: 60
  max_suspension_consecutive_days: 5
  limit_band_threshold: 0.095

factors:
  mom5d_skip: 1
  mom20d_skip: 2
  zscore_window: 20
  zscore_clip: 3.0
  rsi_period: 14
  rvol_window: 20
  atr_period: 14

regime:
  vol_percentile_window: 252
  low_vol_threshold: 0.50
  high_vol_threshold: 0.70
  adx_period: 14
  adx_trending_threshold: 25

strategy:
  min_holdings: 20
  max_holdings: 40
  min_position_weight: 0.005
  max_position_weight: 0.080
  cash_reserve_min: 0.05

backtest:
  initial_capital: 10_000_000    # CNY 10 million (per run; each run is independent)
  commission_rate: 0.00025
  stamp_duty_rate: 0.0005        # sell only
  exchange_fee_rate: 0.00001     # Shanghai-listed stocks only
  slippage_csi300: 0.0005
  slippage_csi500: 0.001
  execution_price: "next_day_open"

  # Three independent backtest runs executed from a single data/factor pass.
  # Factor values are computed once over the full union pool; each run slices
  # to its own point-in-time universe at every rebalance date.
  runs:
    - run_id: "csi300"
      universe: "000300.SH"
      benchmark: "000300.SH"          # CSI 300 Total Return Index
      slippage_override: 0.0005       # large-cap; tighter spread
      ic_weight_source: "csi300"      # use CSI 300-calibrated IC weights
      combined_allocation: null       # not used for single-index runs

    - run_id: "csi500"
      universe: "000905.SH"
      benchmark: "000905.SH"          # CSI 500 Total Return Index
      slippage_override: 0.001        # mid-cap; wider spread
      ic_weight_source: "csi500"
      combined_allocation: null

    - run_id: "combined"
      universe: ["000300.SH", "000905.SH"]   # union pool
      benchmark: "blended"            # 40% CSI300 TR + 60% CSI500 TR
      slippage_override: "by_stock"   # look up per-stock index membership
      ic_weight_source: "blended"     # 40/60 weighted avg of per-universe weights
      combined_allocation:
        "000300.SH": 0.40             # 40% of capital allocated to CSI300 sub-portfolio
        "000905.SH": 0.60             # 60% of capital allocated to CSI500 sub-portfolio

risk:
  drawdown_soft_limit: 0.10      # reduce to 70% invested
  drawdown_hard_limit: 0.15      # reduce to 50% invested
  drawdown_extreme: 0.20         # reduce to 20% invested
  recovery_threshold: 0.05
  recovery_rate_per_month: 0.10

walk_forward:
  train_months: 24
  test_months: 6
  step_months: 3
  is_start: "2015-01-01"
  is_end: "2020-12-31"
  oos_start: "2021-01-01"
  oos_end: "2024-12-31"
```

---

## 14. Acceptance Criteria

The project is considered complete when **all** of the following are satisfied:

### 14.1 Bias Controls

- [ ] `pytest tests/test_no_lookahead.py` passes for all 6 factors
- [ ] `pytest tests/test_universe_pit.py` confirms constituent lists match
  historical Tushare records on 5 randomly sampled dates
- [ ] Train/test split is strictly chronological; no shared data
- [ ] Config parameter `execution_price: next_day_open` is used throughout

### 14.2 Statistical Validity

- [ ] At least 2 of 6 factors have ICIR > 0.3 and IC t-stat p < 0.05 (IS period)
- [ ] At least 1 factor maintains IC direction consistency > 55% of months in OOS
- [ ] Quantile analysis shows monotonic Q1–Q5 return spread for at least 2 factors

### 14.3 Strategy Performance (Minimum Viable, per run)

Performance thresholds are evaluated **independently per run** against each
run's designated benchmark. All three runs must satisfy the bias control and
code quality criteria; performance thresholds apply to each run separately.

| Criterion | `csi300` | `csi500` | `combined` |
|---|---|---|---|
| IS Sharpe (vs. own benchmark) | > 1.0 | > 1.0 | > 1.0 |
| OOS Sharpe (best WF window) | > 0.5 | > 0.5 | > 0.5 |
| Beats benchmark in OOS windows | > 50% | > 50% | > 50% |
| Max Drawdown (OOS) | < 30% | < 35% | < 30% |
| Monthly turnover | < 40% | < 45% | < 45% |

CSI 500 has a slightly relaxed drawdown and turnover threshold because
mid-cap stocks are inherently more volatile and the universe turns over more
at index rebalancing events.

**Cross-run comparison requirement:** The report must include a table comparing
all three runs on identical metrics. If `combined` does not outperform
`csi300` on at least one key metric (Sharpe, Calmar, or IR), the 40/60
allocation rationale must be revisited and documented.

### 14.4 Robustness

- [ ] Sharpe remains positive when round-trip cost is doubled to 0.5%
- [ ] Sharpe degradation < 30% when regime threshold shifts ±10 percentile points
- [ ] Failure cases explicitly documented with quantified PnL impact

### 14.5 Code Quality

- [ ] No magic numbers in `src/`; all parameters in `config.yaml`
- [ ] All public functions have docstrings with input/output types
- [ ] `main.py --fetch --validate --backtest --report` runs end-to-end on a
  fresh environment with no manual steps
- [ ] README contains: motivation, setup instructions, key results table,
  equity curve image, and a "known limitations" section

---

## 15. Roadmap & Milestones

| Week | Module | Deliverable | Three-Run Notes |
|---|---|---|---|
| 1 | Data Layer + Universe | Parquet dataset built; PIT universe validated on 5 spot-check dates for **both** indices | Fetch union pool once; store `csi300/` and `csi500/` universe folders separately |
| 2 | Factor Library | 6 factors computed over **full union pool**; lookahead unit tests pass | Single computation pass; downstream runs slice as needed — no re-computation |
| 3 | Factor Validation | IC table (CSI 300 cols \| CSI 500 cols); at least 2 factors pass p < 0.05 **in both universes** | Side-by-side IC decay chart reveals large-cap vs. mid-cap factor behaviour |
| 4 | Regime Detection + Baseline Backtest | Equal-weight, no-regime baseline backtest runs end-to-end **for all three run_ids** | Verify cost model differences (CSI 300 vs CSI 500 slippage) are correctly applied |
| 5 | Risk Parity + Regime-Aware Strategy | Full strategy; all three runs meet performance minimums; combined allocation attribution computed | Confirm blended benchmark construction is correct before Walk-Forward |
| 6 | Walk-Forward + Report | OOS distribution chart per run; HTML report with cross-run comparison section; README complete | Walk-Forward runs sequentially for all three run_ids; aggregate OOS stats reported side by side |

---

*Document maintained in English. All code, comments, commit messages, and
variable names must be in English. Internal research notes may be in Chinese.*

*Version history:*
*v1.0 — initial specification.*
*v1.1 — confirmed decisions incorporated: long-only scope, Tushare Pro data*
*source (student verification pending), three-run design (csi300 / csi500 /*
*combined) with per-run benchmarks, blended benchmark for combined run,*
*updated Chart 1 (three equity curves), Chart 6 (universe composition),*
*expanded HTML report structure, per-run acceptance criteria table.*
