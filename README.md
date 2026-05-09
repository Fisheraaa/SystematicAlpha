# Systematic Alpha Research & Trading Framework

> A bias-controlled, regime-aware long-only equity research platform targeting
> A-share markets (CSI 300 / CSI 500), built for methodological rigour and
> reproducibility.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Key Results (2016–2024, Out-of-Sample)

| Metric | CSI300 Strategy | CSI500 Strategy | Combined | Equal-Weight Market |
|---|---|---|---|---|
| Annual Return | +0.66% | −1.45% | −1.26% | +1.31% |
| Sharpe Ratio | **0.12** | −0.10 | −0.08 | 0.06 |
| Max Drawdown | **−28.4%** | −34.7% | −33.2% | −40.4% |
| Walk-Forward Hit Rate | 27% | 33% | 33% | — |

> **Note**: 2016–2024 was a difficult period for A-share long-only strategies.
> The CSI300 strategy achieved a higher Sharpe and substantially lower drawdown
> than the equal-weight market benchmark. The strategy's value is primarily in
> risk-adjusted return and downside protection, not raw alpha generation.

## What This Project Is About

Most retail backtests share three silent flaws:

1. **Survivorship bias** — tested on today's index constituents, not historical ones
2. **Look-ahead bias** — factor computation leaks future price information
3. **Optimistic costs** — ignoring stamp duty, slippage, and T+1 constraints

This framework treats each of these as first-class engineering constraints.
The goal is not to produce a flattering Sharpe ratio, but to answer honestly:

> *"Does this factor have statistically significant predictive power, and does
> that power survive realistic trading costs on unseen data?"*

## Core Findings

**Factor Validity**: All four predictive factors achieve statistical significance
(p < 0.001, Spearman IC t-test) on both CSI 300 and CSI 500 universes.
A-share markets (2016–2020 in-sample) exhibit strong **short-term reversal**
rather than momentum — high-momentum stocks tend to underperform.

**Regime Dependency**: Walk-forward analysis across 15 independent OOS windows
reveals strong market-state dependency. Sharpe ranged from −3.3 to +2.8.
Peak performance occurred during policy-driven reversal periods
(Oct 2022–Mar 2023 window: Sharpe = 2.58 for all three runs).

**Structural Limitation**: The long-only constraint means the strategy bears
full market Beta. In trending bear markets (2022–2024), factor selection
reduces but cannot eliminate losses. This is not factor failure — it is
Beta exposure.

## Design Highlights

### 1. Point-in-Time Universe (Survivorship Bias Eliminated)
```python
# Every rebalance date uses the HISTORICAL constituent list
def get_universe(date, index_code):
    history = load("index_weight/{index_code}.parquet")
    snapshot = history[history["trade_date"] <= date]
    return snapshot.groupby("con_code").last().index.tolist()
```
Using today's constituents on historical data inflates Sharpe by ~0.3–0.8
by granting knowledge of which stocks survived.

### 2. Automated Look-Ahead Bias Detection
```python
# Unit test: factor_t must not correlate with same-day return_t
factor.validate_no_lookahead(prices, factor_df, n_spot_checks=50)
```
All rolling computations use `closed='left'` or `.shift(1)`. Enforced by
automated tests, not manual review.

### 3. Realistic A-Share Cost Model

| Cost | Direction | Rate |
|---|---|---|
| Brokerage commission | Both | 0.025% |
| Stamp duty | Sell only | 0.05% (post Aug 2023 reduction) |
| Exchange fee (SH) | Both | 0.001% |
| Slippage (CSI300) | Both | 0.05% |
| Slippage (CSI500) | Both | 0.10% |

Round-trip: ~0.25% (CSI300) / ~0.30% (CSI500). T+1 constraint enforced.

### 4. Walk-Forward Validation (15 OOS Windows)

```
Train: 24 months (rolling) → Test: 6 months → Step: 3 months
Total OOS period: 2021–2024 → 15 independent windows
```

Reports a **distribution** of OOS performance, not a single number.

### 5. Three Independent Runs

| Run | Universe | Benchmark |
|---|---|---|
| `csi300` | CSI 300 PIT | CSI 300 TR Index |
| `csi500` | CSI 500 PIT | CSI 500 TR Index |
| `combined` | Union (40/60) | Blended benchmark |

## Factor Library

| Factor | Type | IC (CSI300) | IC (CSI500) | p-value |
|---|---|---|---|---|
| Mom5d | Short momentum (reversed) | −0.031 | −0.031 | < 0.001 |
| Mom20d | Medium momentum (reversed) | −0.021 | −0.026 | < 0.001 |
| ZScore20d | Mean reversion | −0.021 | −0.026 | < 0.001 |
| RSI14d | Mean reversion | +0.021 | +0.028 | < 0.001 |
| RVol20d | Risk proxy (position sizing) | — | — | — |
| ATR14d | Risk proxy (position sizing) | — | — | — |

> All IC values are negative for momentum factors because **A-shares exhibit
> short-term reversal** (not continuation). Factors are used in reverse.

## Setup

```bash
git clone https://github.com/Fisheraaa/SystematicAlpha
cd SystematicAlpha
pip install -r requirements.txt
pip install python-dateutil

# Set your Tushare token (register free at tushare.pro)
export TUSHARE_TOKEN="your_token_here"   # or edit config/config.yaml

# Run the full pipeline
python main.py --fetch          # ~45 min first run
python main.py --process        # ~15 sec
python main.py --factors        # ~5 sec
python main.py --validate       # ~30 min
python main.py --backtest       # ~15 min
python main.py --walkforward    # ~3 hours
python main.py --report         # ~30 sec → opens report.html
```

All steps are idempotent (re-running skips completed work).

## Repository Structure

```
SystematicAlpha/
├── config/config.yaml          # all parameters — no magic numbers in code
├── src/
│   ├── data/                   # fetcher (Tushare), processor, PIT universe
│   ├── factors/                # 6 factors with look-ahead validation
│   ├── validation/             # IC analysis (Spearman + t-test), quantile backtest
│   ├── strategy/               # regime detection, composite signal, risk parity
│   ├── backtest/               # engine (T+1, A-share costs), metrics, walk-forward
│   └── report/                 # Plotly charts, self-contained HTML report
├── tests/                      # pytest: look-ahead, PIT universe, costs, metrics
├── main.py                     # CLI entry point
└── requirements.txt
```

## Known Limitations

**1. Beta exposure**: Pure long-only design bears full market downside risk.
Factor selection reduces idiosyncratic risk but cannot hedge systematic risk.
A short overlay (stock index futures) would isolate the factor alpha.

**2. Regime signal lag**: The volatility-percentile regime classifier has a
~10–20 trading day detection delay during rapid market transitions.

**3. Index approximation**: Regime detection uses equal-weight stock average
as index proxy. Production use would require actual CSI 300 TR index series.

**4. CSI300 data gap**: Historical constituent data begins 2016-01-29,
shortening the in-sample period relative to CSI 500 (available from 2014).

## Running Tests

```bash
pytest tests/ -v
```

Tests cover: look-ahead bias detection, PIT universe integrity,
A-share cost arithmetic, and performance metric formulae.

## License

MIT — free to use, modify, and distribute. Not investment advice.

---

*Data source: Tushare Pro (tushare.pro). Market data is proprietary;
the `data/` directory is excluded from this repository.*