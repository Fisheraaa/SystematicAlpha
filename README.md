# Systematic Alpha Research & Trading Framework

A bias-controlled, regime-aware long-only equity research platform
targeting A-share (CSI 300 / CSI 500), built for reproducibility
and institutional rigour.

## Key Results

| Metric | CSI300 IS | CSI300 OOS | CSI500 IS | CSI500 OOS | Combined IS | Combined OOS |
|---|---|---|---|---|---|---|
| Sharpe Ratio | — | — | — | — | — | — |
| Max Drawdown | — | — | — | — | — | — |
| Annual Return | — | — | — | — | — | — |
| vs. Benchmark | — | — | — | — | — | — |

*(Fill in after running the backtest.)*

## Design Highlights

**1. Point-in-Time Universe (no survivorship bias)**
Every rebalance date uses the historical constituent snapshot from Tushare Pro,
not today's index membership. This eliminates survivorship bias, which can
inflate Sharpe ratios by 0.3–0.8 in naive backtests.

**2. Automated Look-Ahead Bias Detection**
`BaseFactor.validate_no_lookahead()` checks that factor values at t have
near-zero correlation with same-day returns. Unit tests in `tests/` run this
automatically for all 6 factors.

**3. Walk-Forward Validation (13 OOS windows)**
Rather than a single train/test split, the framework slides a 24-month
training window across the OOS period (2021–2024) in 3-month steps,
producing a distribution of OOS performance.

**4. Three Independent Runs**
- `csi300`: large-cap universe, CSI 300 TR benchmark
- `csi500`: mid-cap universe, CSI 500 TR benchmark
- `combined`: 40% CSI300 + 60% CSI500 sub-portfolios, blended benchmark

**5. Regime-Aware Signal**
Volatility-percentile + ADX dynamically weights momentum vs. mean-reversion
factors. In high-vol choppy markets, ZScore and RSI dominate; in low-vol
trending markets, Mom5d and Mom20d dominate.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/SystematicAlpha
cd SystematicAlpha
pip install -r requirements.txt

# Add your Tushare token to config/config.yaml
# Run full pipeline
python main.py --all
```

## Step-by-Step

```bash
python main.py --fetch          # pull data from Tushare (~30–60 min first run)
python main.py --process        # build processed panels
python main.py --factors        # compute all 6 factors
python main.py --validate       # IC analysis + quantile backtest
python main.py --backtest       # full backtest for all 3 runs
python main.py --walkforward    # 13-window walk-forward validation
python main.py --report         # generate report.html
```

## Repository Structure

```
SystematicAlpha/
├── config/config.yaml          # all parameters — no magic numbers in code
├── src/
│   ├── data/                   # fetcher, processor, universe
│   ├── factors/                # 6 factors with look-ahead validation
│   ├── validation/             # IC analysis, quantile backtest
│   ├── strategy/               # regime detection, composite signal, portfolio
│   ├── backtest/               # engine, costs, metrics, walk-forward
│   └── report/                 # chart builders, HTML report generator
├── tests/                      # pytest unit tests
├── main.py                     # CLI entry point
└── requirements.txt
```

## Known Limitations

- **Regime lag**: The volatility-percentile signal takes ~10–20 days to
  reclassify regime during rapid market transitions, causing excess drawdown.
- **T+1 gap risk**: Sell fills at next-day open; overnight gaps in fast markets
  may exceed the slippage allowance.
- **Factor crowding**: No crowding detection; correlated quant unwinds will
  cause drawdowns not captured by the risk model.
- **Index approximation**: Regime detection uses equal-weight stock average
  as an index proxy; a production system would use actual TR index series.

## A-Share Cost Model

| Cost | Direction | Rate |
|---|---|---|
| Brokerage commission | Both | 0.025% |
| Stamp duty | Sell only | 0.05% |
| Exchange fee (SH) | Both | 0.001% |
| Slippage (CSI300) | Both | 0.05% |
| Slippage (CSI500) | Both | 0.10% |

Round-trip: ~0.25% (CSI300) / ~0.30% (CSI500).
All reported metrics use double the baseline cost as a robustness check.
