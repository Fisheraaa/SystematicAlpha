<div align="center">

# Systematic Alpha Research & Trading Framework

**[English](#english-version) · [中文](#chinese-version-中文说明)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## English Version

### What This Project Is

A bias-controlled, regime-aware **long-only equity research platform** targeting A-share markets (CSI 300 / CSI 500, 2016–2024). The focus is methodological rigour over flattering backtest numbers — the framework is designed to answer honestly:

> *"Does this factor have statistically significant predictive power, and does that power survive realistic trading costs on unseen data?"*

---

### Key Results

#### Full-Period Backtest (IS: 2016–2020 · OOS: 2021–2024)

| Metric | CSI300 | CSI500 | Combined | Equal-Weight Market |
|---|---|---|---|---|
| Annual Return | +0.66% | −1.45% | −1.26% | +1.31% |
| Sharpe Ratio | **0.12** | −0.10 | −0.08 | 0.06 |
| Max Drawdown | **−28.4%** | −34.7% | −33.2% | −40.4% |

#### Walk-Forward OOS (15 independent windows, 2021–2024)

| | CSI300 | CSI500 | Combined |
|---|---|---|---|
| Mean Sharpe | −0.28 | −0.55 | −0.60 |
| Sharpe Std | 1.18 | 1.62 | 1.68 |
| Best Window | **+2.58** | **+2.67** | **+2.77** |
| Hit Rate (Sharpe > 0) | 27% | 33% | 33% |

#### Factor IC Summary (In-Sample, Spearman + t-test)

| Factor | CSI300 ICIR | CSI300 p-val | CSI500 ICIR | CSI500 p-val |
|---|---|---|---|---|
| Mom5d | −0.175 | < 0.001 | −0.220 | < 0.001 |
| Mom20d | −0.107 | < 0.001 | −0.177 | < 0.001 |
| ZScore20d | −0.112 | < 0.001 | −0.174 | < 0.001 |
| RSI14d | +0.105 | < 0.001 | +0.177 | < 0.001 |

> All four factors statistically significant (p < 0.001). Negative IC for momentum factors reflects A-share **short-term reversal** — factors are applied in reverse.

---

### Core Design Decisions

#### 1 · Survivorship Bias Eliminated
Every rebalance date uses the **historical constituent list** from Tushare's `index_weight` snapshots — not today's index. Using today's constituents inflates Sharpe by ~0.3–0.8.

```python
def get_universe(date, index_code):
    history = load("index_weight/{index_code}.parquet")
    snapshot = history[history["trade_date"] <= date]
    return snapshot.groupby("con_code").last().index.tolist()  # point-in-time only
```

#### 2 · Automated Look-Ahead Bias Detection
All rolling windows use `closed='left'` or `.shift(1)`. Enforced by automated unit tests — not manual review.

#### 3 · Realistic A-Share Cost Model

| Cost | Direction | Rate |
|---|---|---|
| Brokerage commission | Both | 0.025% |
| Stamp duty | Sell only | 0.05% (post Aug 2023) |
| Exchange fee (SH) | Both | 0.001% |
| Slippage (CSI300) | Both | 0.05% |
| Slippage (CSI500) | Both | 0.10% |

Round-trip: ~0.25% (CSI300) · ~0.30% (CSI500) · T+1 enforced

#### 4 · Walk-Forward Validation (15 OOS windows)
```
Train 24 months (rolling) → Test 6 months → Step 3 months
OOS period: 2021–2024
```
Reports a **distribution** of OOS performance — not a single number.

#### 5 · Three Independent Runs

| Run | Universe | Benchmark |
|---|---|---|
| `csi300` | CSI 300 PIT | CSI 300 TR Index |
| `csi500` | CSI 500 PIT | CSI 500 TR Index |
| `combined` | Union (40% / 60%) | Blended benchmark |

---

### Key Finding

A-share markets exhibit **short-term reversal** (not momentum continuation) — stocks with recent strong performance tend to underperform. All four predictive factors achieve p < 0.001 significance. However, pure long-only construction bears full market Beta: in the 2022–2024 bear market, factor selection reduced but could not eliminate losses. The CSI300 strategy achieved higher Sharpe (0.12 vs 0.06) and substantially lower drawdown (28% vs 40%) than the equal-weight market benchmark over the full period.

---

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/SystematicAlpha
cd SystematicAlpha
pip install -r requirements.txt
pip install python-dateutil

# Add your Tushare token to config/config.yaml
# (register free at tushare.pro — student verification recommended for index_weight access)

python main.py --fetch          # ~45 min first run
python main.py --process        # ~15 sec
python main.py --factors        # ~5 sec
python main.py --validate       # ~30 min  ← must run before backtest
python main.py --backtest       # ~15 min
python main.py --walkforward    # ~3 hours
python main.py --report         # generates report.html
```

All steps are idempotent — re-running skips completed work.

---

### Repository Structure

```
SystematicAlpha/
├── config/config.yaml          # all parameters — no magic numbers in code
├── src/
│   ├── data/                   # fetcher (Tushare), processor, PIT universe
│   ├── factors/                # 6 factors: Mom5d, Mom20d, ZScore20d, RSI14d, RVol20d, ATR14d
│   ├── validation/             # IC analysis (Spearman + t-test), quantile backtest
│   ├── strategy/               # regime detection (VolPct + ADX), composite signal, risk parity
│   ├── backtest/               # engine (T+1, A-share costs), metrics, walk-forward
│   └── report/                 # Plotly charts, bilingual HTML report
├── tests/                      # pytest: look-ahead bias, PIT universe, costs, metrics
├── main.py                     # CLI entry point
└── requirements.txt
```

---

### Known Limitations

- **Beta exposure**: Long-only design cannot hedge systematic market risk. Factor selection reduces idiosyncratic risk but not directional Beta.
- **Regime lag**: Volatility-percentile classifier has ~10–20 day detection delay during rapid market transitions.
- **CSI500 cost drag**: Monthly rebalancing at 0.30% round-trip produces ~3.6% annual cost drag. Quarterly rebalancing would reduce this to ~1.2%.
- **Index approximation**: Regime detection uses equal-weight stock average as index proxy.
- **CSI300 data gap**: Tushare constituent data begins 2016-01-29, shortening the effective IS period.

---

### Running Tests

```bash
pytest tests/ -v
```

Covers: look-ahead bias detection · PIT universe integrity · A-share cost arithmetic · performance metric formulae

---

### License

MIT. Not investment advice.

---
---

## Chinese Version (中文说明)

<details>
<summary><strong>点击展开中文说明</strong></summary>

### 项目简介

针对A股市场（沪深300 / 中证500，2016–2024年）的偏差控制型纯多头量化股票研究平台。核心理念是方法论严谨性优先于回测结果好看——框架的设计目标是诚实地回答：

> "这些因子是否具有统计显著的预测力？这种预测力能否在真实交易成本下的样本外数据中存活？"

---

### 核心结果

#### 全周期回测（样本内：2016–2020 · 样本外：2021–2024）

| 指标 | 沪深300策略 | 中证500策略 | 混合策略 | 等权市场基准 |
|---|---|---|---|---|
| 年化收益 | +0.66% | −1.45% | −1.26% | +1.31% |
| 夏普比率 | **0.12** | −0.10 | −0.08 | 0.06 |
| 最大回撤 | **−28.4%** | −34.7% | −33.2% | −40.4% |

#### Walk-Forward 样本外验证（15个独立窗口，2021–2024）

| | 沪深300 | 中证500 | 混合 |
|---|---|---|---|
| 平均夏普 | −0.28 | −0.55 | −0.60 |
| 夏普标准差 | 1.18 | 1.62 | 1.68 |
| 最高窗口夏普 | **+2.58** | **+2.67** | **+2.77** |
| 胜率（夏普>0） | 27% | 33% | 33% |

#### 因子IC汇总（样本内，Spearman秩相关 + t检验）

| 因子 | 沪深300 ICIR | p值 | 中证500 ICIR | p值 |
|---|---|---|---|---|
| 5日动量 | −0.175 | < 0.001 | −0.220 | < 0.001 |
| 20日动量 | −0.107 | < 0.001 | −0.177 | < 0.001 |
| 20日Z分 | −0.112 | < 0.001 | −0.174 | < 0.001 |
| RSI14日 | +0.105 | < 0.001 | +0.177 | < 0.001 |

> 全部4个因子统计显著（p < 0.001）。动量因子IC为负，反映A股的**短期反转效应**——因子按IC方向反向使用。

---

### 核心设计决策

**1. 消除幸存者偏差**：每个再平衡日使用Tushare历史成分股快照，而非当前指数成分股。用今天的成分股回看历史会虚增夏普比率约0.3–0.8。

**2. 自动化前视偏差检测**：所有滚动计算使用 `closed='left'` 或 `.shift(1)`，并通过单元测试自动验证，不依赖人工审查。

**3. A股真实成本模型**：买入约0.076%，卖出约0.126%（含印花税0.05%，2023年8月减半后），完整换仓约0.25%（沪深300）/ 0.30%（中证500），严格执行T+1规则。

**4. Walk-Forward验证（15个样本外窗口）**：训练24月→测试6月→步长3月，产出性能分布而非单一数字。

**5. 三条独立回测线**：沪深300 / 中证500 / 混合组合（40%/60%），各自对应专属基准评估。

---

### 核心发现

A股市场存在显著的**短期反转效应**而非动量延续效应——近期强势股倾向于跑输。全部4个预测因子均达到p<0.001显著水平。然而纯多头结构承受完整的市场Beta风险：在2022–2024年熊市中，因子选股能降低但无法消除亏损。全周期来看，沪深300策略的夏普比率（0.12）高于等权市场基准（0.06），最大回撤（28%）显著低于市场（40%）。

---

### 快速开始

```bash
git clone https://github.com/YOUR_USERNAME/SystematicAlpha
cd SystematicAlpha
pip install -r requirements.txt
pip install python-dateutil

# 在 config/config.yaml 填入 Tushare token
# （tushare.pro 免费注册，建议完成学生认证解锁 index_weight 接口）

python main.py --fetch          # 首次约45分钟
python main.py --process        # 约15秒
python main.py --factors        # 约5秒
python main.py --validate       # 约30分钟（必须在 --backtest 之前运行）
python main.py --backtest       # 约15分钟
python main.py --walkforward    # 约3小时
python main.py --report         # 生成 report.html（含中英切换）
```

---

### 已知局限性

- **市场Beta暴露**：纯多头无法对冲系统性市场风险，在单边下跌行情中会承受完整市场跌幅
- **体制信号滞后**：波动率百分位分类器在市场急速转向时有约10–20个交易日的检测延迟
- **中证500成本拖累**：月度再平衡叠加更高滑点，年化成本拖累约3.6%
- **指数近似**：体制检测使用等权股票均值代替真实指数序列
- **沪深300数据缺口**：Tushare成分股数据从2016年1月29日开始

</details>

---

*Data source: Tushare Pro. The `data/` directory is excluded from this repository (proprietary market data). · 数据来源：Tushare Pro。`data/` 目录已从仓库中排除（含版权行情数据）。*
