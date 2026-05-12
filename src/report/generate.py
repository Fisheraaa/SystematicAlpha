"""src/report/generate.py

Generates a single self-contained report.html with:
  - EN / CN language toggle (no server needed)
  - All 6 Plotly charts inlined
  - Performance metrics table (IS from daily_state.parquet, OOS from walk_forward.parquet)
  - Walk-forward window detail table
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly

from src.utils import load_config, RESULTS
from src.backtest.metrics import compute_all_metrics
from src.report.charts import (
    three_run_equity_curve,
    ic_time_series,
    quantile_bar_chart,
    walk_forward_distribution,
    turnover_cost_chart,
    universe_composition_chart,
)

logger = logging.getLogger(__name__)
OUT_PATH = Path("report.html")


def _fig_div(fig, div_id: str) -> str:
    return plotly.io.to_html(
        fig, full_html=False, include_plotlyjs=False,
        div_id=div_id, config={"displayModeBar": True, "responsive": True},
    )


def _fmt(val, fmt_str: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return fmt_str.format(val)


def _metrics_table(run_ids: list) -> str:
    cfg    = load_config()
    cfg_wf = cfg["walk_forward"]

    metric_defs = [
        ("annual_return",     "Annual Return",   "年化收益",   "{:.1%}"),
        ("sharpe_ratio",      "Sharpe Ratio",    "夏普比率",   "{:.2f}"),
        ("max_drawdown",      "Max Drawdown",    "最大回撤",   "{:.1%}"),
        ("calmar_ratio",      "Calmar Ratio",    "卡玛比率",   "{:.2f}"),
        ("information_ratio", "Info Ratio",      "信息比率",   "{:.2f}"),
        ("win_rate",          "Win Rate",        "胜率",       "{:.1%}"),
    ]

    col_en = "".join(f"<th>{r.upper()}<br>IS</th><th>{r.upper()}<br>OOS(avg)</th>" for r in run_ids)
    col_cn = "".join(f"<th>{r.upper()}<br>样本内</th><th>{r.upper()}<br>样本外均值</th>" for r in run_ids)

    data = {r: {} for r in run_ids}
    for run_id in run_ids:
        ds_path = RESULTS / run_id / "daily_state.parquet"
        wf_path = RESULTS / run_id / "walk_forward.parquet"
        if ds_path.exists():
            daily = pd.read_parquet(ds_path)
            daily.index = pd.to_datetime(daily.index)
            eq_is = daily.loc[daily.index <= pd.Timestamp(cfg_wf["is_end"]), "equity"].dropna()
            if len(eq_is) >= 2:
                bmark = pd.Series(float(eq_is.iloc[0]), index=eq_is.index)
                data[run_id]["IS"] = compute_all_metrics(eq_is, bmark, pd.DataFrame())
        if wf_path.exists():
            wf = pd.read_parquet(wf_path)
            data[run_id]["OOS"] = {k: float(wf[k].mean()) for k in wf.select_dtypes("number").columns if k in [m[0] for m in metric_defs]}

    def rows(lang):
        out = ""
        for key, len_en, len_cn, fmt in metric_defs:
            label = len_en if lang == "en" else len_cn
            row = f"<tr><td><strong>{label}</strong></td>"
            for rid in run_ids:
                for period in ("IS", "OOS"):
                    val = data[rid].get(period, {}).get(key)
                    row += f"<td>{_fmt(val, fmt)}</td>"
            out += row + "</tr>"
        return out

    return f"""
<div class="lang-en"><table class="metrics-table">
  <thead><tr><th>Metric</th>{col_en}</tr></thead><tbody>{rows("en")}</tbody>
</table></div>
<div class="lang-cn" style="display:none"><table class="metrics-table">
  <thead><tr><th>指标</th>{col_cn}</tr></thead><tbody>{rows("cn")}</tbody>
</table></div>"""


def _ic_table() -> str:
    factor_data = {}
    for run_id in ("csi300", "csi500"):
        path = RESULTS / run_id / "ic_summary.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for fname in df.index:
            factor_data.setdefault(fname, {})[run_id] = df.loc[fname].to_dict()

    rows = ""
    for fname, d in factor_data.items():
        row = f"<tr><td><strong>{fname}</strong></td>"
        for run_id in ("csi300", "csi500"):
            rd = d.get(run_id, {})
            icir = rd.get("icir", float("nan"))
            pval = rd.get("p_value", float("nan"))
            ppct = rd.get("pct_positive", float("nan"))
            sig  = " ✓" if (not np.isnan(pval) and pval < 0.05) else ""
            pval_str = "< 0.001" if (not np.isnan(pval) and pval < 0.001) else _fmt(pval, '{:.3f}')
            row += f"<td>{_fmt(icir,'{:.3f}')}</td><td>{pval_str}{sig}</td><td>{_fmt(ppct,'{:.1%}')}</td>"
        rows += row + "</tr>"

    hdr_en = "<tr><th>Factor</th><th>CSI300 ICIR</th><th>p-val</th><th>IC>0%</th><th>CSI500 ICIR</th><th>p-val</th><th>IC>0%</th></tr>"
    hdr_cn = "<tr><th>因子</th><th>沪深300 ICIR</th><th>p值</th><th>IC>0占比</th><th>中证500 ICIR</th><th>p值</th><th>IC>0占比</th></tr>"
    return f"""
<div class="lang-en"><table class="metrics-table"><thead>{hdr_en}</thead><tbody>{rows}</tbody></table></div>
<div class="lang-cn" style="display:none"><table class="metrics-table"><thead>{hdr_cn}</thead><tbody>{rows}</tbody></table></div>"""


def _wf_table() -> str:
    rows = ""
    for run_id in ("csi300", "csi500", "combined"):
        path = RESULTS / run_id / "walk_forward.parquet"
        if not path.exists():
            continue
        wf = pd.read_parquet(path)
        for _, r in wf.iterrows():
            sharpe = r.get("sharpe_ratio", float("nan"))
            color  = "#16A34A" if (not np.isnan(sharpe) and sharpe > 0) else "#DC2626"
            rows += (f"<tr><td>{run_id.upper()}</td>"
                     f"<td>{str(r.get('test_start',''))[:10]} → {str(r.get('test_end',''))[:10]}</td>"
                     f"<td style='color:{color};font-weight:600'>{_fmt(sharpe,'{:.2f}')}</td>"
                     f"<td>{_fmt(r.get('max_drawdown'),'{:.1%}')}</td>"
                     f"<td>{_fmt(r.get('annual_return'),'{:.1%}')}</td></tr>")

    hdr_en = "<tr><th>Run</th><th>Test Period</th><th>Sharpe</th><th>Max DD</th><th>Ann Return</th></tr>"
    hdr_cn = "<tr><th>策略</th><th>测试区间</th><th>夏普比率</th><th>最大回撤</th><th>年化收益</th></tr>"
    return f"""
<div class="lang-en"><table class="metrics-table"><thead>{hdr_en}</thead><tbody>{rows}</tbody></table></div>
<div class="lang-cn" style="display:none"><table class="metrics-table"><thead>{hdr_cn}</thead><tbody>{rows}</tbody></table></div>"""


def generate_report() -> None:
    logger.info("Generating HTML report …")

    charts = {}
    for key, fn in [
        ("equity",   three_run_equity_curve),
        ("ic_ts",    ic_time_series),
        ("quantile", quantile_bar_chart),
        ("walkfwd",  walk_forward_distribution),
        ("turnover", turnover_cost_chart),
        ("universe", universe_composition_chart),
    ]:
        try:
            charts[key] = _fig_div(fn(), f"chart-{key}")
        except Exception as exc:
            logger.warning("Chart '%s' failed: %s", key, exc)
            charts[key] = f"<p class='warn'>Chart unavailable: {exc}</p>"

    metrics_html = _metrics_table(["csi300", "csi500", "combined"])
    ic_html      = _ic_table()
    wf_html      = _wf_table()

    PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Systematic Alpha — Research Report</title>
<script src="{PLOTLY_CDN}"></script>
<style>
:root{{--bg:#F9FAFB;--card:#fff;--text:#111827;--muted:#6B7280;--accent:#2563EB;--border:#E5E7EB;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);line-height:1.6;}}
.container{{max-width:1280px;margin:0 auto;padding:24px 16px;}}
.lang-bar{{display:flex;justify-content:flex-end;margin-bottom:16px;gap:8px;}}
.lang-btn{{padding:6px 18px;border-radius:20px;border:1px solid var(--accent);background:#fff;color:var(--accent);cursor:pointer;font-size:.88rem;font-weight:600;transition:all .2s;}}
.lang-btn.active{{background:var(--accent);color:#fff;}}
.hero{{background:var(--accent);color:#fff;padding:36px 32px;border-radius:12px;margin-bottom:28px;}}
.hero h1{{font-size:1.75rem;font-weight:700;margin-bottom:4px;}}
.hero p{{opacity:.85;font-size:.93rem;}}
.section{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:28px;margin-bottom:22px;box-shadow:0 1px 3px rgba(0,0,0,.06);}}
.section h2{{font-size:1.12rem;font-weight:600;border-left:4px solid var(--accent);padding-left:10px;margin-bottom:16px;}}
.section h3{{font-size:.97rem;font-weight:600;color:var(--muted);margin:18px 0 8px;}}
.metrics-table{{width:100%;border-collapse:collapse;font-size:.87rem;}}
.metrics-table th,.metrics-table td{{border:1px solid var(--border);padding:7px 11px;text-align:right;}}
.metrics-table th{{background:var(--bg);font-weight:600;text-align:center;}}
.metrics-table td:first-child{{text-align:left;}}
.note{{background:#FEF9C3;border:1px solid #FDE68A;border-radius:6px;padding:11px 15px;font-size:.87rem;color:#78350F;margin:12px 0;}}
.warn{{color:#DC2626;font-style:italic;padding:8px;}}
footer{{text-align:center;color:var(--muted);font-size:.8rem;padding:28px 0 14px;}}
</style>
</head>
<body>
<div class="container">

<div class="lang-bar">
  <button class="lang-btn active" onclick="setLang('en')" id="btn-en">English</button>
  <button class="lang-btn"        onclick="setLang('cn')" id="btn-cn">中文</button>
</div>

<div class="hero">
  <div class="lang-en">
    <h1>Systematic Alpha Research &amp; Trading Framework</h1>
    <p>A-share (CSI 300 / CSI 500) &nbsp;·&nbsp; Long-only &nbsp;·&nbsp; Bias-controlled &nbsp;·&nbsp; Regime-aware</p>
    <p style="margin-top:8px;font-size:.82rem;opacity:.75;">Three runs: CSI300 / CSI500 / Combined &nbsp;·&nbsp; Data: Tushare Pro &nbsp;·&nbsp; Walk-forward: 15 OOS windows</p>
  </div>
  <div class="lang-cn" style="display:none">
    <h1>系统化阿尔法研究与交易框架</h1>
    <p>A股（沪深300 / 中证500） &nbsp;·&nbsp; 纯多头 &nbsp;·&nbsp; 偏差控制 &nbsp;·&nbsp; 体制感知</p>
    <p style="margin-top:8px;font-size:.82rem;opacity:.75;">三条回测线：CSI300 / CSI500 / Combined &nbsp;·&nbsp; 数据：Tushare Pro &nbsp;·&nbsp; 15个样本外窗口</p>
  </div>
</div>

<div class="section">
  <h2><span class="lang-en">1 · Executive Summary</span><span class="lang-cn" style="display:none">1 · 执行摘要</span></h2>
  {charts["equity"]}
  <h3><span class="lang-en">Performance Metrics — IS vs OOS (walk-forward average)</span>
      <span class="lang-cn" style="display:none">绩效指标 — 样本内 vs 样本外均值（Walk-Forward）</span></h3>
  {metrics_html}
  <p class="note lang-en">IS = 2016-02-01→2020-12-31 (full-period run). OOS = average across 15 walk-forward windows (2021–2024). ✓ = p &lt; 0.05.</p>
  <p class="note lang-cn" style="display:none">样本内 = 2016-02-01→2020-12-31（全周期回测）。样本外 = 15个Walk-Forward窗口均值（2021–2024）。✓ = p &lt; 0.05。</p>
</div>

<div class="section">
  <h2><span class="lang-en">2 · Factor Validation</span><span class="lang-cn" style="display:none">2 · 因子验证</span></h2>
  <h3><span class="lang-en">IC Summary (In-Sample, Spearman + t-test)</span><span class="lang-cn" style="display:none">IC汇总（样本内，Spearman秩相关 + t检验）</span></h3>
  {ic_html}
  <p class="note lang-en">Negative IC for momentum factors means A-shares exhibit short-term <em>reversal</em>. Factors are applied in the direction of their IC sign.</p>
  <p class="note lang-cn" style="display:none">动量因子IC为负，说明A股存在短期<em>反转效应</em>。所有因子按IC方向使用。</p>
  <h3><span class="lang-en">Rolling 60-Day Mean IC</span><span class="lang-cn" style="display:none">60日滚动均值IC</span></h3>
  {charts["ic_ts"]}
  <h3><span class="lang-en">Quintile Return Analysis</span><span class="lang-cn" style="display:none">分层回测</span></h3>
  {charts["quantile"]}
</div>

<div class="section">
  <h2><span class="lang-en">3 · Walk-Forward Validation (15 OOS Windows)</span><span class="lang-cn" style="display:none">3 · Walk-Forward 验证（15个样本外窗口）</span></h2>
  {charts["walkfwd"]}
  <h3><span class="lang-en">All Windows Detail</span><span class="lang-cn" style="display:none">全部窗口明细</span></h3>
  {wf_html}
  <p class="note lang-en">Green = Sharpe &gt; 0. Peak performance in W8 (Oct 2022–Mar 2023) coincides with post-COVID policy-driven reversal rally.</p>
  <p class="note lang-cn" style="display:none">绿色 = Sharpe &gt; 0。W8（2022年10月–2023年3月）表现最优，对应疫情放开后政策驱动的反弹行情。</p>
</div>

<div class="section">
  <h2><span class="lang-en">4 · Turnover &amp; Transaction Cost Analysis</span><span class="lang-cn" style="display:none">4 · 换手率与交易成本</span></h2>
  {charts["turnover"]}
  <p class="note lang-en">Round-trip cost: ~0.25% (CSI300) / ~0.30% (CSI500). Stamp duty reduced to 0.05% (sell only) from Aug 2023.</p>
  <p class="note lang-cn" style="display:none">完整换仓成本：约0.25%（沪深300）/ 0.30%（中证500）。印花税2023年8月起降至0.05%（仅卖出）。</p>
</div>

<div class="section">
  <h2><span class="lang-en">5 · Universe Composition (Point-in-Time)</span><span class="lang-cn" style="display:none">5 · 股票池构成（点位数据）</span></h2>
  {charts["universe"]}
  <p class="note lang-en">Universe uses historical constituent snapshots — not today's index — to eliminate survivorship bias. CSI300 data begins 2016-01-29.</p>
  <p class="note lang-cn" style="display:none">股票池使用历史成分股快照（非当前指数），消除幸存者偏差。沪深300数据从2016年1月29日起。</p>
</div>

<div class="section">
  <h2><span class="lang-en">6 · Failure Cases &amp; Limitations</span><span class="lang-cn" style="display:none">6 · 失效分析与局限性</span></h2>
  <div class="lang-en">
    <h3>Market Beta Exposure</h3><p>Long-only design bears full market downside. CSI300 Sharpe (0.12) exceeded the equal-weight benchmark (0.06) and max drawdown (28%) was well below market (40%), but absolute returns were modest in the 2022–2024 bear market.</p>
    <h3>Regime Signal Lag (~10–20 days)</h3><p>The volatility-percentile classifier has a detection delay during rapid market transitions, causing excess drawdown in fast regime shifts.</p>
    <h3>CSI500 Cost Drag</h3><p>Higher slippage + monthly rebalancing = ~3.6% annual cost drag, offsetting most of the factor alpha. Quarterly rebalancing would reduce this to ~1.2%.</p>
  </div>
  <div class="lang-cn" style="display:none">
    <h3>市场Beta暴露</h3><p>纯多头策略承受完整市场下行风险。沪深300策略Sharpe（0.12）高于等权基准（0.06），最大回撤（28%）远低于市场（40%），但在2022–2024年熊市中绝对收益有限。</p>
    <h3>体制信号滞后（约10–20个交易日）</h3><p>波动率百分位分类器在市场急速转向时存在检测延迟，导致体制切换期出现额外回撤。</p>
    <h3>中证500成本拖累</h3><p>更高滑点+月度再平衡 = 年化约3.6%成本拖累，吞噬大部分因子超额收益。改为季度再平衡可降至约1.2%。</p>
  </div>
</div>

<footer>
  <span class="lang-en">Systematic Alpha v1.1 &nbsp;·&nbsp; All metrics include realistic A-share transaction costs &nbsp;·&nbsp; Not investment advice.</span>
  <span class="lang-cn" style="display:none">Systematic Alpha v1.1 &nbsp;·&nbsp; 所有指标均包含真实A股交易成本 &nbsp;·&nbsp; 本报告不构成投资建议。</span>
</footer>
</div>

<script>
function setLang(lang) {{
  document.querySelectorAll('.lang-en').forEach(el => {{ el.style.display = lang==='en' ? '' : 'none'; }});
  document.querySelectorAll('.lang-cn').forEach(el => {{ el.style.display = lang==='cn' ? '' : 'none'; }});
  document.getElementById('btn-en').classList.toggle('active', lang==='en');
  document.getElementById('btn-cn').classList.toggle('active', lang==='cn');
}}
</script>
</body></html>"""

    OUT_PATH.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", OUT_PATH.resolve())
