"""src/report/charts.py

Builds all 6 required Plotly charts as JSON-serialisable figures.
Each function returns a plotly.graph_objects.Figure.

Charts:
  1. three_run_equity_curve   — headline chart
  2. ic_time_series           — per-universe, per-factor IC with ±1σ band
  3. quantile_bar_chart       — Q1–Q5 annualised returns per factor/universe
  4. walk_forward_distribution — OOS performance box plots
  5. turnover_cost_chart      — monthly turnover + cumulative cost drag
  6. universe_composition     — eligible stock count over time
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import RESULTS, PROCESSED

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "csi300":    "#2563EB",    # blue
    "csi500":    "#16A34A",    # green
    "combined":  "#DC2626",    # red
    "bmark300":  "#93C5FD",    # light blue
    "bmark500":  "#86EFAC",    # light green
    "bmark_blend": "#FCA5A5",  # light red
    "mom_5d":    "#7C3AED",
    "mom_20d":   "#2563EB",
    "zscore_20d":"#16A34A",
    "rsi_14d":   "#DC2626",
    "regime_lo": "rgba(37,99,235,0.06)",
    "regime_hi": "rgba(220,38,38,0.06)",
    "neutral":   "#6B7280",
}

FACTOR_LABELS = {
    "mom_5d":     "Mom5d",
    "mom_20d":    "Mom20d",
    "zscore_20d": "ZScore20d",
    "rsi_14d":    "RSI14d",
}


# ---------------------------------------------------------------------------
# Helper: load daily_state and build equity index (base 100)
# ---------------------------------------------------------------------------

def _equity_series(run_id: str) -> pd.Series:
    path = RESULTS / run_id / "daily_state.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    eq = df["equity"]
    return eq / eq.iloc[0] * 100


def _drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak * 100   # in percent


def _benchmark_proxy(run_id: str, equity_index: pd.Index) -> pd.Series:
    """Equal-weight index return proxy, same length as equity."""
    returns = pd.read_parquet(PROCESSED / "returns.parquet")
    idx_ret = returns.mean(axis=1).reindex(equity_index).fillna(0)
    cum = np.exp(idx_ret.cumsum())
    return cum / cum.iloc[0] * 100


# ---------------------------------------------------------------------------
# Chart 1: Three-Run Equity Curve
# ---------------------------------------------------------------------------

def three_run_equity_curve() -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.04,
        subplot_titles=("Cumulative Return (base 100)", "Drawdown (%)"),
    )

    run_colors = {
        "csi300": C["csi300"], "csi500": C["csi500"], "combined": C["combined"]
    }
    bmark_colors = {
        "csi300": C["bmark300"], "csi500": C["bmark500"], "combined": C["bmark_blend"]
    }
    labels = {"csi300": "CSI 300 Strategy", "csi500": "CSI 500 Strategy",
              "combined": "Combined Strategy"}
    bmarks = {"csi300": "CSI 300 Benchmark", "csi500": "CSI 500 Benchmark",
              "combined": "Blended Benchmark"}

    for run_id in ("csi300", "csi500", "combined"):
        eq = _equity_series(run_id)
        if eq.empty:
            continue
        bmark = _benchmark_proxy(run_id, eq.index)
        dd    = _drawdown_series(eq)

        # Strategy equity
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=labels[run_id],
            line=dict(color=run_colors[run_id], width=2),
        ), row=1, col=1)

        # Benchmark equity (dashed)
        fig.add_trace(go.Scatter(
            x=bmark.index, y=bmark.values,
            name=bmarks[run_id],
            line=dict(color=bmark_colors[run_id], width=1.5, dash="dash"),
        ), row=1, col=1)

        # Drawdown
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=f"{run_id} DD",
            line=dict(color=run_colors[run_id], width=1),
            fill="tozeroy",
            fillcolor=run_colors[run_id].replace(")", ",0.10)").replace("rgb", "rgba"),
            showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        title="Three-Run Strategy vs. Benchmark — Full Period",
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Chart 2: IC Time Series (per factor, per universe)
# ---------------------------------------------------------------------------

def ic_time_series() -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("CSI 300 Universe", "CSI 500 Universe"),
        shared_yaxes=True,
    )

    for col_idx, run_id in enumerate(("csi300", "csi500"), start=1):
        path = RESULTS / run_id / "ic_series.parquet"
        if not path.exists():
            continue
        ic_df = pd.read_parquet(path)

        for fname, colour in C.items():
            if fname not in ic_df.columns:
                continue
            ic = ic_df[fname].dropna()
            roll_mean = ic.rolling(60, min_periods=20).mean()
            roll_std  = ic.rolling(60, min_periods=20).std()

            label = FACTOR_LABELS.get(fname, fname)

            # ±1σ band
            fig.add_trace(go.Scatter(
                x=pd.concat([roll_mean.index.to_series(), roll_mean.index.to_series()[::-1]]),
                y=pd.concat([roll_mean + roll_std, (roll_mean - roll_std)[::-1]]),
                fill="toself",
                fillcolor=colour.replace(")", ",0.12)").replace("#", "rgba(").replace("rgba(", "rgba("),
                line=dict(width=0),
                showlegend=False,
                name=f"{label} ±1σ",
            ), row=1, col=col_idx)

            # Mean line
            fig.add_trace(go.Scatter(
                x=roll_mean.index, y=roll_mean.values,
                name=label if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                line=dict(color=colour, width=1.8),
            ), row=1, col=col_idx)

        # Zero line
        fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1,
                      row=1, col=col_idx)

    fig.update_layout(
        title="60-Day Rolling Mean IC by Factor and Universe",
        height=420,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="IC (Spearman)", col=1)
    return fig


# ---------------------------------------------------------------------------
# Chart 3: Quantile Return Bar Chart
# ---------------------------------------------------------------------------

def quantile_bar_chart() -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("CSI 300 Universe", "CSI 500 Universe"),
        shared_yaxes=True,
    )

    for col_idx, run_id in enumerate(("csi300", "csi500"), start=1):
        path = RESULTS / run_id / "quantile_returns.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)

        for fname, colour in C.items():
            if fname not in df.index:
                continue
            row = df.loc[fname]
            q_vals = [row.get(f"Q{q}", np.nan) for q in range(1, 6)]

            fig.add_trace(go.Bar(
                name=FACTOR_LABELS.get(fname, fname),
                x=[f"Q{q}" for q in range(1, 6)],
                y=[v * 100 if not np.isnan(v) else None for v in q_vals],
                marker_color=colour,
                showlegend=(col_idx == 1),
                offsetgroup=fname,
            ), row=1, col=col_idx)

    fig.update_layout(
        title="Quintile Annualised Return by Factor — In-Sample",
        barmode="group",
        height=420,
        template="plotly_white",
        yaxis_title="Annualised Return (%)",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 4: Walk-Forward Performance Distribution
# ---------------------------------------------------------------------------

def walk_forward_distribution() -> go.Figure:
    metrics_to_show = ["sharpe_ratio", "annual_return", "max_drawdown"]
    metric_labels   = ["Sharpe Ratio", "Annual Return (%)", "Max Drawdown (%)"]
    multipliers     = [1, 100, 100]

    fig = make_subplots(
        rows=1, cols=len(metrics_to_show),
        subplot_titles=metric_labels,
    )

    run_colors = {"csi300": C["csi300"], "csi500": C["csi500"], "combined": C["combined"]}

    for col_idx, (metric, label, mult) in enumerate(
        zip(metrics_to_show, metric_labels, multipliers), start=1
    ):
        for run_id in ("csi300", "csi500", "combined"):
            path = RESULTS / run_id / "walk_forward.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if metric not in df.columns:
                continue

            vals = df[metric].dropna() * mult

            fig.add_trace(go.Box(
                y=vals.values,
                name=run_id.upper(),
                marker_color=run_colors[run_id],
                boxmean="sd",
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

        # Reference line for Sharpe=0
        if metric == "sharpe_ratio":
            fig.add_hline(y=0, line_dash="dot", line_color="grey", row=1, col=col_idx)

    fig.update_layout(
        title="Walk-Forward OOS Performance Distribution (13 Windows)",
        height=450,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 5: Turnover & Cost Analysis
# ---------------------------------------------------------------------------

def turnover_cost_chart() -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    run_colors = {"csi300": C["csi300"], "csi500": C["csi500"], "combined": C["combined"]}
    from src.backtest.costs import round_trip_cost_rate

    for run_id in ("csi300", "csi500", "combined"):
        path = RESULTS / run_id / "daily_state.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        monthly_to = df["turnover"].resample("ME").sum()

        cost_rate = round_trip_cost_rate(run_id)
        cum_cost  = (monthly_to * cost_rate).cumsum() * 100

        fig.add_trace(go.Bar(
            x=monthly_to.index, y=monthly_to.values * 100,
            name=f"{run_id.upper()} Turnover",
            marker_color=run_colors[run_id],
            opacity=0.6,
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=cum_cost.index, y=cum_cost.values,
            name=f"{run_id.upper()} Cum. Cost",
            line=dict(color=run_colors[run_id], width=2, dash="dash"),
        ), secondary_y=True)

    fig.update_yaxes(title_text="Monthly Turnover (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Cost Drag (%)", secondary_y=True)
    fig.update_layout(
        title="Monthly Turnover & Cumulative Transaction Cost Drag",
        barmode="group",
        height=420,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 6: Universe Composition Over Time
# ---------------------------------------------------------------------------

def universe_composition_chart() -> go.Figure:
    """Show eligible stock count per run over the full backtest period."""
    from src.data.universe import get_universe, rebalance_dates
    cfg = load_config_safe()

    reb_dates = rebalance_dates(
        cfg["walk_forward"]["is_start"],
        cfg["walk_forward"]["oos_end"],
    )

    fig = go.Figure()
    run_colors = {"csi300": C["csi300"], "csi500": C["csi500"]}

    for run_id, index_code in (("csi300", "000300.SH"), ("csi500", "000905.SH")):
        counts = []
        dates  = []
        for d in reb_dates[::3]:   # sample every 3 months to keep it fast
            try:
                n = len(get_universe(d, index_code))
                counts.append(n)
                dates.append(pd.Timestamp(d))
            except Exception:
                pass

        fig.add_trace(go.Scatter(
            x=dates, y=counts,
            name=run_id.upper(),
            line=dict(color=run_colors[run_id], width=2),
            fill="tozeroy" if run_id == "csi300" else None,
            fillcolor="rgba(37,99,235,0.07)",
        ))

    fig.update_layout(
        title="Eligible Universe Size Over Time (After PIT + Eligibility Filters)",
        height=380,
        template="plotly_white",
        yaxis_title="Number of Eligible Stocks",
        hovermode="x unified",
    )
    return fig


def load_config_safe() -> dict:
    from src.utils import load_config
    return load_config()
