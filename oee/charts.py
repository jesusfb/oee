"""Plotly figures in one consistent dark theme (accent #ff8a3d). UI-agnostic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BG = "#0b0c0f"
PANEL = "#12141a"
INK = "#ece9e4"
INK_2 = "#a9a59c"
GRID = "#262a33"
ACCENT = "#ff8a3d"
# fixed categorical order (entity -> colour never depends on rank)
SERIES = ["#ff8a3d", "#3987e5", "#199e70", "#9085e9", "#e87ba4", "#c98500", "#e66767", "#7f8794"]
FACTOR_COLOURS = {"Availability": "#3987e5", "Performance": "#9085e9", "Quality": "#e87ba4"}
WORLD_CLASS = 0.85
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"


def colour_map(keys) -> dict:
    return {k: SERIES[i % len(SERIES)] for i, k in enumerate(sorted(keys))}


def style(fig: go.Figure, title: str | None = None, height: int = 420, pct_y: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL, height=height,
        font=dict(family=FONT, color=INK, size=13),
        title=dict(text=title, font=dict(size=17, color=INK), x=0.01, xanchor="left") if title else None,
        margin=dict(l=60, r=30, t=60 if title else 25, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#1c1f27", bordercolor=ACCENT, font=dict(color=INK, family=FONT)),
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_2))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_2))
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    return fig


def oee_trend(daily: pd.DataFrame, group_col: str, total: pd.DataFrame | None = None, title="Daily OEE") -> go.Figure:
    fig = go.Figure()
    cmap = colour_map(daily[group_col].unique())
    for key, g in daily.groupby(group_col):
        g = g.sort_values("shift_date")
        fig.add_trace(go.Scatter(x=g["shift_date"], y=g["oee"], name=str(key), mode="lines+markers",
                                 line=dict(width=2, color=cmap[key]), marker=dict(size=7),
                                 hovertemplate="%{x|%a %d %b}<br>" + f"{key}: " + "%{y:.1%}<extra></extra>"))
    if total is not None and len(total):
        t = total.sort_values("shift_date")
        fig.add_trace(go.Scatter(x=t["shift_date"], y=t["oee"], name="All selected", mode="lines",
                                 line=dict(width=3, color=INK, dash="dot"),
                                 hovertemplate="%{x|%a %d %b}<br>All: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=WORLD_CLASS, line=dict(color=INK_2, width=1, dash="dash"),
                  annotation_text="World-class 85%", annotation_font_color=INK_2, annotation_position="top left")
    style(fig, title, pct_y=True)
    fig.update_layout(hovermode="x unified")
    lo = min(0.4, float(np.nanmin(daily["oee"])) - 0.05) if len(daily) else 0.0
    fig.update_yaxes(range=[max(0.0, lo), 1])
    return fig


def oee_by(df: pd.DataFrame, label_col: str, colour_col: str, title: str) -> go.Figure:
    d = df.dropna(subset=["oee"]).sort_values("oee")
    cmap = colour_map(d[colour_col].unique())
    fig = go.Figure()
    for key, g in d.groupby(colour_col):
        fig.add_trace(go.Bar(
            y=g[label_col], x=g["oee"], orientation="h", name=str(key), marker_color=cmap[key],
            text=[f"{v:.1%}" for v in g["oee"]], textposition="outside", textfont=dict(color=INK),
            customdata=np.stack([g["availability"], g["performance"], g["quality"]], axis=1),
            hovertemplate="<b>%{y}</b><br>OEE %{x:.1%}<br>A %{customdata[0]:.1%} · P %{customdata[1]:.1%}"
                          " · Q %{customdata[2]:.1%}<extra></extra>"))
    style(fig, title, height=max(320, 42 * len(d) + 90))
    fig.update_xaxes(tickformat=".0%", range=[0, 1.08])
    fig.update_yaxes(categoryorder="array", categoryarray=list(d[label_col]))
    return fig


def apq_grouped(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    cmap = colour_map(df[group_col].unique())
    metrics = ["availability", "performance", "quality", "oee"]
    fig = go.Figure()
    for key, g in df.groupby(group_col):
        vals = [g[m].iloc[0] for m in metrics]
        fig.add_trace(go.Bar(x=["Availability", "Performance", "Quality", "OEE"], y=vals, name=str(key),
                             marker_color=cmap[key], text=[f"{v:.1%}" for v in vals], textposition="outside",
                             textfont=dict(color=INK, size=11),
                             hovertemplate=f"{key}<br>" + "%{x}: %{y:.1%}<extra></extra>"))
    style(fig, title, pct_y=True)
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_yaxes(range=[0, 1.12])
    return fig


def waterfall(wf: pd.DataFrame, title="From planned time to fully productive time") -> go.Figure:
    """Waterfall built from floating bars so each loss can wear its OEE-factor colour."""
    labels = [s.replace(" (", "<br>(") for s in wf["step"]]
    bases, heights, colours, level = [], [], [], 0.0
    for h, kind, factor in zip(wf["hours"], wf["kind"], wf["factor"]):
        if kind == "total":
            bases.append(0.0); heights.append(h); colours.append(ACCENT); level = h
        else:
            level += h  # h is negative
            bases.append(level); heights.append(-h); colours.append(FACTOR_COLOURS.get(factor, "#3987e5"))
    pct = [hh / wf["hours"].iloc[0] if wf["hours"].iloc[0] else np.nan for hh in heights]
    fig = go.Figure(go.Bar(
        x=labels, y=heights, base=bases, marker_color=colours, showlegend=False,
        text=[f"{hh:,.0f} h" for hh in heights], textposition="outside", textfont=dict(color=INK),
        customdata=pct, hovertemplate="%{x}<br>%{y:,.1f} h (%{customdata:.1%} of planned)<extra></extra>"))
    for name, c in FACTOR_COLOURS.items():
        fig.add_trace(go.Scatter(x=[labels[0]], y=[None], mode="markers", name=f"{name} loss",
                                 marker=dict(size=12, color=c, symbol="square")))
    style(fig, title, height=480)
    fig.update_yaxes(title_text="hours", title_font=dict(color=INK_2), range=[0, wf["hours"].iloc[0] * 1.1])
    fig.update_layout(bargap=0.18)
    return fig


def pareto(p: pd.DataFrame, cat_col: str, value_col: str, title: str, unit: str = "min") -> go.Figure:
    """Pareto on ONE axis: bars are % of total, the line is cumulative %."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=p[cat_col], y=p["pct"], name="Share of total", marker_color=ACCENT,
                         customdata=p[value_col],
                         hovertemplate="%{x}<br>%{customdata:,.0f} " + unit + " (%{y:.1%})<extra></extra>"))
    fig.add_trace(go.Scatter(x=p[cat_col], y=p["cum_pct"], name="Cumulative", mode="lines+markers",
                             line=dict(color=INK, width=2), marker=dict(size=8, color=INK),
                             hovertemplate="%{x}<br>cumulative %{y:.1%}<extra></extra>"))
    fig.add_hline(y=0.8, line=dict(color=INK_2, width=1, dash="dash"), annotation_text="80%",
                  annotation_font_color=INK_2, annotation_position="top left")
    style(fig, title, pct_y=True)
    fig.update_yaxes(range=[0, 1.05])
    return fig


def heatmap(pivot: pd.DataFrame, title: str, fmt: str = ",.0f", pct: bool = False, reverse: bool = False) -> go.Figure:
    scale = [[0, "#1a1c22"], [0.5, "#a3521f"], [1, ACCENT]]
    if reverse:
        scale = [[0, ACCENT], [0.5, "#a3521f"], [1, "#1a1c22"]]
    z = pivot.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in pivot.columns], y=[str(i) for i in pivot.index], colorscale=scale,
        xgap=2, ygap=2, colorbar=dict(tickformat=".0%" if pct else fmt, tickfont=dict(color=INK_2)),
        hovertemplate="%{y} · %{x}<br>%{z:" + (".1%" if pct else fmt) + "}<extra></extra>"))
    style(fig, title, height=max(330, 34 * len(pivot) + 120))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def mtbf_mttr_scatter(df: pd.DataFrame, title="Reliability map: MTBF vs MTTR") -> go.Figure:
    d = df.dropna(subset=["mtbf_h", "mttr_min"])
    cmap = colour_map(d["line_id"].unique())
    fig = go.Figure()
    for key, g in d.groupby("line_id"):
        fig.add_trace(go.Scatter(
            x=g["mtbf_h"], y=g["mttr_min"], mode="markers+text", name=str(key), text=g["machine_id"],
            textposition="top center", textfont=dict(color=INK_2, size=11),
            marker=dict(size=14, color=cmap[key], line=dict(color=BG, width=2)), customdata=g["failures"],
            hovertemplate="<b>%{text}</b><br>MTBF %{x:.1f} h<br>MTTR %{y:.1f} min<br>%{customdata} failures"
                          "<extra></extra>"))
    style(fig, title, height=440)
    fig.update_xaxes(title_text="MTBF (run hours between failures) → better")
    fig.update_yaxes(title_text="MTTR (minutes to repair) ↓ better")
    return fig


def box_by(df: pd.DataFrame, group_col: str, value_col: str, title: str) -> go.Figure:
    cmap = colour_map(df[group_col].unique())
    fig = go.Figure()
    for key, g in df.groupby(group_col):
        fig.add_trace(go.Box(y=g[value_col], name=str(key), marker_color=cmap[key], boxpoints="all", jitter=0.4,
                             pointpos=0, marker=dict(size=5, opacity=0.6), line=dict(width=2),
                             hovertemplate=f"{key}<br>" + "%{y:.1%}<extra></extra>"))
    style(fig, title, pct_y=True)
    fig.update_layout(showlegend=False)
    return fig


def what_if_bars(base, scen, title="Baseline vs scenario") -> go.Figure:
    labels = ["Availability", "Performance", "Quality", "OEE"]
    b = [base.availability, base.performance, base.quality, base.oee]
    s = [scen.availability, scen.performance, scen.quality, scen.oee]
    fig = go.Figure([
        go.Bar(x=labels, y=b, name="Baseline", marker_color="#7f8794", text=[f"{v:.1%}" for v in b],
               textposition="outside", textfont=dict(color=INK)),
        go.Bar(x=labels, y=s, name="Scenario", marker_color=ACCENT, text=[f"{v:.1%}" for v in s],
               textposition="outside", textfont=dict(color=INK)),
    ])
    style(fig, title, pct_y=True, height=400)
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1.12])
    return fig
