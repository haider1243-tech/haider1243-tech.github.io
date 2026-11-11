"""Plotly figure builders for the Streamlit app (pure: snapshot dicts in, figures out).

Colour follows the dataviz reference palette (validated with its CVD /
contrast checker in light and dark mode):
* one entity per chart -> categorical slot 1 (blue); a 2nd series -> slot 2 (orange)
* SHAP contributions are signed -> diverging blue (raises the forecast) /
  red (lowers it), always with a signed text label so colour is never alone
* text uses ink tokens, never the series colour; grid/axes are recessive
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTES = {
    "light": dict(s1="#2a78d6", s2="#eb6834", pos="#2a78d6", neg="#e34948",
                  ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", neutral="#f0efec"),
    "dark": dict(s1="#3987e5", s2="#d95926", pos="#3987e5", neg="#e66767",
                 ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", neutral="#383835"),
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

FEATURE_LABELS = {
    "ret_21d": "1-month return", "ret_63d": "3-month return", "ret_252d": "12-month return",
    "vol_63d": "3-month realised vol", "close_vs_ma200": "Price vs 200-day MA",
    "ma50_vs_ma200": "50-day vs 200-day MA", "rsi14": "RSI (14)", "drawdown_252d": "Drawdown from 1y high",
    "valuation_vs_5y": "Price vs 5-year MA", "vix": "VIX level", "vix_vs_1y": "VIX vs 1y average",
    "yield_10y": "10Y yield", "cpi_yoy": "CPI YoY", "cpi_surprise_z": "CPI surprise (z)",
    "core_cpi_3m_ann": "Core CPI 3m ann.", "ppi_yoy": "PPI YoY", "fed_funds": "Fed funds rate",
    "fed_funds_chg_1y": "Fed funds 1y change", "unrate_chg_1y": "Unemployment 1y change",
    "curve_10y2y": "10Y-2Y curve", "gdp_qoq_saar": "GDP growth (SAAR)",
    "umich_sentiment_yoy": "Consumer sentiment YoY", "real_yield_10y": "Real 10Y yield",
    "p_bull": "P(bull regime)", "p_bear": "P(bear regime)", "p_high_vol": "P(high-vol regime)",
    "news_sentiment_30d": "News sentiment (30d)", "risk_density": "Event risk density",
    "days_to_target": "Days to target",
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, pal: dict, height: int, legend: bool = True,
            y_title: str | None = None, y_format: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=pal["ink2"], size=13),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=pal["ink2"])),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=pal["axis"], tickcolor=pal["axis"],
                     tickfont=dict(color=pal["muted"]), zeroline=False)
    fig.update_yaxes(gridcolor=pal["grid"], linecolor=pal["axis"], zeroline=False,
                     tickfont=dict(color=pal["muted"]), title=y_title, tickformat=y_format)
    return fig


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def unit(snap: dict) -> str:
    """Unit of the snapshot's prices: 'CAD', or 'index points' for VIX-like series."""
    return snap.get("currency") or "index points"


def money(v: float, snap: dict, decimals: int = 2) -> str:
    return f"C${v:,.{decimals}f}" if snap.get("currency") == "CAD" else f"{v:,.{decimals}f}"


def currency_note(snap: dict) -> str:
    if snap.get("converted_from") == "USD":
        return "USD prices converted to CAD at the daily USD/CAD rate (Yahoo CAD=X), before forecasting, so the range includes currency risk."
    if snap.get("currency") == "CAD":
        return "Priced in CAD."
    return "Index points: not a currency amount, so not converted."


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #

def fan_chart(snap: dict, pal: dict, height: int = 440, compact: bool = False) -> go.Figure:
    """Weekly history, then the P10-P90 band and P50 for each quarter-end.

    compact=True (phones): 1 year of history, no legend or end labels (the
    caption under the chart states the range), small margins.
    """
    hist = pd.DataFrame(snap.get("history") or [])
    fc = pd.DataFrame(snap["forecast"])
    as_of = pd.Timestamp(snap["as_of"])
    last = float(snap["last_close"])
    x = [as_of] + list(pd.to_datetime(fc["target_date"]))
    p10, p50, p90 = ([last] + list(fc[c]) for c in ("p10", "p50", "p90"))
    q = [""] + list(fc["quarter"])
    cad = snap.get("currency") == "CAD"
    pre = "C$" if cad else ""

    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(hist["date"]), y=hist["close"], name=snap.get("history_label", "Weekly close"),
            mode="lines", line=dict(color=pal["s1"], width=2),
            hovertemplate=pre + "%{y:,.2f}<extra>Close</extra>"))
    # band: upper edge first, lower edge fills up to it
    fig.add_trace(go.Scatter(x=x, y=p90, mode="lines", line=dict(color=_rgba(pal["s1"], 0.45), width=1),
                             name="P90", showlegend=False, customdata=q,
                             hovertemplate="P90 " + pre + "%{y:,.2f}<extra>%{customdata}</extra>"))
    fig.add_trace(go.Scatter(x=x, y=p10, mode="lines", line=dict(color=_rgba(pal["s1"], 0.45), width=1),
                             fill="tonexty", fillcolor=_rgba(pal["s1"], 0.12), name="P10–P90 range",
                             customdata=q, hovertemplate="P10 " + pre + "%{y:,.2f}<extra>%{customdata}</extra>"))
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines+markers", name="Median (P50)",
                             line=dict(color=pal["s1"], width=2, dash="dash"),
                             marker=dict(size=8, color=pal["s1"]), customdata=q,
                             hovertemplate="P50 " + pre + "%{y:,.2f}<extra>%{customdata}</extra>"))
    fig.add_vline(x=as_of, line=dict(color=pal["muted"], width=1, dash="dot"))
    fig.add_annotation(x=as_of, y=1, yref="paper", text=f"Forecast from {as_of:%b %d, %Y}",
                       showarrow=False, xanchor="right", yanchor="top", xshift=-6,
                       font=dict(color=pal["muted"], size=12))
    for name, ys in (("P90", p90), ("P50", p50), ("P10", p10)):
        fig.add_annotation(x=x[-1], y=ys[-1], text=f"{name} {pre}{ys[-1]:,.0f}", showarrow=False,
                           xanchor="left", xshift=8, font=dict(color=pal["ink2"], size=12))
    fig.update_layout(hovermode="x unified")
    _layout(fig, pal, height, y_format=",.0f", y_title=f"{snap.get('value_label', 'Price')} ({unit(snap)})")
    fig.update_layout(margin=dict(r=110))
    if compact:
        fig.layout.annotations = ()
        fig.update_layout(height=230, showlegend=False, margin=dict(l=4, r=6, t=6, b=4),
                          hovermode="x unified", dragmode=False)
        fig.update_yaxes(title=None, nticks=4, tickfont=dict(size=11))
        fig.update_xaxes(range=[x[0] - pd.DateOffset(years=1), x[-1] + pd.DateOffset(days=15)],
                         nticks=4, tickfont=dict(size=11))
    else:
        fig.update_xaxes(range=[x[0] - pd.DateOffset(years=2) if hist.empty else pd.to_datetime(hist["date"]).min(),
                                x[-1] + pd.DateOffset(days=120)])
    return fig


def forecast_table(snap: dict) -> pd.DataFrame:
    last = float(snap["last_close"])
    u = "CAD" if snap.get("currency") == "CAD" else "pts"
    rows = []
    for f in snap["forecast"]:
        rows.append({
            "Quarter": f["quarter"],
            "Target date": pd.Timestamp(f["target_date"]).date(),
            f"P10 ({u})": f["p10"], f"P50 ({u})": f["p50"], f"P90 ({u})": f["p90"],
            "P10 %": f["p10"] / last - 1, "P50 %": f["p50"] / last - 1, "P90 %": f["p90"] / last - 1,
            "P(>10% drawdown by then)": f["prob_drawdown"],
        })
    return pd.DataFrame(rows)


def shap_chart(drivers: list[dict], quarter: str, quantile: float, pal: dict, top_n: int = 8) -> go.Figure:
    d = pd.DataFrame(drivers)
    d = d[(d["quarter"] == quarter) & (d["quantile"].round(2) == round(quantile, 2))]
    d = d.reindex(d["shap"].abs().sort_values(ascending=False).index).head(top_n)
    d = d.iloc[::-1]  # largest at the top of a horizontal bar chart
    pp = d["shap"] * 100
    names = [f"{label(f)}  ({v:,.2f})" if pd.notna(v) else f"{label(f)}  (missing)"
             for f, v in zip(d["feature"], d["value"])]
    fig = go.Figure(go.Bar(
        x=pp, y=names, orientation="h",
        marker=dict(color=[pal["pos"] if v >= 0 else pal["neg"] for v in pp], cornerradius=4),
        text=[f"{v:+.1f} pp" for v in pp], textposition="outside", cliponaxis=False,
        textfont=dict(color=pal["ink2"]),
        hovertemplate="%{y}<br>%{x:+.2f} pp<extra></extra>"))
    lim = max(1.0, float(pp.abs().max()) * 1.35) if len(pp) else 1.0
    _layout(fig, pal, height=48 + 34 * max(len(d), 1), legend=False, y_title=None)
    fig.update_layout(bargap=0.4)
    fig.update_xaxes(range=[-lim, lim], showgrid=True, gridcolor=pal["grid"], zeroline=True,
                     zerolinecolor=pal["axis"], title="Contribution to forecast log return (percentage points)",
                     title_font=dict(color=pal["muted"], size=12))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=pal["ink2"]))
    return fig


# --------------------------------------------------------------------------- #
# Macro / events
# --------------------------------------------------------------------------- #

def macro_chart(macro_history: dict, series: list[tuple[str, str, str]], pal: dict,
                y_title: str, height: int = 280) -> go.Figure:
    """series: (series_id, column, display name). <= 2 series per chart."""
    fig = go.Figure()
    colors = [pal["s1"], pal["s2"]]
    for i, (sid, col, name) in enumerate(series):
        rows = macro_history.get(sid) or []
        if not rows:
            continue
        df = pd.DataFrame(rows)
        fig.add_trace(go.Scatter(x=pd.to_datetime(df["date"]), y=df[col], name=name, mode="lines",
                                 line=dict(color=colors[i], width=2),
                                 hovertemplate=f"{name} %{{y:.2f}}<extra></extra>"))
        last = df.dropna(subset=[col]).iloc[-1] if df[col].notna().any() else None
        if last is not None and len(series) > 1:
            fig.add_annotation(x=pd.Timestamp(last["date"]), y=last[col], text=name, showarrow=False,
                               xanchor="left", xshift=6, font=dict(color=pal["ink2"], size=12))
    fig.update_layout(hovermode="x unified")
    return _layout(fig, pal, height, legend=len(series) > 1, y_title=y_title)


def risk_density_chart(rows: list[dict], pal: dict, height: int = 260) -> go.Figure:
    df = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(
        x=df["quarter"], y=df["density"], marker=dict(color=pal["s1"], cornerradius=4),
        text=[f"{v:.0f}" for v in df["density"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=pal["ink2"]),
        customdata=df[["n_high", "n_medium", "n_low", "missing"]].fillna("").to_numpy(),
        hovertemplate=("%{x}: density %{y:.1f}<br>high %{customdata[0]}, medium %{customdata[1]}, "
                       "low %{customdata[2]}<br>not yet scheduled: %{customdata[3]}<extra></extra>")))
    _layout(fig, pal, height, legend=False, y_title="Risk points per 63 trading days")
    fig.update_layout(bargap=0.55)
    fig.update_yaxes(range=[0, max(1.0, float(df["density"].max()) * 1.2)])
    return fig


# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #

def describe_sentiment(v: float) -> str:
    if v is None or pd.isna(v):
        return "No data"
    a = abs(v)
    tone = "neutral" if a < 0.1 else ("mildly " if a < 0.3 else "") + ("positive" if v > 0 else "negative")
    return tone[0].upper() + tone[1:]


def sentiment_gauge(value: float, pal: dict, title: str, height: int = 230) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(valueformat="+.2f", font=dict(color=pal["ink"], size=34)),
        title=dict(text=title, font=dict(color=pal["ink2"], size=14)),
        gauge=dict(
            axis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1], tickcolor=pal["axis"],
                      tickfont=dict(color=pal["muted"])),
            # no fill bar: on a -1..+1 scale a bar growing from -1 makes a neutral
            # reading look strongly negative. A needle-style threshold marks the value.
            bar=dict(color="rgba(0,0,0,0)", thickness=0),
            threshold=dict(value=value, line=dict(color=pal["ink"], width=4), thickness=0.95),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[-1, -0.1], color=_rgba(pal["neg"], 0.22)),
                   dict(range=[-0.1, 0.1], color=pal["neutral"]),
                   dict(range=[0.1, 1], color=_rgba(pal["pos"], 0.22))],
        )))
    fig.update_layout(height=height, margin=dict(l=24, r=24, t=48, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT))
    return fig


def sentiment_trend(rows: list[dict], pal: dict, height: int = 260) -> go.Figure:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure(go.Scatter(x=df["date"], y=df["sentiment"], mode="lines", name="Daily sentiment",
                               line=dict(color=pal["s1"], width=2),
                               customdata=df["headlines"],
                               hovertemplate="%{y:+.2f} (%{customdata} headlines)<extra></extra>"))
    sh = df[df["shock"]]
    if len(sh):
        fig.add_trace(go.Scatter(x=sh["date"], y=sh["sentiment"], mode="markers", name="Shock (|z| > 2)",
                                 marker=dict(size=10, color=pal["ink"], symbol="diamond"),
                                 hovertemplate="shock %{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=pal["axis"], width=1))
    _layout(fig, pal, height, legend=len(sh) > 0, y_title="P(pos) − P(neg)")
    fig.update_yaxes(range=[-1, 1])
    return fig


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #

def allocation_chart(items: list[dict], pal: dict, highlight_over: float | None = None) -> go.Figure:  # noqa: C901
    """Ranked horizontal bars of weights (one series: slot 1). Bars over
    `highlight_over` keep the same colour; the label carries the warning so colour is never alone."""
    df = pd.DataFrame(items).sort_values("weight")
    txt = [f"{w:.1%}" + ("  ⚠ over limit" if highlight_over is not None and w > highlight_over else "")
           for w in df["weight"]]
    fig = go.Figure(go.Bar(x=df["weight"], y=df["label"], orientation="h", text=txt, textposition="outside",
                           cliponaxis=False, marker=dict(color=pal["s1"], cornerradius=4),
                           textfont=dict(color=pal["ink2"]),
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    _layout(fig, pal, height=40 + 30 * max(len(df), 1), legend=False)
    fig.update_layout(bargap=0.4, margin=dict(l=8, r=90, t=8, b=8))
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.05, float(df["weight"].max()) * 1.3)],
                     showgrid=True, gridcolor=pal["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=pal["ink2"]))
    if highlight_over is not None:
        fig.add_vline(x=highlight_over, line=dict(color=pal["muted"], width=1, dash="dot"))
    return fig


def live_gold_chart(df: pd.DataFrame, pal: dict, height: int = 320) -> go.Figure:
    """1-minute gold candles with session VWAP and +-1 / +-2 sigma bands (live_feed.vwap_frame).
    Up candles slot 1 (blue), down slot 2 (orange): red stays reserved for problems."""
    fig = go.Figure()
    x = df.index.tz_convert("America/New_York")
    for k, dash, name in ((2, "dot", "±2σ"), (1, "dash", "±1σ")):
        for sgn in (1, -1):
            fig.add_trace(go.Scatter(x=x, y=df["vwap"] + sgn * k * df["sd"], mode="lines", name=name,
                                     line=dict(color=pal["muted"], width=1, dash=dash), showlegend=sgn > 0,
                                     hovertemplate=f"{'+' if sgn > 0 else '-'}{k}σ %{{y:,.2f}}<extra></extra>"))
    fig.add_trace(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="XAU/USD 1m",
        increasing=dict(line=dict(color=pal["s1"], width=1), fillcolor=pal["s1"]),
        decreasing=dict(line=dict(color=pal["s2"], width=1), fillcolor=pal["s2"])))
    fig.add_trace(go.Scatter(x=x, y=df["vwap"], mode="lines", name="Session VWAP",
                             line=dict(color=pal["ink"], width=1.6), hovertemplate="VWAP %{y:,.2f}<extra></extra>"))
    _layout(fig, pal, height, legend=False, y_format=",.0f")
    fig.update_layout(margin=dict(l=4, r=6, t=6, b=4), dragmode=False, hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=False, nticks=5, tickformat="%H:%M", tickfont=dict(size=11))
    fig.update_yaxes(nticks=5, tickfont=dict(size=11), side="right")
    return fig


def lab_random_hist(edges: list, counts: list, real: float, pal: dict, height: int = 220) -> go.Figure:
    """Expectancy of 1,000 random-entry strategies (bars) vs the real out-of-sample expectancy (line)."""
    e = np.asarray(edges) * 1e4
    mids = (e[:-1] + e[1:]) / 2
    fig = go.Figure(go.Bar(x=mids, y=counts, marker=dict(color=pal["muted"]), width=(e[1] - e[0]) * 0.92,
                           hovertemplate="%{x:+.1f} bps: %{y} random strategies<extra></extra>", name="Random entries"))
    fig.add_vline(x=real * 1e4, line=dict(color=pal["s1"], width=2.5))
    fig.add_annotation(x=real * 1e4, y=1, yref="paper", text=f"this setup {real * 1e4:+.1f} bps", showarrow=False,
                       xanchor="left", yanchor="top", xshift=4, font=dict(color=pal["s1"], size=12))
    _layout(fig, pal, height, legend=False)
    fig.update_layout(margin=dict(l=4, r=6, t=8, b=4), dragmode=False, bargap=0)
    fig.update_xaxes(title=dict(text="expectancy per trade after costs (bps)", font=dict(size=11)), tickfont=dict(size=11))
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return fig


def lab_equity(eq: list, pal: dict, height: int = 200) -> go.Figure:
    y = np.asarray(eq, float) * 100
    fig = go.Figure(go.Scatter(y=y, mode="lines", line=dict(color=pal["s1"], width=2),
                               hovertemplate="%{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=pal["axis"], width=1))
    _layout(fig, pal, height, legend=False, y_format="+.1f")
    fig.update_layout(margin=dict(l=4, r=6, t=8, b=4), dragmode=False)
    fig.update_xaxes(title=dict(text="out-of-sample trades in order", font=dict(size=11)), tickfont=dict(size=11))
    fig.update_yaxes(title=dict(text="cumulative %", font=dict(size=11)), tickfont=dict(size=11))
    return fig


def lab_slices(rows: list[dict], pal: dict, height: int | None = None) -> go.Figure:
    """Expectancy per slice (horizontal bars, one series); the trade count is in the label."""
    keys = [f"{r['key']} ({r['trades']})" for r in rows]           # trade count next to the label
    vals = [r["expectancy"] * 1e4 for r in rows]
    txt = [f"{v:+.1f}" for v in vals]
    lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
    pad = 0.35 * (hi - lo or 1.0)                                    # room for labels on both sides
    fig = go.Figure(go.Bar(x=vals, y=keys, orientation="h", text=txt, textposition="outside", cliponaxis=False,
                           marker=dict(color=pal["s1"], cornerradius=3), textfont=dict(color=pal["ink2"], size=11),
                           hovertemplate="%{y}: %{x:+.1f} bps<extra></extra>"))
    fig.update_xaxes(range=[lo - (pad if lo < 0 else 0), hi + pad])
    _layout(fig, pal, height or 40 + 26 * max(len(rows), 1), legend=False)
    fig.update_layout(margin=dict(l=4, r=8, t=4, b=4), dragmode=False, bargap=0.35)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11), showgrid=False)
    fig.update_xaxes(zeroline=True, zerolinecolor=pal["axis"], tickfont=dict(size=10), gridcolor=pal["grid"],
                     title=dict(text="expectancy per trade, bps (trades in brackets)", font=dict(size=11)))
    return fig


# --------------------------------------------------------------------------- #
# Simulator charts
# --------------------------------------------------------------------------- #

def sim_equity(res: dict, pal: dict, height: int = 300) -> go.Figure:
    """Strategy equity (slot 1) with its drawdown shaded below, and buy-and-hold (slot 2)."""
    from plotly.subplots import make_subplots

    eq = res["equity"]
    t = pd.to_datetime(eq["t"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    if res.get("buy_hold"):
        bh = res["buy_hold"]
        fig.add_trace(go.Scatter(x=pd.to_datetime(bh["t"]), y=bh["equity"], name="Buy & hold",
                                 line=dict(color=pal["s2"], width=1.5),
                                 hovertemplate="Buy & hold C$%{y:,.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=eq["equity"], name="Strategy", line=dict(color=pal["s1"], width=2),
                             hovertemplate="Strategy C$%{y:,.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=eq["drawdown"], name="Drawdown", fill="tozeroy", mode="lines",
                             line=dict(color=pal["neg"], width=1), fillcolor=_rgba(pal["neg"], 0.25),
                             hovertemplate="Drawdown C$%{y:,.0f}<extra></extra>"), row=2, col=1)
    _layout(fig, pal, height, legend=True, y_format=",.0f")
    fig.update_layout(margin=dict(l=4, r=6, t=26, b=4), dragmode=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)))
    fig.update_yaxes(tickfont=dict(size=10), gridcolor=pal["grid"], zeroline=False)
    fig.update_xaxes(tickfont=dict(size=10), showgrid=False)
    return fig


def sim_mc(mc: dict, pal: dict, height: int = 230) -> go.Figure:
    """Monte Carlo of trade order: 5-95% band, median, best and worst paths."""
    x = mc["x"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=mc["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=mc["p5"], fill="tonexty", fillcolor=_rgba(pal["s1"], 0.18), line=dict(width=0),
                             name="5-95% of orders", hovertemplate="5%% path C$%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=mc["p50"], name="Median", line=dict(color=pal["s1"], width=2),
                             hovertemplate="median C$%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=mc["best"], name="Best", line=dict(color=pal["muted"], width=1, dash="dot"),
                             hovertemplate="best C$%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=mc["worst"], name="Worst", line=dict(color=pal["s2"], width=1, dash="dot"),
                             hovertemplate="worst C$%{y:,.0f}<extra></extra>"))
    _layout(fig, pal, height, legend=True, y_format=",.0f")
    fig.update_layout(margin=dict(l=4, r=6, t=26, b=4), dragmode=False,
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)))
    fig.update_xaxes(title=dict(text="trade number", font=dict(size=11)), tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def sim_trades_chart(bars: pd.DataFrame, log: pd.DataFrame, pal: dict, max_trades: int = 150,
                     height: int = 300) -> go.Figure:
    """Close price with the latest `max_trades` entries (triangles) and exits (x) marked."""
    tl = log.tail(max_trades)
    if len(tl):
        lo, hi = pd.Timestamp(tl["entry_time"].min()), pd.Timestamp(tl["exit_time"].max())
        b = bars[(bars.index >= lo - (hi - lo) * 0.03) & (bars.index <= hi + (hi - lo) * 0.03)]
    else:
        b = bars.tail(500)
    step = max(1, len(b) // 1500)
    bb = b.iloc[::step]
    x = bb.index.tz_convert("America/New_York")
    fig = go.Figure(go.Scatter(x=x, y=bb["Close"], mode="lines", line=dict(color=pal["ink2"], width=1),
                               name="Close", hovertemplate="%{y:,.2f}<extra></extra>"))
    for side, sym, col, name in ((1, "triangle-up", pal["s1"], "Long entry"), (-1, "triangle-down", pal["s2"], "Short entry")):
        s = tl[tl["side"] == side]
        if len(s):
            fig.add_trace(go.Scatter(x=pd.DatetimeIndex(s["entry_time"]).tz_convert("America/New_York"), y=s["entry"],
                                     mode="markers", marker=dict(symbol=sym, size=9, color=col), name=name,
                                     hovertemplate=name + " %{y:,.2f}<extra></extra>"))
    if len(tl):
        win = tl["pnl_cad"] > 0
        fig.add_trace(go.Scatter(x=pd.DatetimeIndex(tl["exit_time"]).tz_convert("America/New_York"), y=tl["exit"],
                                 mode="markers", name="Exit",
                                 marker=dict(symbol="x", size=7, color=[pal["ink"] if w else pal["neg"] for w in win]),
                                 customdata=np.c_[tl["pnl_cad"], tl["reason"]],
                                 hovertemplate="Exit %{y:,.2f} · C$%{customdata[0]:,.0f} · %{customdata[1]}<extra></extra>"))
    _layout(fig, pal, height, legend=True, y_format=",.2f")
    fig.update_layout(margin=dict(l=4, r=6, t=26, b=4), dragmode="pan",
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def sim_compare_equity(results: list[dict], names: list[str], pal: dict, height: int = 260) -> go.Figure:
    colors = [pal["s1"], pal["s2"], "#8f6bd8", "#2aa198"]
    fig = go.Figure()
    for res, name, col in zip(results, names, colors):
        eq = res["equity"]
        fig.add_trace(go.Scatter(x=pd.to_datetime(eq["t"]), y=eq["equity"], name=name, line=dict(color=col, width=2),
                                 hovertemplate=name + " C$%{y:,.0f}<extra></extra>"))
    _layout(fig, pal, height, legend=True, y_format=",.0f")
    fig.update_layout(margin=dict(l=4, r=6, t=30, b=4), dragmode=False,
                      legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)))
    return fig


def sparkline_svg(values: list[float], up: str, down: str, w: int = 84, h: int = 24) -> str:
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return f'<svg width="{w}" height="{h}" aria-hidden="true"></svg>'
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    pts = " ".join(f"{i * (w - 2) / (len(v) - 1) + 1:.1f},{h - 2 - (x - lo) / rng * (h - 4):.1f}" for i, x in enumerate(v))
    col = up if v[-1] >= v[0] else down
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<polyline fill="none" stroke="{col}" stroke-width="1.5" points="{pts}"/></svg>')
