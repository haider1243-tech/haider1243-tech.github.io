"""Quarterly Forecaster dashboard.

    streamlit run app.py

Reads JSON snapshots written by `python -m forecaster.pipeline` (or the
Refresh button / scripts/weekly_refresh.sh). Never computes models on page
load. Backtest accuracy is shown above every page.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # works without `pip install -e .`

from forecaster import dashboard as viz  # noqa: E402
from forecaster import pipeline  # noqa: E402
from forecaster.config import load_settings  # noqa: E402

st.set_page_config(page_title="Quarterly Forecaster", page_icon="📈", layout="wide")

PAGES = ("Forecast", "Macro & events", "Sentiment & news")
QUANTILE_LABELS = {0.1: "P10 (downside)", 0.5: "P50 (median)", 0.9: "P90 (upside)"}


# --------------------------------------------------------------------------- #
# Data access (cached on file modification time, so new snapshots show up)
# --------------------------------------------------------------------------- #

settings = load_settings()
snap_dir = pipeline.snapshot_dir(settings)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read(path: str, mtime: float) -> dict | None:
    return pipeline.read_json(Path(path))


def read_snapshot(path: Path) -> dict | None:
    return _read(str(path), _mtime(path))


def ticker_snap(t: str) -> dict | None:
    return read_snapshot(snap_dir / f"ticker_{pipeline.safe_name(t)}.json")


def market_snap() -> dict | None:
    return read_snapshot(snap_dir / "market.json")


def backtest_snap(t: str) -> tuple[dict | None, bool]:
    own = read_snapshot(snap_dir / f"backtest_{pipeline.safe_name(t)}.json")
    if own is not None:
        return own, False
    return read_snapshot(pipeline.REFERENCE_BACKTEST), True


def age_text(iso: str | None) -> str:
    if not iso:
        return "never"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    return f"{h:.0f} h ago" if h < 48 else f"{h / 24:.0f} days ago"


def theme() -> dict:
    try:
        mode = st.context.theme.type or "light"
    except Exception:
        mode = "light"
    return viz.PALETTES["dark" if mode == "dark" else "light"]


pal = theme()

# --------------------------------------------------------------------------- #
# Sidebar: ticker, page, refresh
# --------------------------------------------------------------------------- #

known = list(settings.tickers)
for p in sorted(snap_dir.glob("ticker_*.json")):
    s = read_snapshot(p)
    if s and s.get("ticker") and s["ticker"] not in known:
        known.append(s["ticker"])

with st.sidebar:
    st.title("Quarterly Forecaster")
    ticker = st.selectbox("Ticker", known, key="ticker")
    page = st.radio("Page", PAGES, key="page")
    st.divider()
    st.subheader("Data")
    ts, ms = ticker_snap(ticker), market_snap()
    st.caption(f"Forecast for {ticker}: {age_text(ts and ts.get('generated_at'))}  \n"
               f"Macro, events, news: {age_text(ms and ms.get('generated_at'))}")
    scope = st.radio("Refresh", ("Selected ticker + market", "All tickers + market"), key="scope")
    with_bt = st.checkbox("Also re-run backtest (slow: ~5 min per ticker)", key="with_bt")
    if st.button("Refresh data", type="primary", use_container_width=True):
        log: list[str] = []
        tickers = [ticker] if scope.startswith("Selected") else list(known)
        with st.status("Refreshing…", expanded=True) as status:
            warn = lambda m: (log.append(m), status.write(m))  # noqa: E731
            result = pipeline.refresh_all(settings, tickers, warn=warn,
                                          backtest_stale_days=0 if with_bt else None)
            failed = {k: v for k, v in result.items() if v != "ok"}
            status.update(label="Refresh finished with errors" if failed else "Refresh finished",
                          state="error" if failed else "complete")
        st.session_state["refresh_log"] = [f"{k}: {v}" for k, v in result.items()] + log
        st.cache_data.clear()
        st.rerun()
    if st.session_state.get("refresh_log"):
        with st.expander("Last refresh log"):
            st.code("\n".join(st.session_state["refresh_log"]), language=None)
    st.caption("Weekly auto-refresh: `scripts/weekly_refresh.sh` (see README).")


# --------------------------------------------------------------------------- #
# Banner: backtest accuracy, shown on every page
# --------------------------------------------------------------------------- #

def backtest_banner(t: str) -> None:
    bt, is_ref = backtest_snap(t)
    with st.container(border=True):
        if bt is None:
            st.info("No backtest yet. Run `python -m forecaster.backtest "
                    f"{t}` or tick “Also re-run backtest” and refresh. Until then, "
                    "treat these forecasts as unvalidated.", icon="ℹ️")
            return
        m, b = bt["model"], bt["baseline"]
        span = f"{str(bt.get('first_target') or bt['first_origin'])[:4]}–{str(bt['last_target'])[:4]}"
        title = (f"Backtest accuracy: reference S&P 500 run ({span}), not {t}" if is_ref
                 else f"Backtest accuracy: {t} ({span})")
        st.markdown(f"**{title}** · {bt['n_origins']} quarterly origins, {bt['n_forecasts']} scored forecasts")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Outcomes inside P10–P90", f"{m['coverage']:.0%}", help="Target ≈ 80%.")
        c1.caption(f"target 80% · baseline {b['coverage']:.0%}")
        lo, hi = bt["diff_ci"]
        c2.metric("Pinball skill vs naive baseline", f"{bt['skill']:+.1%}",
                  help="1 − model loss / baseline loss; > 0 means better than the historical-average baseline.")
        c2.caption("90% CI " + ("excludes" if (hi < 0 or lo > 0) else "includes") + " zero")
        c3.metric("Direction hit rate", f"{m['hit_rate']:.0%}")
        c3.caption(f"baseline (≈ always up) {b['hit_rate']:.0%}")
        c4.metric("Drawdown-probability Brier", f"{m['brier_dd']:.3f}", help="Lower is better.")
        c4.caption(f"baseline {b['brier_dd']:.3f} · lower is better")
        if bt["beats_baseline"]:
            st.success("Beats the naive baseline with a 90% CI that excludes zero.", icon="✅")
        elif bt["worse_than_baseline"]:
            st.error("Worse than the naive historical-average baseline. Don't rely on these forecasts.", icon="⛔")
        else:
            st.warning("Not reliably better than a naive historical-average baseline: the edge is within noise. "
                       "Use the ranges as a rough guide, not a signal.", icon="⚠️")
        if is_ref:
            st.caption(f"No backtest for {t} yet. {bt.get('note', '')}")
        with st.expander("Backtest details"):
            for v in bt["verdict"]:
                st.markdown(f"- {v}")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def page_forecast(t: str) -> None:
    snap = ticker_snap(t)
    if snap is None:
        st.info(f"No forecast for {t} yet. Press **Refresh data** in the sidebar, or run "
                f"`python -m forecaster.pipeline {t}`.", icon="ℹ️")
        return
    as_of = pd.Timestamp(snap["as_of"])
    st.subheader(f"{t}: 4-quarter forecast range")
    st.caption(f"As of {as_of:%b %d, %Y} · last close {viz.money(snap['last_close'], snap)} "
               f"({viz.unit(snap)}) · {snap['model']} · generated {age_text(snap['generated_at'])}  \n"
               f"{viz.currency_note(snap)}")
    if (pd.Timestamp.today() - as_of).days > 10:
        st.warning(f"Price data ends {as_of:%b %d, %Y}; refresh for an up-to-date forecast.", icon="⚠️")

    st.plotly_chart(viz.fan_chart(snap, pal), use_container_width=True)
    tbl = viz.forecast_table(snap)
    st.dataframe(
        tbl, hide_index=True, use_container_width=True,
        column_config={
            **{c: st.column_config.NumberColumn(c, format="%.2f") for c in tbl.columns if c[:3] in ("P10", "P50", "P90") and "%" not in c},
            **{c: st.column_config.NumberColumn(c, format="percent") for c in ("P10 %", "P50 %", "P90 %")},
            "P(>10% drawdown by then)": st.column_config.ProgressColumn(
                "P(>10% drawdown by then)", min_value=0.0, max_value=1.0, format="percent"),
        })
    st.caption("Ranges, not predictions: each quarter-end price should land between P10 and P90 about "
               "80% of the time if the model is calibrated (see the backtest above). The drawdown "
               "probability overstated risk at the high end in the backtest.")

    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Current regime (HMM, filtered)**")
        reg = snap["regime"]["current"]
        for k, name in (("bull", "Bull"), ("bear", "Bear"), ("high_vol", "High volatility")):
            st.progress(float(reg[k]), text=f"{name}: {reg[k]:.0%}")
        st.caption("Annualised drift by regime: " + ", ".join(
            f"{k.replace('_', '-')} {v:+.0%}" for k, v in snap["regime"]["drift_annualized"].items())
            + ". Labels are assigned by rule: “high volatility” is the most volatile state and “bear” "
              "is the lower-return of the two calmer states, so “bear” can still have positive drift.")
        g = snap.get("garch") or {}
        if g.get("persistence") is not None:
            st.caption(f"GARCH persistence {g['persistence']:.3f} · long-run vol {g['long_run_vol']:.0%}")
    with right:
        st.markdown("**What drives the LightGBM part (SHAP)**")
        if not snap["drivers"]:
            st.info("LightGBM wasn't used for this forecast (not enough training history), so there are "
                    "no SHAP drivers.", icon="ℹ️")
        else:
            c1, c2 = st.columns(2)
            quarter = c1.selectbox("Quarter", [f["quarter"] for f in snap["forecast"]], key="shap_q")
            qlev = c2.selectbox("Quantile", list(QUANTILE_LABELS), index=1,
                                format_func=QUANTILE_LABELS.get, key="shap_level")
            st.plotly_chart(viz.shap_chart(snap["drivers"], quarter, qlev, pal), use_container_width=True)
            st.caption("Blue raises the forecast, red lowers it. These explain the LightGBM model only, "
                       f"which has {snap['weights']['lgb']:.0%} weight in the blend. The backtest found it "
                       "overconfident and prone to fitting noise, so read drivers as the model's reasoning, "
                       "not established causes.")


def _metric(col, label: str, rows: list[dict], key: str, unit: str) -> None:
    if not rows:
        col.metric(label, "n/a")
        return
    df = pd.DataFrame(rows).dropna(subset=[key])
    if df.empty:
        col.metric(label, "n/a")
        return
    cur, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else None
    delta = None if prev is None else f"{round(cur[key] - prev[key], 2) + 0.0:+.2f} {unit} vs prior month"
    col.metric(label, f"{cur[key]:.2f}{'%' if unit == 'pp' else ''}", delta, delta_color="off")
    col.caption(f"latest: {pd.Timestamp(cur['date']):%b %Y}")


def page_macro(t: str) -> None:
    ms = market_snap()
    if ms is None:
        st.info("No macro/events snapshot yet. Press **Refresh data**.", icon="ℹ️")
        return
    for w in ms.get("warnings", []):
        if "FRED" in w or "FINNHUB" in w:
            st.warning(w, icon="⚠️")
    mh = ms.get("macro_history") or {}
    st.subheader("Macro")
    c1, c2, c3, c4 = st.columns(4)
    _metric(c1, "CPI, YoY", mh.get("CPIAUCSL", []), "yoy", "pp")
    _metric(c2, "Core CPI, YoY", mh.get("CPILFESL", []), "yoy", "pp")
    _metric(c3, "PPI, YoY", mh.get("PPIACO", []), "yoy", "pp")
    _metric(c4, "Fed funds rate", mh.get("FEDFUNDS", []), "value", "pp")
    if mh:
        a, b = st.columns(2)
        a.markdown("**Inflation, % year over year**")
        a.plotly_chart(viz.macro_chart(mh, [("CPIAUCSL", "yoy", "CPI"), ("PPIACO", "yoy", "PPI")], pal,
                                       "% YoY"), use_container_width=True)
        b.markdown("**Fed funds rate, %**")
        b.plotly_chart(viz.macro_chart(mh, [("FEDFUNDS", "value", "Fed funds")], pal, "%"),
                       use_container_width=True)
        st.caption("Latest revised values from FRED, for display. The forecast model uses first-release "
                   "values keyed to their publication dates.")

    st.subheader("Next 90 days: scheduled events")
    ev = pd.DataFrame(ms.get("events") or [])
    high_only = st.toggle("High-impact only", value=True, key="high_only")
    if ev.empty:
        st.info("No events in the snapshot.", icon="ℹ️")
    else:
        ev["date"] = pd.to_datetime(ev["date"])
        today = pd.Timestamp.today().normalize()
        ev = ev[(ev["date"] >= today) & (ev["date"] <= today + pd.Timedelta(days=90))]
        if high_only:
            ev = ev[ev["impact"] == "high"]
        ev = ev.sort_values(["date", "points"], ascending=[True, False])
        if ev.empty:
            st.info("Nothing scheduled in the next 90 days for this filter (the snapshot may be old).", icon="ℹ️")
        else:
            show = pd.DataFrame({
                "Date": ev["date"].dt.date, "In": (ev["date"] - today).dt.days.astype(str) + " d",
                "Event": ev["name"], "Impact": ev["impact"].str.capitalize(),
                "Ticker": ev["ticker"].fillna(""), "Detail": ev["detail"].fillna(""), "Source": ev["source"]})
            st.dataframe(show, hide_index=True, use_container_width=True)

    rd = ms.get("risk_density") or []
    if rd:
        st.subheader("Event risk density by quarter")
        st.plotly_chart(viz.risk_density_chart(rd, pal), use_container_width=True)
        missing = [r for r in rd if r.get("missing")]
        if missing:
            st.caption("Lower bound where some releases aren't scheduled yet: " + "; ".join(
                f"{r['quarter']} missing {r['missing']}" for r in missing))


def page_sentiment(t: str) -> None:
    ms = market_snap()
    if ms is None:
        st.info("No news snapshot yet. Press **Refresh data**.", icon="ℹ️")
        return
    sent = ms.get("sentiment") or {}
    if not sent:
        st.info("No scored headlines yet. News needs FINNHUB_API_KEY, and scoring needs FinBERT "
                "(`pip install -r requirements-ml.txt`). History builds up with each refresh.", icon="ℹ️")
        return

    def last_week(tag: str) -> tuple[float, int]:
        df = pd.DataFrame(sent.get(tag) or [])
        if df.empty:
            return float("nan"), 0
        df["date"] = pd.to_datetime(df["date"])
        w = df[df["date"] >= df["date"].max() - pd.Timedelta(days=6)]
        n = int(w["headlines"].sum())
        return float((w["sentiment"] * w["headlines"]).sum() / n) if n else float("nan"), n

    st.subheader("News sentiment (FinBERT)")
    cols = st.columns(2)
    for col, tag, name in ((cols[0], "MARKET", "Market"), (cols[1], t.upper(), t)):
        v, n = last_week(tag)
        with col:
            if n == 0:
                st.metric(f"{name}: 7-day sentiment", "n/a")
                st.caption("No headlines for this tag yet (indices/ETFs have no company news).")
            else:
                st.plotly_chart(viz.sentiment_gauge(v, pal, f"{name}: 7-day sentiment"),
                                use_container_width=True)
                st.caption(f"{viz.describe_sentiment(v)} · {n} headlines · scale −1 (all negative) to +1")
    if sent.get("MARKET"):
        st.markdown("**Market sentiment, daily**")
        st.plotly_chart(viz.sentiment_trend(sent["MARKET"], pal), use_container_width=True)
    shocks = ms.get("shocks") or []
    if shocks:
        st.caption("Recent shocks (|z| > 2 vs prior 30 news days; about 5% of quiet days trigger by chance): " +
                   "; ".join(f"{s['tag']} {str(s['date'])[:10]} ({s['sentiment']:+.2f})" for s in shocks[:6]))

    st.subheader("Top headlines")
    h = pd.DataFrame(ms.get("headlines") or [])
    if h.empty:
        st.info("No recent headlines.", icon="ℹ️")
        return
    tags = ["MARKET"] + sorted(set(h["tag"]) - {"MARKET"})
    c1, c2 = st.columns(2)
    tag = c1.selectbox("Source", tags, index=tags.index(t.upper()) if t.upper() in tags else 0,
                       format_func=lambda x: "Market (general)" if x == "MARKET" else x, key="h_tag")
    order = c2.radio("Sort", ("Most recent", "Most negative", "Most positive"), horizontal=True, key="h_sort")
    f = h[h["tag"] == tag]
    f = (f.sort_values("ts", ascending=False) if order == "Most recent" else
         f.sort_values("sentiment", ascending=order == "Most negative"))
    st.dataframe(
        f.head(15).assign(date=pd.to_datetime(f["date"]).dt.date)[["date", "source", "headline", "sentiment", "url"]],
        hide_index=True, use_container_width=True,
        column_config={
            "date": "Date", "source": "Source", "headline": st.column_config.TextColumn("Headline", width="large"),
            "sentiment": st.column_config.NumberColumn("Score", format="%+.2f",
                                                       help="P(positive) − P(negative), −1 to +1"),
            "url": st.column_config.LinkColumn("Link", display_text="open"),
        })


# --------------------------------------------------------------------------- #

backtest_banner(ticker)
{"Forecast": page_forecast, "Macro & events": page_macro, "Sentiment & news": page_sentiment}[page](ticker)
