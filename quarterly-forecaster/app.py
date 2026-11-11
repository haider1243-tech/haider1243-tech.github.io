"""Quarterly Forecaster: private, mobile-first dashboard.

    streamlit run app.py        (or start.bat / python -m forecaster.launcher)

Bottom tabs: This Week · Calendar · Gold VWAP · Portfolio · Reports. Reads JSON
snapshots only (never runs models on page load). Password-gated, dark theme,
installable to the phone home screen (PWA manifest in static/). A red banner
shows whenever a data source failed or is older than its limit.
"""

from __future__ import annotations

import html
import os
import sys
import time as _time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # works without `pip install -e .`

from forecaster import appdata as ad  # noqa: E402
from forecaster import dashboard as viz  # noqa: E402
from forecaster import pipeline  # noqa: E402
from forecaster import status as health  # noqa: E402
from forecaster.config import PROJECT_ROOT, load_settings  # noqa: E402

st.set_page_config(page_title="Forecaster", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")

TABS = {"week": "This Week", "live": "Live", "signals": "Signals", "calendar": "Calendar", "gold": "Gold VWAP", "portfolio": "Portfolio",
        "reports": "Reports", "status": "Status"}
ICONS = ["📊", "⚡", "🎯", "📅", "🥇", "💼", "📄", "🚦"]
PAL = viz.PALETTES["dark"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
settings = load_settings()


@st.cache_resource(show_spinner=False)
def _status_check_on_start() -> bool:
    """Once per app (server) start: run the build check in the background -> STATUS.md."""
    from forecaster import project_status

    try:
        return project_status.start_background(settings.cache_dir)
    except Exception:  # never block the app on it
        return False


_status_check_on_start()

# --------------------------------------------------------------------------- #
# Head tags (PWA) + mobile CSS
# --------------------------------------------------------------------------- #

PWA_HEAD = """<script>
(function () {
  const d = document;
  if (d.querySelector('link[rel="manifest"]')) return;
  const add = (tag, attrs) => { const el = d.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v)); d.head.appendChild(el); };
  add('link', {rel: 'manifest', href: '/app/static/manifest.json'});
  add('link', {rel: 'apple-touch-icon', href: '/app/static/apple-touch-icon.png'});
  add('meta', {name: 'theme-color', content: '#0d0d0d'});
  add('meta', {name: 'mobile-web-app-capable', content: 'yes'});
  add('meta', {name: 'apple-mobile-web-app-capable', content: 'yes'});
  add('meta', {name: 'apple-mobile-web-app-status-bar-style', content: 'black-translucent'});
  add('meta', {name: 'apple-mobile-web-app-title', content: 'Forecaster'});
})();
</script>"""

_nav_icons = "\n".join(
    f'.st-key-bottomnav [role="radiogroup"] > div:nth-child({i + 1}) p::before {{ content: "{ic}"; }}'
    for i, ic in enumerate(ICONS))
CSS = """<style>
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
.stDeployButton { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 0.75rem 0.85rem 6.5rem !important; max-width: 760px; }
html, body { overscroll-behavior-y: none; }
h1, h2, h3 { letter-spacing: -0.01em; }
.st-key-pwa { display: none; }
[data-testid="stHeaderActionElements"] { display: none !important; }   /* heading link icons */

/* bottom tab bar: radiogroup > div[data-selected] > label > (span>input, div > (circle, markdown>p)) */
.st-key-bottomnav { position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000; background: #1a1a19;
  border-top: 1px solid rgba(255,255,255,0.10); padding: 4px 4px calc(4px + env(safe-area-inset-bottom)); gap: 0; }
.st-key-bottomnav .stElementContainer, .st-key-bottomnav [data-testid="stRadio"] { width: 100% !important; }
.st-key-bottomnav [data-testid="stWidgetLabel"] { display: none; }
.st-key-bottomnav [role="radiogroup"] { display: flex; flex-wrap: nowrap; justify-content: space-around;
  gap: 0 !important; width: 100%; max-width: 760px; margin: 0 auto; }
.st-key-bottomnav [role="radiogroup"] > div { flex: 1 1 0; min-width: 0; display: flex; }
.st-key-bottomnav [data-testid="stRadioOption"] { flex: 1 1 auto; justify-content: center; margin: 0 !important;
  padding: 5px 0 3px; border-radius: 10px; cursor: pointer; -webkit-tap-highlight-color: transparent; }
.st-key-bottomnav [data-testid="stRadioOption"] > div { display: flex; justify-content: center; width: 100%; }
.st-key-bottomnav [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none !important; }
.st-key-bottomnav [data-testid="stRadioOption"] p { font-size: 0.68rem; line-height: 1.15; text-align: center;
  color: #898781; white-space: nowrap; margin: 0; }
.st-key-bottomnav [data-testid="stRadioOption"] p::before { display: block; font-size: 1.3rem; line-height: 1.35;
  filter: grayscale(1) opacity(0.65); }
.st-key-bottomnav [role="radiogroup"] > div[data-selected="true"] p { color: #3987e5; font-weight: 600; }
.st-key-bottomnav [role="radiogroup"] > div[data-selected="true"] p::before { filter: none; }
.st-key-bottomnav [role="radiogroup"] > div[data-selected="true"] [data-testid="stRadioOption"] { background: rgba(57,135,229,0.14); }
""" + _nav_icons + """

/* live tiles */
.qf-tile.stale { background: #d03b3b !important; }
.qf-tile.stale .k, .qf-tile.stale .d, .qf-tile.stale .v { color: #ffffff !important; }
.qf-tile.closed, .qf-tile.off { opacity: 0.55; }
.qf-tile .ms { font-variant-numeric: tabular-nums; font-size: 0.7rem; color: #898781; }
.qf-tile.stale .ms { color: #ffffff; }

/* signals */
.qf-setup.noedge { opacity: 0.42; filter: grayscale(0.8); }
.qf-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin: 6px 0 0; }
.qf-stats div { background: #232322; border-radius: 8px; padding: 5px 7px; font-size: 0.72rem; color: #898781; }
.qf-stats b { display: block; color: #f0efec; font-size: 0.88rem; font-weight: 600; }

/* cards, chips, badges */
.qf-card { background: #1a1a19; border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 12px 14px; margin: 0 0 10px; }
.qf-muted { color: #898781; font-size: 0.8rem; }
.qf-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 8px; }
.qf-chip { flex: 1 1 100px; background: #232322; border-radius: 10px; padding: 8px 10px; min-width: 0; }
.qf-chip .k { color: #898781; font-size: 0.72rem; }
.qf-chip .v { color: #f0efec; font-size: 1.05rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.qf-chip .d { font-size: 0.75rem; color: #c3c2b7; }
.qf-badge { display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: 0.72rem; font-weight: 700; color: #0b0b0b; }
.qf-banner { background: #d03b3b; color: #ffffff; border-radius: 12px; padding: 10px 12px; margin: 0 0 10px; font-size: 0.85rem; }
.qf-banner b { display: block; margin-bottom: 2px; }
.qf-item { display: flex; gap: 10px; align-items: flex-start; border-left: 4px solid #383835; padding: 8px 10px; margin: 0 0 8px;
  background: #1a1a19; border-radius: 0 10px 10px 0; }
.qf-item .t { font-weight: 600; color: #f0efec; font-size: 0.92rem; }
.qf-item .s { color: #c3c2b7; font-size: 0.78rem; }
.qf-item .r { margin-left: auto; text-align: right; white-space: nowrap; font-size: 0.8rem; color: #f0efec; }
.qf-outlook p { margin: 0 0 4px; font-size: 0.95rem; }
[data-testid="stPlotlyChart"] { margin-bottom: -6px; }
.st-key-report table { display: block; overflow-x: auto; max-width: 100%; white-space: nowrap; font-size: 0.82rem; }
.st-key-report h1 { font-size: 1.35rem; } .st-key-report h2 { font-size: 1.1rem; } .st-key-report h3 { font-size: 1rem; }
</style>"""


def head_and_css() -> None:
    with st.container(key="pwa"):
        st.html(PWA_HEAD, unsafe_allow_javascript=True)
    st.html(CSS)


head_and_css()


# --------------------------------------------------------------------------- #
# Login (fails closed)
# --------------------------------------------------------------------------- #

def require_login() -> None:
    from forecaster.auth import LoginGuard, password_hash_from_env, verify_password

    stored = password_hash_from_env()
    if not stored:
        st.error("Dashboard locked: set APP_PASSWORD_HASH in .env (run `python -m forecaster.auth` to create it), "
                 "then restart.", icon="🔒")
        st.stop()
    if st.session_state.get("auth_ok"):
        return
    guard = LoginGuard(settings.cache_dir / "auth_failures.json")
    st.markdown("## 📈 Forecaster")
    with st.form("login"):
        pw = st.text_input("Password", type="password", key="login_pw")
        ok = st.form_submit_button("Log in", width="stretch")
    if ok:
        fails = st.session_state.get("login_fails", 0)
        until = st.session_state.get("login_locked_until", 0.0)
        wait = max(guard.locked_for(), until - _time.time())
        if wait > 0:
            st.error(f"Too many failed attempts. Try again in {wait / 60:.0f} min.", icon="⛔")
        elif verify_password(pw, stored):
            st.session_state["auth_ok"] = True
            st.session_state["login_fails"] = 0
            st.rerun()
        else:
            guard.record_failure()
            fails += 1
            st.session_state["login_fails"] = fails
            if fails >= 5:
                st.session_state["login_locked_until"] = _time.time() + 300
            _time.sleep(min(2.0, 0.25 * fails))
            st.error("Wrong password.", icon="⛔")
    st.stop()


require_login()
snap_dir = pipeline.snapshot_dir(settings)


# --------------------------------------------------------------------------- #
# Data access (cached by file modification time)
# --------------------------------------------------------------------------- #

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def _read(path: str, mtime: float) -> dict | None:
    return pipeline.read_json(Path(path))


def snap(name: str) -> dict | None:
    p = snap_dir / name
    return _read(str(p), _mtime(p))


def ticker_snap(t: str) -> dict | None:
    return snap(f"ticker_{pipeline.safe_name(t)}.json")


def all_ticker_snaps() -> dict[str, dict]:
    out = {}
    for p in sorted(snap_dir.glob("ticker_*.json")):
        s = _read(str(p), _mtime(p))
        if s and s.get("ticker"):
            out[s["ticker"]] = s
    return out


def now_utc() -> pd.Timestamp:
    from forecaster.clock import now

    return now()                       # shifted in demo mode, so the simulated market is open


def esc(x) -> str:
    return html.escape(str(x))


# --------------------------------------------------------------------------- #
# Shared header / footer
# --------------------------------------------------------------------------- #

def header(title: str) -> None:
    from forecaster.clock import is_demo

    now = now_utc()
    if is_demo():
        st.html('<div class="qf-banner" style="background:#8a5a00"><b>⚡ Demo mode: simulated market</b>'
                f'Prices are generated on this PC, not real quotes. Simulated time {now.tz_convert(ad.NY):%H:%M} ET. '
                "Strategies, shadow trades and signals run exactly as they would live. Stop with stop.bat.</div>")
    # demo mode skips the daily/weekly data jobs on purpose, so their "no data" isn't a problem there
    probs = [] if is_demo() else health.problems(settings.cache_dir, now, health.expected_sources(settings))
    if probs:
        items = "".join(f"<li>{esc(p.message)}</li>" for p in probs)
        st.html(f'<div class="qf-banner" role="alert"><b>⚠ Data problem</b><ul style="margin:0;padding-left:18px">'
                f"{items}</ul></div>")
    st.markdown(f"### {title}")
    lu = lambda s: ad.ago(health.last_updated(settings.cache_dir, s), now)  # noqa: E731
    st.html(f'<div class="qf-muted">Last updated · prices {esc(lu("prices"))} · forecasts {esc(lu("forecasts"))}'
            f' · macro {esc(lu("market"))}</div>')


def model_note() -> None:
    bt = pipeline.read_json(pipeline.REFERENCE_BACKTEST)
    if bt:
        m = bt["model"]
        st.caption(f"Model check (S&P 500 backtest 2005–2022): {m['coverage']:.0%} of outcomes inside P10–P90 "
                   f"(target 80%), skill {bt['skill']:+.1%} vs a naive baseline, within noise. "
                   "Treat ranges as a rough guide.")


def settings_footer() -> None:
    with st.expander("⚙️  Refresh & account"):
        st.caption("Prices and news refresh every 15 min, macro twice a day, and forecasts rebuild every "
                   "Sunday at 6 pm MT, all automatically.")
        if st.button("Refresh macro, events & portfolio now", width="stretch", key="refresh"):
            with st.status("Refreshing…") as s:
                res = pipeline.refresh_all(settings, warn=s.write, forecasts=False)
                s.update(label="Done" if all(v == "ok" for v in res.values()) else "Finished with errors",
                         state="complete")
            st.cache_data.clear()
            st.rerun()
        if st.button("Log out", width="stretch", key="logout"):
            st.session_state.clear()
            st.rerun()


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def chip(k: str, v: str, d: str = "") -> str:
    return (f'<div class="qf-chip"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
            f'<div class="d">{esc(d)}</div></div>')


def quote_chips(prices: dict | None, symbols: list[tuple[str, str]], extra=()) -> str:
    q = (prices or {}).get("quotes") or {}
    chips = []
    for sym, label in symbols:
        x = q.get(sym)
        if not x:
            chips.append(f'<div class="qf-chip"><div class="k">{esc(label)}</div><div class="v">n/a</div></div>')
            continue
        if x.get("price_cad") is not None and sym not in ("^VIX", "CAD=X"):
            val = f"C${x['price_cad']:,.2f}"
        elif sym == "CAD=X":
            val = f"{x['price']:.4f}"
        else:
            val = f"{x['price']:.2f}"
        chg = "" if x.get("change") is None else f"{x['change']:+.2%} today"
        chips.append(f'<div class="qf-chip"><div class="k">{esc(label)}</div><div class="v">{esc(val)}</div>'
                     f'<div class="d">{esc(chg)}</div></div>')
    chips += [chip(*e) for e in extra]
    return '<div class="qf-row">' + "".join(chips) + "</div>"


def fan_block(t: str, label: str) -> None:
    s = ticker_snap(t)
    st.markdown(f"**{label}**")
    if not s or not s.get("forecast"):
        st.caption("No forecast yet: it rebuilds every Sunday at 6 pm MT.")
        return
    st.plotly_chart(viz.fan_chart(s, PAL, compact=True), width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False}, key=f"fan_{t}")
    f = s["forecast"][-1]
    st.caption(f"{f['quarter']}: likely range {viz.money(f['p10'], s, 0)} – {viz.money(f['p90'], s, 0)}, "
               f"middle {viz.money(f['p50'], s, 0)} (from {viz.money(s['last_close'], s, 0)}). "
               "Blue band = P10–P90; dashed = P50.")


def tab_week() -> None:
    header("This Week")
    snaps, market, prices = all_ticker_snaps(), snap("market.json"), snap("prices.json")
    reg = ad.regime_of(snaps.get("SPY"))
    vix = ((prices or {}).get("quotes") or {}).get("^VIX")
    reg_txt = f"{ad.REGIME_WORDS.get(reg[0], reg[0]).capitalize()} · {reg[1]:.0%}" if reg else "n/a"
    vix_txt = f"{vix['price']:.1f}" if vix else "n/a"
    vix_d = "" if not vix or vix.get("change") is None else f"{vix['change']:+.1%} today"
    lines = "".join(f"<p>{esc(x)}</p>" for x in ad.outlook_lines(snaps, market, prices, now_utc()))
    st.html(f"""<div class="qf-card">
      <div class="qf-row">
        <div class="qf-chip"><div class="k">Market regime (SPY)</div><div class="v">{esc(reg_txt)}</div></div>
        <div class="qf-chip"><div class="k">VIX</div><div class="v">{esc(vix_txt)}</div><div class="d">{esc(vix_d)}</div></div>
      </div>
      <div class="qf-outlook">{lines}</div>
    </div>""")
    st.html(quote_chips(prices, [("SPY", "SPY"), ("QQQ", "QQQ"), ("GC=F", "Gold")]))
    for t, label in (("SPY", "S&P 500 (SPY) · CAD"), ("QQQ", "Nasdaq 100 (QQQ) · CAD"), ("GC=F", "Gold · CAD")):
        fan_block(t, label)
    model_note()


IMPACT = {"high": ("critical", "High"), "medium": ("warning", "Medium"), "low": ("good", "Low")}


def tab_calendar() -> None:
    header("Calendar")
    st.caption("Next 14 days: CPI, PPI, FOMC, jobs, GDP and earnings, with typical release times in New York "
               "time (ET; Mountain time is 2 h earlier). Countdowns update every minute.")

    @st.fragment(run_every="60s")
    def events() -> None:
        items = ad.calendar_items(snap("market.json"), now_utc(), days=14)
        if not items:
            st.info("No events in the next 14 days (or the events snapshot hasn't been built yet).", icon="ℹ️")
            return
        out = []
        for it in items:
            col, word = IMPACT.get(it.impact, ("good", it.impact.title()))
            et = it.when.tz_convert(ad.NY)
            tick = f" · {esc(it.ticker)}" if it.ticker else ""
            out.append(f"""<div class="qf-item" style="border-left-color:{STATUS[col]}">
              <div><div class="t">{esc(it.name)}{tick}</div>
              <div class="s">{et:%a %b %d · %H:%M} ET{(' · ' + esc(it.detail)) if it.detail else ''}</div></div>
              <div class="r"><span class="qf-badge" style="background:{STATUS[col]}">{word}</span>
              <div style="margin-top:4px">{esc(it.countdown)}</div></div></div>""")
        st.html("".join(out))
    events()
    model_note()


VERDICT = {"TAKE": "good", "CAUTION": "warning", "SKIP": "critical"}


def tab_gold() -> None:
    header("Gold VWAP")
    prices = snap("prices.json")
    g = ((prices or {}).get("quotes") or {}).get("GC=F")
    if g:
        z = g.get("z")
        ztxt = "n/a (early in session)" if z is None else f"{z:+.1f} st. dev."
        vw = f"VWAP {g['vwap']:,.2f} USD" if g.get("vwap") else ""
        st.html(quote_chips(prices, [("GC=F", "Gold (GC=F)")], extra=[("vs session VWAP", ztxt, vw)]))
    j = ad.gold_journal(settings.cache_dir / "alerts.sqlite")
    st.markdown("**Journal win rate**")
    if j["stats"]:
        rows = []
        for r in j["stats"]:
            rows.append({"Verdict": f'{r["verdict"]} · {r["horizon"]}', "n": r["n"],
                         "Win rate": 100 * r["win_rate"], "Avg": 100 * r["avg_ret"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Verdict": st.column_config.TextColumn("Verdict · after", width="small"),
            "n": st.column_config.NumberColumn("Trades", width="small"),
            "Win rate": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.0f%%", width="small"),
            "Avg": st.column_config.NumberColumn("Avg move", format="%+.2f%%", width="small")})
        st.caption("Win = the trade's direction paid off at that horizon, measured on GC=F from the alert time. "
                   "Small samples are noisy: compare TAKE with SKIP before trusting the filters.")
    else:
        st.caption("No resolved outcomes yet: win rates appear 1 h / 4 h / 1 day after alerts arrive.")
    st.markdown(f"**Latest alerts** · {j['n_alerts']} total")
    if not j["alerts"]:
        st.info("No TradingView alerts yet. They appear here as soon as the webhook receives one.", icon="ℹ️")
    out = []
    now = now_utc()
    for a in j["alerts"]:
        col = STATUS[VERDICT.get(a["verdict"], "warning")]
        reasons = "; ".join(a["reasons"][:2]) if a["reasons"] else ""
        t = pd.Timestamp(a["alert_time"])
        out.append(f"""<div class="qf-item" style="border-left-color:{col}">
          <div style="min-width:0"><div class="t">{esc(a['side'].upper())} {esc(a['ticker'])} @ {a['price']:,.2f}</div>
          <div class="s">{esc(a['strategy'])} · {esc(ad.ago(t, now))}</div>
          <div class="s" style="white-space:normal">{esc(reasons)}</div></div>
          <div class="r"><span class="qf-badge" style="background:{col}">{esc(a['verdict'])}</span></div></div>""")
    if out:
        st.html("".join(out))
    model_note()


def tab_portfolio() -> None:
    header("Portfolio")
    ps = snap("portfolio.json")
    if ps is None:
        st.info("No portfolio yet. Copy `portfolio.example.yaml` to `portfolio.yaml`, enter your holdings, and it "
                "appears after the next refresh (or use ⚙️ below).", icon="ℹ️")
        settings_footer()
        return
    pnl_pct = ps.get("total_pnl_pct")
    vol = (ps.get("risk") or {}).get("volatility")
    st.html(f"""<div class="qf-row">
      <div class="qf-chip"><div class="k">Value</div><div class="v">C${ps['total_value']:,.0f}</div></div>
      <div class="qf-chip"><div class="k">P&amp;L</div><div class="v">{'+' if ps['total_pnl'] >= 0 else '-'}C${abs(ps['total_pnl']):,.0f}</div>
        <div class="d">{'' if pnl_pct is None else f'{pnl_pct:+.1%} on cost'}</div></div>
      <div class="qf-chip"><div class="k">Volatility</div><div class="v">{'n/a' if vol is None else f'{vol:.1%}'}</div></div>
    </div>""")
    for f in ps.get("flags", []):
        st.warning(f["message"], icon="⚠️")
    h = pd.DataFrame(ps["holdings"])
    out = []
    for _, r in h.iterrows():
        pnl = "" if r["ticker"] == "CASH" else f"{'+' if r['pnl'] >= 0 else '-'}C${abs(r['pnl']):,.0f} ({r['pnl_pct']:+.1%})"
        out.append(f"""<div class="qf-item" style="border-left-color:{PAL['s1']}">
          <div style="min-width:0"><div class="t">{esc(r['ticker'])}</div><div class="s">{esc(r['account'])} · {r['weight']:.1%}</div></div>
          <div class="r">C${r['market_value']:,.0f}<div class="s">{esc(pnl)}</div></div></div>""")
    st.html("".join(out))
    view = st.radio("Allocation", ["asset_class", "sector", "account", "region"], horizontal=True, key="alloc",
                    format_func=lambda k: {"asset_class": "Class", "sector": "Sector", "account": "Account",
                                           "region": "Region"}[k])
    st.plotly_chart(viz.allocation_chart(ps["allocation"][view], PAL), width="stretch",
                    config={"displayModeBar": False}, key="alloc_chart")
    if ps.get("forecast") and "p10" in ps["forecast"][0]:
        st.markdown("**Forecast range (CAD)**")
        st.plotly_chart(viz.fan_chart(ps, PAL, compact=True), width="stretch",
                        config={"displayModeBar": False}, key="pf_fan")
        f = ps["forecast"][-1]
        st.caption(f"{f['quarter']}: likely C${f['p10']:,.0f} – C${f['p90']:,.0f}, middle C${f['p50']:,.0f}; "
                   f"{f.get('prob_loss', 0):.0%} chance of being below today. Joint bootstrap of your holdings' "
                   "history, buy-and-hold, not backtested.")
    with st.expander("💡 Plan a contribution"):
        planner_block(ps)
    settings_footer()


def planner_block(ps: dict) -> None:
    from forecaster import planner as pl

    st.caption("Splits a contribution toward your target allocation by buying what's most underweight. "
               "Buy-only: it never sells, never uses leverage, margin or options, and the buys always add up "
               "to exactly your contribution.")
    h = pd.DataFrame(ps["holdings"])
    held = h[h["ticker"] != "CASH"]
    current = held.groupby("ticker")["market_value"].sum().to_dict()
    prices = held.groupby("ticker")["price_cad"].last().to_dict()
    names = held.groupby("ticker")["name"].last().to_dict() if "name" in held else {}
    snaps = all_ticker_snaps()
    for t, s in snaps.items():
        if s.get("currency") == "CAD" and t not in prices and not s.get("converted_from"):
            prices[t] = s["last_close"]
    defaults = ps.get("planner") or {}
    inv = sum(current.values())
    default_targets = defaults.get("targets") or (
        {t: round(v / inv * 100, 1) for t, v in current.items()} if inv else {})
    amount = st.number_input("Monthly contribution (CAD)", min_value=0.0, step=50.0,
                             value=float(defaults.get("monthly_amount") or 500.0), key="pl_amount")
    whole = st.checkbox("Whole shares only", value=False, key="pl_whole",
                        help="Leave unticked if your broker supports fractional shares.")
    scale = 100 if sum(default_targets.values()) <= 1.01 else 1
    edited = st.data_editor(
        pd.DataFrame({"Ticker": list(default_targets), "Target %": [float(v) * scale for v in default_targets.values()]}),
        num_rows="dynamic", hide_index=True, width="stretch", key="pl_targets",
        column_config={"Target %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, format="%.1f")})
    targets = {str(r["Ticker"]).strip().upper(): float(r["Target %"] or 0) for _, r in edited.iterrows()
               if str(r["Ticker"] or "").strip()}
    st.caption(f"Targets add up to {sum(targets.values()):.1f}% (must be 100%). Defaults come from the "
               "`planner:` section of portfolio.yaml.")
    try:
        plan = pl.plan_contribution(current, targets, amount, prices, whole_shares=whole, names=names)
    except pl.PlannerError as exc:
        st.error(str(exc), icon="⛔")
        return
    r = plan.rows
    st.markdown("**Suggested split**")
    st.dataframe(pd.DataFrame({
        "Ticker": r["ticker"], "Buy (CAD)": r["buy"],
        "≈ Shares": [("" if s is None or b <= 0 else (f"{s:,.0f}" if whole else f"{s:,.2f}"))
                     for s, b in zip(r["shares"], r["buy"])],
        "Now %": r["current_weight"], "Target %": r["target_weight"], "After %": r["weight_after"]}),
        hide_index=True, width="stretch", column_config={
            "Buy (CAD)": st.column_config.NumberColumn(format="%.2f"),
            **{c: st.column_config.NumberColumn(format="percent") for c in ("Now %", "Target %", "After %")}})
    months = ("already there" if plan.months_to_target == 0 else
              ("10+ years" if plan.months_to_target is None else f"{plan.months_to_target} months"))
    st.caption(f"Largest gap to target after this buy: {plan.max_drift_after:.1%} "
               f"(was {plan.max_drift_before:.1%}). Within 1 pt of target: {months} at this amount if prices "
               f"don't move. Not invested (rounding): C${plan.leftover:,.2f}.")
    for n in plan.notes:
        st.caption(n)
    st.caption("The split is by ticker across all accounts; which account to buy in depends on your "
               "contribution room, which isn't tracked here.")
    st.markdown("**Timing check**")
    market = snap("market.json")
    flags = pl.risk_flags(list(r["ticker"]), snaps, ps, market, pd.Timestamp.today())
    if flags:
        for f in flags:
            st.warning(f["message"], icon="⚠️")
        sched = pl.tranche_schedule(amount - plan.leftover, pd.Timestamp.today(), market, n=pl.PlannerConfig().tranches)
        st.markdown(f"You could split this buy into {len(sched)} weekly parts, on days without "
                    "high-impact events. Each part uses the same split:")
        st.dataframe(pd.DataFrame({"Date": [d["date"].date() for d in sched], "Amount (CAD)": [d["amount"] for d in sched]}),
                     hide_index=True, width="stretch",
                     column_config={"Amount (CAD)": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.success("No elevated drawdown, volatility, VIX or event-risk flags: one buy is fine by these checks.", icon="✅")
    st.caption("Rough risk checks, not predictions: the forecaster's drawdown probabilities run high, and "
               "splitting a buy mainly smooths the entry price. Not financial advice.")


def tab_reports() -> None:
    header("Reports")
    reps = ad.list_reports(Path(os.environ.get("FORECASTER_REPORTS_DIR") or PROJECT_ROOT / "reports"))
    if not reps:
        st.info("No weekly reports yet. The first one is written on Sunday evening after the forecast rebuild.",
                icon="ℹ️")
        return
    pick = st.selectbox("Report", range(len(reps)), key="report_pick",
                        format_func=lambda i: f"Week ending {reps[i]['date'] - pd.Timedelta(days=2):%b %d, %Y}"
                        + ("  (latest)" if i == 0 else ""))
    r = reps[pick]
    if r["pdf"] is not None:
        st.download_button("Download PDF", r["pdf"].read_bytes(), file_name=r["pdf"].name,
                           mime="application/pdf", width="stretch", key="report_pdf")
    with st.container(border=True, key="report"):
        st.markdown(r["md"].read_text())


TV_SYMBOLS = {"SP:SPX": "S&P 500", "NASDAQ:IXIC": "Nasdaq", "TVC:VIX": "VIX", "OANDA:XAUUSD": "Gold"}


def _fmt_price(sym: str, p: float | None) -> str:
    if p is None:
        return "—"
    return f"{p:,.4f}" if sym in ("USD_CAD",) else f"{p:,.2f}"


def _ms_time(ts_ms: int | None) -> str:
    if not ts_ms:
        return "no tick yet"
    t = pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_convert(ad.NY)
    return f"{t:%H:%M:%S}.{t.microsecond // 1000:03d} ET"


def tab_live() -> None:
    from forecaster import live_feed as lf
    from forecaster.config import load_watchlist

    header("Live")
    view = st.radio("View", ["feed", "tv"], horizontal=True, key="live_view", label_visibility="collapsed",
                    format_func={"feed": "⚡ Live feed", "tv": "📈 TradingView"}.get)
    if view == "tv":
        tradingview_block()
        return
    db = lf.live_db(settings.cache_dir)
    stocks = lf.finnhub_symbols(load_watchlist(PROJECT_ROOT / "watchlist.yaml"))

    @st.fragment(run_every="1s")
    def tiles() -> None:
        now = now_utc()
        chips = []
        for t in lf.tiles(db, now, stocks):
            chg = "" if t.change is None else f"{t.change:+.2%} today"
            label = ("⚠ STALE · " if t.state == "stale" else "") + t.label
            chips.append(f'<div class="qf-chip qf-tile {t.state}" style="flex:1 1 105px">'
                         f'<div class="k">{esc(label)}</div><div class="v">{esc(_fmt_price(t.symbol, t.price))}</div>'
                         f'<div class="d">{esc(chg or t.note)}</div><div class="ms">{esc(_ms_time(t.ts_ms))}</div></div>')
        st.html('<div class="qf-row">' + "".join(chips) + "</div>")
        feeds, gaps = lf.read_feeds(db), lf.read_gaps(db, int((now - pd.Timedelta(hours=24)).value // 1_000_000))
        lines = []
        for name, label in (("finnhub", "Finnhub (stocks)"), ("oanda", "OANDA (gold, FX)")):
            f = feeds.get(name)
            if not f:
                lines.append(f"{label}: not running. It starts with start.bat.")
                continue
            age = "" if not f.get("last_msg_ms") else \
                f" · last message {(now.value // 1_000_000 - f['last_msg_ms']) / 1000:.1f} s ago"
            g = [x for x in gaps if x["feed"] == name]
            gap = f" · {len(g)} gap(s) in 24 h, longest {max(x['seconds'] for x in g):.0f} s" if g else ""
            detail = f" ({f['detail']})" if f.get("detail") and f.get("status") != "connected" else ""
            lines.append(f"{label}: {f['status']}{detail}{age} · {f.get('reconnects') or 0} reconnects{gap}")
        st.caption("  \n".join(lines) + "  \nA tile turns red when its feed has been silent for more than 10 s "
                   "while its market is open. % change is vs the previous close.")

    tiles()

    @st.fragment(run_every="5s")
    def gold() -> None:
        now = now_utc()
        df = lf.vwap_frame(db, now, hours=4)
        g = lf.gold_vwap(db, now)
        st.markdown("**Gold 1-minute · session VWAP ±1σ/±2σ**")
        if df.empty:
            st.caption("No gold bars yet this session (sessions start 18:00 New York; needs the OANDA feed).")
            return
        st.plotly_chart(viz.live_gold_chart(df, PAL), width="stretch",
                        config={"displayModeBar": False, "scrollZoom": False}, key="live_gold_chart")
        if g and g.get("z") is not None:
            st.caption(f"Live {g['price']:,.2f} · VWAP {g['vwap']:,.2f} · {g['z']:+.1f}σ · tick volume "
                       f"(OANDA has no traded volume, same as TradingView's OANDA:XAUUSD). Blue = up candle, "
                       "orange = down. The webhook checks gold alerts against this live price.")

    gold()


def tradingview_block() -> None:
    import json as _json

    import streamlit.components.v1 as components

    sym = st.radio("Symbol", list(TV_SYMBOLS), horizontal=True, key="tv_sym", label_visibility="collapsed",
                   format_func=TV_SYMBOLS.get)
    cfg = {"autosize": True, "symbol": sym, "interval": "1", "timezone": "America/New_York", "theme": "dark",
           "style": "1", "locale": "en", "backgroundColor": "#0d0d0d", "gridColor": "rgba(255,255,255,0.06)",
           "hide_side_toolbar": True, "hide_volume": sym != "OANDA:XAUUSD", "allow_symbol_change": False,
           "save_image": False, "calendar": False, "studies": ["STD;VWAP"] if sym == "OANDA:XAUUSD" else [],
           "support_host": "https://www.tradingview.com"}
    components.html(f"""<div class="tradingview-widget-container" style="height:100%;width:100%">
      <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
        async>{_json.dumps(cfg)}</script></div>""", height=460)
    st.caption("Free TradingView widget, loaded straight from TradingView in your browser (no keys involved). "
               "TradingView decides which feeds are real-time in widgets; delayed ones are marked with a D.")


VERDICT_COLOR = {"EDGE": "#0ca30c", "WEAK": "#fab219", "NO EDGE": "#5d5c58"}


def _bps(x) -> str:
    return "n/a" if x is None else f"{x * 1e4:+.1f} bps"


def _pct(x, d=0) -> str:
    return "n/a" if x is None else f"{x:.{d}%}"


def _costs_txt(c: dict) -> str:
    return (f"commission {c.get('commission', 0):g}/unit per side, spread {c.get('spread', 0):g} "
            f"(or the bar's real bid/ask), slippage {c.get('slippage', 0):g} on market and stop fills")


def _pval(p) -> str:
    return "n/a" if p is None else f"{p:.3f}"


def _ci(ci) -> str:
    return "n/a" if not ci else f"{ci[0] * 1e4:+.1f} to {ci[1] * 1e4:+.1f} bps"


def _verdict_badge(v: str) -> str:
    fg = "#ffffff" if v == "NO EDGE" else "#0b0b0b"
    return f'<span class="qf-badge" style="background:{VERDICT_COLOR.get(v, "#5d5c58")};color:{fg}">{esc(v)}</span>'


def _lab_doc() -> dict | None:
    from forecaster import lab as L

    p = L.lab_dir(settings.cache_dir) / "results.json"
    return _read(str(p), _mtime(p))


def tab_signals() -> None:
    header("Signals")
    view = st.radio("View", ["board", "leaders", "lab"], horizontal=True, key="sig_view", label_visibility="collapsed",
                    format_func={"board": "🎯 Board", "leaders": "🏆 Leaderboard", "lab": "🧪 Lab"}.get)
    doc = _lab_doc()
    if not doc:
        st.info("No Strategy Lab results yet. On the PC run `python -m forecaster.lab download --years 3` "
                "(needs OANDA + Alpaca keys), then `python -m forecaster.lab run`.", icon="🧪")
        return
    if doc.get("synthetic"):
        st.html('<div class="qf-banner" style="background:#8a5a00"><b>⚠ Demo results from synthetic random-walk data</b>'
                "Not real markets: every setup should read NO EDGE here. Download real data and rerun the lab.</div>")
    {"board": signal_board, "leaders": leaderboard_view, "lab": lab_view}[view](doc)


def _setup_card(s: dict, side: int, lines: list[str], when: str, extra: str = "") -> str:
    side_name = "long" if side > 0 else "short"
    o = (s.get("oos_by_side") or {}).get(side_name) or s["oos"]
    ci = (s.get("side_ci") or {}).get(side_name) or s.get("ci")
    v = s["verdict"]
    col = "#3987e5" if side > 0 else "#d95926"
    cls = "qf-setup noedge" if v == "NO EDGE" else "qf-setup"
    pv = _pval(s.get("p_value"))
    stats = (f'<div class="qf-stats"><div>Win rate<b>{_pct(o.get("win_rate"))}</b></div>'
             f'<div>Expectancy<b>{_bps(o.get("expectancy"))}</b></div><div>Sample<b>{o.get("trades", 0)} trades</b></div>'
             f'<div style="grid-column:span 2">95% CI<b>{esc(_ci(ci))}</b></div><div>p vs random<b>{pv}</b></div></div>')
    return (f'<div class="qf-card {cls}" style="border-left:4px solid {col}">'
            f'<div style="display:flex;justify-content:space-between;gap:8px"><div><b>{esc(s["name"])}</b>'
            f'<div class="qf-muted">{esc(s["instrument"])} · {esc(s["timeframe"])} · {esc(when)}</div></div>'
            f'<div>{_verdict_badge(v)}</div></div>'
            f'<div style="margin-top:6px;font-size:0.95rem"><b style="color:{col}">{side_name.upper()}</b> · '
            + " · ".join(esc(x) for x in lines) + f"</div>{stats}{extra}</div>")


def signal_board(doc: dict) -> None:
    from forecaster import shadow as SH

    by_key = {s["key"]: s for s in doc["setups"]}
    db = SH.shadow_db(settings.cache_dir)

    @st.fragment(run_every="5s")
    def board() -> None:
        now = now_utc()
        pending = [p for p in SH.read_pending(db, pd.Timedelta(minutes=20), now) if p["key"] in by_key]
        open_t = SH.read_trades(db, "open")
        rank = {"EDGE": 0, "WEAK": 1, "NO EDGE": 2}
        cards = []
        for p in sorted(pending, key=lambda p: rank.get(by_key[p["key"]]["verdict"], 3)):
            s = by_key[p["key"]]
            rr = abs(p["target"] - p["ref"]) / max(abs(p["ref"] - p["stop"]), 1e-9)
            closed_at = pd.Timestamp(p["signal_time"]) + pd.Timedelta(minutes=int(p["timeframe"].rstrip("mh")) *
                                                                       (60 if p["timeframe"].endswith("h") else 1))
            when = "setup " + ad.ago(closed_at, now) + ", entry at the next bar's open"
            cards.append(_setup_card(s, p["side"], [f"entry ≈ {p['ref']:,.2f}", f"stop {p['stop']:,.2f}",
                                                    f"target {p['target']:,.2f}", f"R:R {rr:.1f}"], when))
        for _, t in (open_t.iterrows() if len(open_t) else []):
            s = by_key.get(t["key"])
            if s is None:
                continue
            cards.append(_setup_card(s, int(t["side"]), [f"in at {t['entry']:,.2f}", f"stop {t['stop']:,.2f}",
                                                         f"target {t['target']:,.2f}"],
                                     "shadow trade open since " + ad.ago(pd.Timestamp(t["entry_time"]), now)))
        hb = SH.heartbeat(db)
        alive = hb is not None and (now - hb) < pd.Timedelta(minutes=2)
        st.caption(("🟢 Shadow trading running" if alive else "⚪ Shadow trading not running (starts with start.bat)")
                   + (f" · last pass {ad.ago(hb, now)}" if hb else "")
                   + ". Setups appear here the moment a strategy triggers on the live feed. Paper only: "
                     "nothing is ever ordered.")
        if cards:
            st.html("".join(cards))
        else:
            st.info("No setups right now. Greyed cards = NO EDGE (historically indistinguishable from luck).", icon="🎯")
        closed = SH.read_trades(db, "closed")
        if len(closed):
            c = closed.head(8)
            st.markdown("**Recent shadow trades**")
            st.html("".join(
                f'<div class="qf-item" style="border-left-color:{"#3987e5" if r.net_ret > 0 else "#d95926"}">'
                f'<div style="min-width:0"><div class="t">{esc(by_key.get(r.key, {}).get("name", r.key))} · '
                f'{esc(r.instrument)} {esc(r.timeframe)}</div><div class="s">{"LONG" if r.side > 0 else "SHORT"} '
                f'{r.entry:,.2f} → {r.exit:,.2f} · {esc(r.reason)}</div></div><div class="r">{_bps(r.net_ret)}</div></div>'
                for r in c.itertuples()))

    board()


def leaderboard_view(doc: dict) -> None:
    from forecaster import lab as L
    from forecaster import shadow as SH

    c1, c2 = st.columns(2)
    inst = c1.selectbox("Instrument", ["All", "SPY", "QQQ", "XAU_USD"], key="lb_inst")
    tf = c2.selectbox("Timeframe", ["All", "5m", "15m", "1m", "1h"], key="lb_tf")
    rows = L.leaderboard(doc, None if inst == "All" else inst, None if tf == "All" else tf)
    live = SH.read_trades(SH.shadow_db(settings.cache_dir), "closed")
    m = rows[0]["setups_tested"] if rows else 0
    st.caption(f"Ranked by out-of-sample expectancy per trade after costs. {m} setups tested, so a few "
               f"will pass p < 0.05 by luck; ★ = also passes the stricter p < {0.05 / max(m, 1):.4f} "
               "(Bonferroni). Greyed = NO EDGE.")
    out = []
    for n, s in enumerate(rows, 1):
        o = s["oos"]
        lt = live[live["key"] == s["key"]]["net_ret"].dropna() if len(live) else pd.Series(dtype=float)
        live_txt = f" · live {_bps(lt.mean())} ({len(lt)})" if len(lt) else ""
        star = " ★" if s.get("bonferroni_ok") else ""
        cls = "qf-setup noedge" if s["verdict"] == "NO EDGE" else "qf-setup"
        out.append(f'<div class="qf-item {cls}" style="border-left-color:{VERDICT_COLOR.get(s["verdict"])}">'
                   f'<div style="min-width:0"><div class="t">{n}. {esc(s["name"])}{star}</div>'
                   f'<div class="s">{esc(s["instrument"])} · {esc(s["timeframe"])} · {o["trades"]} OOS trades · '
                   f'win {_pct(o.get("win_rate"))} · p {_pval(s["p_value"])}'
                   f'{esc(live_txt)}</div></div><div class="r">{_bps(o.get("expectancy"))}<div style="margin-top:3px">'
                   f'{_verdict_badge(s["verdict"])}</div></div></div>')
    st.html("".join(out) if out else '<div class="qf-muted">No setups for this filter.</div>')


def lab_view(doc: dict) -> None:
    from forecaster import lab as L

    rows = L.leaderboard(doc)
    keys = [s["key"] for s in rows]
    by = {s["key"]: s for s in rows}
    k = st.selectbox("Setup", keys, key="lab_pick",
                     format_func=lambda k: f"{by[k]['name']} · {by[k]['instrument']} {by[k]['timeframe']} · {by[k]['verdict']}")
    s = by[k]
    o, i = s["oos"], s["is"]
    st.html(f'<div class="qf-card"><div style="display:flex;justify-content:space-between"><b>{esc(s["name"])}</b>'
            f'{_verdict_badge(s["verdict"])}</div><div class="qf-muted" style="margin-top:4px">'
            + "<br>".join(esc(w) for w in s["why"]) + "</div></div>")
    for w in s.get("warnings", []):
        st.warning(f"Overfitting: {w}", icon="⚠️")
    if s["verdict"] == "EDGE" and not s.get("bonferroni_ok"):
        st.warning(f"Passes p < 0.05 but not the multiple-testing bar for {s['setups_tested']} setups: "
                   "could still be the lucky one in the batch. Watch its shadow trades.", icon="⚠️")
    cells = [("Trades (OOS)", str(o["trades"])), ("Win rate", _pct(o.get("win_rate"))),
             ("Expectancy", _bps(o.get("expectancy"))), ("Avg win", _bps(o.get("avg_win"))),
             ("Avg loss", _bps(o.get("avg_loss"))),
             ("Profit factor", "n/a" if o.get("profit_factor") is None else f"{o['profit_factor']:.2f}"),
             ("Sharpe", "n/a" if o.get("sharpe") is None else f"{o['sharpe']:.2f}"),
             ("Max drawdown", f"{o['max_dd']:.1%}"), ("Losing streak", str(o["losing_streak"])),
             ("95% CI", _ci(s.get("ci"))), ("p vs random", "n/a" if s["p_value"] is None else f"{s['p_value']:.3f}"),
             ("In-sample exp.", _bps(i.get("expectancy")))]
    st.html('<div class="qf-stats" style="margin-bottom:10px">' + "".join(
        f"<div>{esc(a)}<b>{esc(b)}</b></div>" for a, b in cells) + "</div>")
    dd95 = (s.get("mc") or {}).get("dd_p95")
    dd_txt = "n/a" if dd95 is None else f"{dd95:.1%}"
    st.caption(f"Out-of-sample = walk-forward: tuned on 12 months, traded on the next 3 unseen months, rolled "
               f"{len(s['folds'])} times. Costs: {_costs_txt(s['costs'])}. Shuffling the trade order, the worst 5% "
               f"of paths draw down {dd_txt}.")
    if s["random"]["hist"]:
        st.markdown("**This setup vs 1,000 random-entry strategies** (same trades, holding times, costs)")
        st.plotly_chart(viz.lab_random_hist(s["random"]["edges"], s["random"]["hist"], o["expectancy"] or 0, PAL),
                        width="stretch", config={"displayModeBar": False}, key="lab_hist")
    if s["equity"]:
        st.markdown("**Out-of-sample equity** (one unit per trade)")
        st.plotly_chart(viz.lab_equity(s["equity"], PAL), width="stretch", config={"displayModeBar": False},
                        key="lab_eq")
    sl = st.selectbox("Slice results by", ["time_of_day", "weekday", "vix_band", "event_day", "year", "month", "week"],
                      key="lab_slice", format_func=lambda x: {"time_of_day": "Time of day (NY)", "weekday": "Day of week",
                                                             "vix_band": "VIX band (prior close)",
                                                             "event_day": "CPI / PPI / FOMC days", "year": "Year",
                                                             "month": "Month", "week": "Week"}[x])
    rows_s = s["slices"].get(sl) or []
    if sl in ("month", "week") and len(rows_s) > 24:
        rows_s = rows_s[-24:]
        st.caption("Latest 24 shown.")
    if rows_s and not all(r["key"] == "unknown" for r in rows_s):
        st.plotly_chart(viz.lab_slices(rows_s, PAL), width="stretch", config={"displayModeBar": False}, key="lab_sl")
    else:
        st.caption("Not available for this run (VIX history or FRED event dates were missing).")
    with st.expander("Walk-forward windows & parameters"):
        st.dataframe(pd.DataFrame([{"Test window": f"{f['test'][0]} → {f['test'][1]}",
                                    "Chosen": ", ".join(f"{a}={b}" for a, b in f["best"].items()) or "—",
                                    "IS": _bps(f["is_expectancy"]), "OOS": _bps(f["oos_expectancy"]),
                                    "OOS trades": f["oos_trades"]} for f in s["folds"]]),
                     hide_index=True, width="stretch")
        st.caption(f"Tuned: {', '.join(s['tuned']) or 'nothing'} ({s['combos']} combinations per window). "
                   f"Current parameters (used by shadow trading): {esc(str(s['params']))}")
        st.caption(s["description"])


STATE_COLOR = {"built": "good", "failing": "warning", "missing": "critical", "unchecked": "serious"}


def tab_status() -> None:
    from forecaster import project_status as ps

    header("Status")
    s = ps.build(settings.cache_dir)
    running = ps.running(settings.cache_dir)
    st.html(f"""<div class="qf-card" style="border-color:{PAL['s1']}">
      <div class="qf-muted">Next step</div>
      <div style="font-size:1rem;font-weight:600;margin-top:2px">{esc(s['next_step'])}</div></div>""")

    st.markdown(f"**Phase tracker** · Phase {s['phase']}: {ps.PHASES[s['phase'] - 1]}")
    out = []
    for g in s["gates"]:
        cur = g["phase"] == s["phase"]
        col = PAL["s1"] if cur else STATUS["good"] if g["passed"] else "#383835"
        word = "current" if cur else "passed" if g["passed"] else ("your call" if g["phase"] == 5 else "not yet")
        bg = "background:rgba(57,135,229,0.14);" if cur else ""
        out.append(f"""<div class="qf-item" style="border-left-color:{col};{bg}">
          <div style="min-width:0"><div class="t">{ps.gate_icon(g, s['phase'])} {g['phase']} · {esc(g['name'])}</div>
          <div class="s">Gate: {esc(g['gate'])}</div><div class="s" style="white-space:normal">{esc(g['evidence'])}</div></div>
          <div class="r"><span class="qf-badge" style="background:{col};color:{'#fff' if cur else '#0b0b0b'}">{word}</span></div></div>""")
    st.html("".join(out))

    t = s["tests"]
    n_ok = sum(m["state"] == "built" for m in s["modules"])
    if running:
        when = "checking now (about a minute)…"
    elif t and not t.get("note"):
        when = (f"{t['passed']} tests passed" + (f", {t['failed'] + t['errors']} failing" if t["failed"] + t["errors"] else "")
                + f" · checked {ad.ago(pd.Timestamp(s['checked_at']), now_utc())}")
    else:
        when = (t or {}).get("note") or "not checked yet"
    st.markdown(f"**Build progress** · {n_ok}/{len(s['modules'])} modules ✅")
    st.caption(when)
    st.html('<div class="qf-row">' + "".join(
        f'<div class="qf-chip" style="flex:1 1 150px;border-left:3px solid {STATUS[STATE_COLOR[m["state"]]]}">'
        f'<div class="v" style="font-size:0.9rem">{ps.ICON[m["state"]]} {esc(m["name"])}</div>'
        f'<div class="d">{esc(m["detail"])}</div></div>' for m in s["modules"]) + "</div>")
    if t and t.get("failures"):
        with st.expander(f"Failing tests ({len(t['failures'])})"):
            st.code("\n".join(t["failures"][:30]), language=None)
    if st.button("Re-check now (runs every test)", key="recheck", width="stretch", disabled=running):
        with st.spinner("Importing modules and running the test suite…"):
            ps.refresh(settings.cache_dir)
        st.rerun()

    st.markdown("**Data health**")
    now = now_utc()
    st.html("".join(f"""<div class="qf-item" style="border-left-color:{STATUS['good'] if d['ok'] else STATUS['critical']}">
      <div style="min-width:0"><div class="t">{'✅' if d['ok'] else '🔴'} {esc(d['label'])}</div>
      <div class="s" style="white-space:normal">{esc(d['message'])}</div></div>
      <div class="r">{esc(ad.ago(pd.Timestamp(d['last_success']) if d['last_success'] else None, now))}</div></div>"""
                    for d in s["data"]))
    keys = s["keys"]
    st.html('<div class="qf-muted" style="margin:-2px 0 10px">API keys in .env (values never shown): ' + " · ".join(
        f"{esc(k)} {'✅' if v else '🔴 missing'}" for k, v in keys.items() if k in ps.KEYS) + "<br>Demo phase: "
        + " · ".join(f"{esc(k)} {'✅' if v else '— not set'}" for k, v in keys.items() if k in ps.DEMO_KEYS) + "</div>")

    st.markdown("**Model health**")
    if not s["backtests"]:
        st.caption("No backtest yet.")
    out = []
    for b in s["backtests"]:
        col = STATUS["good"] if b["beats_baseline"] else STATUS["warning"]
        out.append(f"""<div class="qf-item" style="border-left-color:{col}">
          <div style="min-width:0"><div class="t">{esc(b['ticker'])}{' (reference run)' if b['reference'] else ''} · {esc(b['period'])}</div>
          <div class="s">P10–P90 coverage {b['coverage']:.0%} (target 80%) · hit rate {b['hit_rate']:.0%}</div>
          <div class="s">Last run {esc(str(b['run_at'])[:10])}</div></div>
          <div class="r"><span class="qf-badge" style="background:{col}">{'beats baseline' if b['beats_baseline'] else 'no edge'}</span></div></div>""")
    st.html("".join(out))
    j, d = s["journal"], s["demo"]
    st.caption(f"Journal: {j['alerts']} alerts, {j['journaled']} with a resolved outcome. Practice account: "
               f"{d['filled']} filled orders, {d['days']} days of P&L history"
               + ("" if d["pnl"] is None else f", P&L {d['pnl']:+,.2f}") + ". Nothing here ever turns on live trading.")
    st.caption("Also written to STATUS.md on every app start and Sunday job; the summary goes to Telegram with "
               "the weekly report.")


# --------------------------------------------------------------------------- #
# Navigation (bottom tab bar, synced with ?tab=)
# --------------------------------------------------------------------------- #

keys = list(TABS)
if "tab" not in st.session_state:
    qp = st.query_params.get("tab")
    st.session_state["tab"] = qp if qp in TABS else "week"
with st.container(key="bottomnav"):
    tab = st.radio("Navigation", keys, key="tab", horizontal=True, label_visibility="collapsed",
                   format_func=lambda k: TABS[k].replace("This Week", "Week").replace("Gold VWAP", "Gold"))
if st.query_params.get("tab") != tab:
    st.query_params["tab"] = tab

{"week": tab_week, "calendar": tab_calendar, "gold": tab_gold, "portfolio": tab_portfolio,
 "reports": tab_reports, "status": tab_status, "live": tab_live, "signals": tab_signals}[tab]()
if tab not in ("portfolio",):
    settings_footer()
