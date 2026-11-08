"""4-quarter-ahead ensemble forecast: HMM regimes + regime/GARCH Monte Carlo +
LightGBM quantile regression, blended into P10/P50/P90 with SHAP drivers.

Targets
-------
Forecast date T. Targets are the closes at the ends of the 4 calendar quarters
*after* the quarter containing T (same labels as model.future_quarter_labels).
All internal maths is in log returns from the close at T.

Components
----------
1. RegimeModel: 3-state Gaussian HMM on [daily log return, log VIX], states
   labelled by rule: high_vol = highest return variance; of the other two,
   bull = higher mean return, bear = lower. "Current regime" uses *filtered*
   probabilities (forward pass only). Smoothed probabilities use later data
   and are only used inside a fit window, to estimate parameters.
2. Monte Carlo: n_paths daily paths. Regimes evolve by the HMM transition
   matrix from the filtered state at T; each regime has its own drift
   (posterior-weighted mean return); noise is GARCH(1,1) with Student-t
   shocks fit on regime-demeaned returns. Gives quantiles per target plus
   P(max drawdown from T exceeds DRAWDOWN_THRESHOLD) by each target date.
3. LightGBM: one quantile model per level (0.1/0.5/0.9) trained on
   month-end decision dates x 4 targets with features for macro, sentiment,
   trend, valuation proxies, regime probabilities and calendar risk density,
   plus days_to_target. SHAP (TreeExplainer) explains each live prediction.

Blend: weighted average of log-return quantiles at each level (quantile
averaging), weights from EnsembleConfig; LightGBM falls back to weight 0 if
it has fewer than `min_train_rows`. Drawdown probability is from the Monte
Carlo (the only component with paths).

No look-ahead, enforced at each layer
------------------------------------
* `PointInTimeData.asof(T)` drops every price after T, every macro value
  first published after T, and sentiment from day T onward. ensemble_forecast
  always slices first; tests check that changing data after T changes
  nothing.
* Macro features use first-release values and their publication dates
  (FRED/ALFRED output_type=4), so they include neither revisions nor
  release lags. If no vintage is available, a conservative fixed lag is used.
  Availability is cummax'd so a row is only usable once every earlier row is.
* Regime features for training rows come from HMMs refit on expanding windows
  at completed quarter-ends. A row at date d only ever uses a model fit on
  data <= d, run forward-only.
* Rolling price features only look backward.
* LightGBM training is purged: a (decision date, target) row is used only if
  the target date <= T, i.e. its outcome was known at T.
* Calendar risk density uses the FRED-scheduled releases (CPI/PPI/jobs/GDP),
  which are published in advance; quarters beyond T's horizon are dropped.
Residual caveats: the ticker universe is today's (survivorship), and
historical FRED release dates are realised rather than originally scheduled.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from forecaster.config import HORIZON_QUARTERS
from forecaster.macro import INDICATORS, SeriesSpec, compute_indicator
from forecaster.market_data import rsi
from forecaster.model import Forecast, QuarterRange, future_quarter_labels

REGIMES: tuple[str, ...] = ("bull", "bear", "high_vol")
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
TRADING_DAYS = 252
FALLBACK_LAG_DAYS = {"M": 45, "Q": 120}

# (FRED series id, indicator column, feature name)
MACRO_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("CPIAUCSL", "yoy", "cpi_yoy"),
    ("CPIAUCSL", "surprise_z", "cpi_surprise_z"),
    ("CPILFESL", "trend_3m", "core_cpi_3m_ann"),
    ("PPIACO", "yoy", "ppi_yoy"),
    ("FEDFUNDS", "value", "fed_funds"),
    ("FEDFUNDS", "yoy", "fed_funds_chg_1y"),
    ("UNRATE", "yoy", "unrate_chg_1y"),
    ("T10Y2Y", "value", "curve_10y2y"),
    ("GDP", "trend_3m", "gdp_qoq_saar"),
    ("UMCSENT", "yoy", "umich_sentiment_yoy"),
)

PRICE_FEATURES = (
    "ret_21d", "ret_63d", "ret_252d", "vol_63d", "close_vs_ma200", "ma50_vs_ma200",
    "rsi14", "drawdown_252d", "valuation_vs_5y", "vix", "vix_vs_1y", "yield_10y",
)
DERIVED_FEATURES = ("real_yield_10y",)
OTHER_FEATURES = ("news_sentiment_30d", "risk_density", "days_to_target")
REGIME_FEATURES = tuple(f"p_{r}" for r in REGIMES)


def feature_names() -> list[str]:
    return [*PRICE_FEATURES, *(n for _, _, n in MACRO_FEATURES), *DERIVED_FEATURES,
            *REGIME_FEATURES, *OTHER_FEATURES]


# --------------------------------------------------------------------------- #
# Point-in-time data
# --------------------------------------------------------------------------- #

def pit_indicator(first_release: pd.DataFrame, spec: SeriesSpec) -> pd.DataFrame:
    """Indicator history (macro.compute_indicator) plus an `available` date.

    `first_release` has columns date, value and optionally available (first
    publication date, from FredClient.get_first_release).
    """
    fr = first_release.dropna(subset=["value"]).sort_values("date")
    values = pd.Series(fr["value"].to_numpy(float), index=pd.DatetimeIndex(fr["date"]))
    values = values[~values.index.duplicated(keep="first")]
    ind = compute_indicator(values, spec)

    if spec.freq == "D":
        # A monthly mean of daily data is known once the month has ended.
        avail = pd.Series(ind.index + pd.offsets.MonthEnd(0), index=ind.index)
    else:
        fallback = pd.Series(ind.index + pd.Timedelta(days=FALLBACK_LAG_DAYS[spec.freq]), index=ind.index)
        if "available" in fr:
            a = pd.Series(pd.to_datetime(fr["available"]).to_numpy(), index=pd.DatetimeIndex(fr["date"]))
            a = a[~a.index.duplicated(keep="first")]
            a = a.resample("MS" if spec.freq == "M" else "QS").max()
            avail = a.reindex(ind.index).fillna(fallback)
        else:
            avail = fallback
    # A row is usable only once it *and every earlier row* has been published.
    ind["available"] = pd.to_datetime(avail).cummax()
    return ind


@dataclass(frozen=True)
class PointInTimeData:
    prices: pd.Series                        # ticker daily close
    vix: pd.Series                           # ^VIX daily close
    tnx: pd.Series | None = None             # ^TNX daily close (10Y yield)
    macro: dict[str, pd.DataFrame] = field(default_factory=dict)  # pit_indicator frames
    sentiment: pd.Series | None = None       # daily market sentiment (news.daily_index)
    risk_points: pd.Series | None = None     # PeriodIndex[Q] -> scheduled-release points
    ticker: str = ""

    def asof(self, t: pd.Timestamp) -> "PointInTimeData":
        """Everything an investor could have known at the close of day t."""
        t = pd.Timestamp(t)
        rp = self.risk_points
        if rp is not None:
            rp = rp[rp.index <= pd.Period(t, freq="Q") + HORIZON_QUARTERS]
        return PointInTimeData(
            prices=self.prices.loc[:t],
            vix=self.vix.loc[:t],
            tnx=None if self.tnx is None else self.tnx.loc[:t],
            macro={k: v[v["available"] <= t] for k, v in self.macro.items()},
            sentiment=None if self.sentiment is None else self.sentiment[self.sentiment.index < t],
            risk_points=rp,
            ticker=self.ticker,
        )


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

def to_yield_percent(tnx: pd.Series) -> pd.Series:
    """^TNX has been quoted as yield x10 on some feeds; normalise to percent."""
    return tnx / 10 if tnx.dropna().median() > 20 else tnx


def _align_past(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Last known value of `s` at or before each date in `index` (forward-fill only)."""
    s = s.dropna().sort_index()
    pos = s.index.searchsorted(index, side="right") - 1
    out = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(out, index=index, dtype=float)


def price_features(close: pd.Series, vix: pd.Series, tnx: pd.Series | None) -> pd.DataFrame:
    """Backward-looking technical / valuation-proxy features on the ticker's dates."""
    c = close.dropna().sort_index()
    lr = np.log(c).diff()
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    ma5y = c.rolling(1260, min_periods=756).mean()
    v = _align_past(vix, c.index)
    f = pd.DataFrame(index=c.index)
    f["ret_21d"] = np.log(c / c.shift(21))
    f["ret_63d"] = np.log(c / c.shift(63))
    f["ret_252d"] = np.log(c / c.shift(252))
    f["vol_63d"] = lr.rolling(63).std() * np.sqrt(TRADING_DAYS)
    f["close_vs_ma200"] = np.log(c / ma200)
    f["ma50_vs_ma200"] = np.log(ma50 / ma200)
    f["rsi14"] = rsi(c)
    f["drawdown_252d"] = c / c.rolling(252, min_periods=63).max() - 1
    f["valuation_vs_5y"] = np.log(c / ma5y)
    f["vix"] = v
    f["vix_vs_1y"] = np.log(v / v.rolling(252, min_periods=126).mean())
    f["yield_10y"] = _align_past(to_yield_percent(tnx), c.index) if tnx is not None else np.nan
    return f


def macro_features(macro: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=dates)
    for sid, col, name in MACRO_FEATURES:
        pit = macro.get(sid)
        if pit is None or pit.empty:
            out[name] = np.nan
            continue
        avail = pit["available"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(avail, dates.to_numpy(dtype="datetime64[ns]"), side="right") - 1
        vals = pit[col].to_numpy(float)
        out[name] = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
    return out


def sentiment_feature(sentiment: pd.Series | None, dates: pd.DatetimeIndex,
                      window_days: int = 30, min_days: int = 5) -> pd.Series:
    """Mean daily market sentiment over [t - window, t), strictly before t."""
    if sentiment is None or sentiment.dropna().empty:
        return pd.Series(np.nan, index=dates)
    s = sentiment.dropna().sort_index()
    out = []
    for t in dates:
        w = s[(s.index >= t - pd.Timedelta(days=window_days)) & (s.index < t)]
        out.append(w.mean() if len(w) >= min_days else np.nan)
    return pd.Series(out, index=dates, dtype=float)


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #

class RegimeModel:
    """3-state Gaussian HMM with deterministic bull/bear/high_vol labelling."""

    def __init__(self, n_init: int = 3, n_iter: int = 200, seed: int = 0) -> None:
        self.n_init, self.n_iter, self.seed = n_init, n_iter, seed

    def fit(self, obs: np.ndarray) -> "RegimeModel":
        from hmmlearn.hmm import GaussianHMM
        from threadpoolctl import threadpool_limits

        obs = np.asarray(obs, float)
        self.center_, self.scale_ = obs.mean(0), obs.std(0)
        self.scale_[self.scale_ == 0] = 1.0
        z = (obs - self.center_) / self.scale_
        best, best_score = None, -np.inf
        logging.getLogger("hmmlearn").setLevel(logging.ERROR)
        for i in range(self.n_init):
            m = GaussianHMM(3, covariance_type="full", n_iter=self.n_iter,
                            random_state=self.seed + i, tol=1e-4)
            # Single-threaded BLAS/OpenMP: multithreaded reductions in the k-means
            # init make fits differ at ~1e-13, which LightGBM splits can amplify
            # and which would break bit-for-bit reproducibility.
            with warnings.catch_warnings(), threadpool_limits(limits=1):
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")
                m.fit(z)
                score = m.score(z)
            if score > best_score:
                best, best_score = m, score
        self.hmm_ = best
        self.log_likelihood_ = best_score

        means, covars = best.means_, best.covars_
        hv = int(np.argmax(covars[:, 0, 0]))
        rest = sorted((s for s in range(3) if s != hv), key=lambda s: -means[s, 0])
        self.order_ = [rest[0], rest[1], hv]  # bull, bear, high_vol
        o = self.order_
        self.transmat_ = best.transmat_[np.ix_(o, o)]
        self.startprob_ = best.startprob_[o]
        self.means_, self.covars_ = means[o], covars[o]
        return self

    def _log_emissions(self, obs: np.ndarray) -> np.ndarray:
        z = (np.asarray(obs, float) - self.center_) / self.scale_
        return np.column_stack([
            np.atleast_1d(multivariate_normal(m, c, allow_singular=True).logpdf(z))
            for m, c in zip(self.means_, self.covars_)
        ])

    def filter(self, obs: np.ndarray) -> np.ndarray:
        """Filtered P(state_t | obs_1..t), forward pass only, label order."""
        logb = self._log_emissions(obs)
        b = np.exp(logb - logb.max(1, keepdims=True))
        out = np.empty_like(b)
        a = self.startprob_ * b[0]
        out[0] = a / a.sum() if a.sum() > 0 else np.full(3, 1 / 3)
        A = self.transmat_
        for t in range(1, len(b)):
            a = (out[t - 1] @ A) * b[t]
            s = a.sum()
            out[t] = a / s if s > 0 else np.full(3, 1 / 3)
        return out

    def smooth(self, obs: np.ndarray) -> np.ndarray:
        """Smoothed posteriors (uses the whole window): parameter estimation only."""
        z = (np.asarray(obs, float) - self.center_) / self.scale_
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")
            return self.hmm_.predict_proba(z)[:, self.order_]


def regime_observations(close: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """[log return, log VIX] on dates where both exist (prices aligned first)."""
    px = pd.concat({"close": close, "vix": vix}, axis=1, join="inner").dropna()
    px = px[(px > 0).all(axis=1)]
    obs = pd.DataFrame({"ret": np.log(px["close"]).diff(), "log_vix": np.log(px["vix"])})
    return obs.dropna()


def refit_dates(index: pd.DatetimeIndex, asof: pd.Timestamp, every: int, min_obs: int) -> list[pd.Timestamp]:
    """Last trading day of each *completed* quarter (quarter end <= asof), every
    `every` quarters, once at least `min_obs` observations exist."""
    s = pd.Series(np.arange(len(index)), index=index)
    q_last = s.groupby(index.to_period("Q")).tail(1)
    done = [d for d in q_last.index
            if pd.Period(d, freq="Q").end_time.normalize() <= asof and q_last[d] + 1 >= min_obs]
    return done[::every]


def regime_feature_table(obs: pd.DataFrame, asof: pd.Timestamp, every: int = 4,
                         min_obs: int = 750, n_init: int = 3, n_iter: int = 200,
                         seed: int = 0) -> pd.DataFrame:
    """Filtered regime probabilities where each row uses an HMM fit only on
    data up to the most recent refit date <= that row. NaN before first fit."""
    out = pd.DataFrame(np.nan, index=obs.index, columns=list(REGIME_FEATURES))
    refits = refit_dates(obs.index, asof, every, min_obs)
    arr = obs.to_numpy()
    for i, r in enumerate(refits):
        nxt = refits[i + 1] if i + 1 < len(refits) else None
        r_pos = obs.index.get_loc(r)
        model = RegimeModel(n_init, n_iter, seed).fit(arr[: r_pos + 1])
        end_pos = obs.index.get_loc(nxt) if nxt is not None else len(obs)
        probs = model.filter(arr[:end_pos])
        out.iloc[r_pos:end_pos] = probs[r_pos:end_pos]
    return out


# --------------------------------------------------------------------------- #
# GARCH + Monte Carlo
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GarchParams:
    omega: float      # daily, decimal^2
    alpha: float
    beta: float
    nu: float         # Student-t degrees of freedom
    next_var: float   # one-step-ahead conditional variance at T (decimal^2)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        """Annualised unconditional volatility."""
        p = self.persistence
        return float(np.sqrt(self.omega / (1 - p) * TRADING_DAYS)) if p < 1 else float("nan")


def fit_garch(resid: np.ndarray) -> GarchParams:
    from arch import arch_model

    y = np.asarray(resid, float) * 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # arch ConvergenceWarning on short/flat samples
        res = arch_model(y, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
        fvar = float(res.forecast(horizon=1, reindex=False).variance.to_numpy()[-1, 0])
    p = res.params
    return GarchParams(
        omega=float(p["omega"]) / 1e4, alpha=float(p["alpha[1]"]), beta=float(p["beta[1]"]),
        nu=float(p["nu"]), next_var=fvar / 1e4,
    )


def simulate_paths(
    drift: np.ndarray, transmat: np.ndarray, p0: np.ndarray, garch: GarchParams,
    target_steps: list[int], n_paths: int, rng: np.random.Generator,
    drawdown_threshold: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Regime-switching GARCH-t paths of daily log returns.

    Returns (log return from T at each target step [n_paths, n_targets],
             P(max drawdown from T through each target < -threshold) [n_targets]).
    """
    k = len(drift)
    n_steps = max(target_steps)
    cum_tm = np.cumsum(transmat, axis=1)
    cum_tm[:, -1] = 1.0
    state = rng.choice(k, size=n_paths, p=p0 / p0.sum())
    var = np.full(n_paths, garch.next_var)
    t_scale = np.sqrt((garch.nu - 2) / garch.nu)
    cum = np.zeros(n_paths)
    run_max = np.zeros(n_paths)       # log of running peak, start price included
    worst_dd = np.zeros(n_paths)      # min of log(P / peak)
    out_ret = np.empty((n_paths, len(target_steps)))
    out_dd = np.empty(len(target_steps))
    targets = {s: i for i, s in enumerate(target_steps)}
    log_thr = np.log(1 - drawdown_threshold)
    for step in range(1, n_steps + 1):
        u = rng.random(n_paths)
        state = np.minimum((u[:, None] > cum_tm[state]).sum(1), k - 1)
        eps = np.sqrt(var) * rng.standard_t(garch.nu, n_paths) * t_scale
        cum += drift[state] + eps
        var = garch.omega + garch.alpha * eps ** 2 + garch.beta * var
        run_max = np.maximum(run_max, cum)
        worst_dd = np.minimum(worst_dd, cum - run_max)
        if step in targets:
            i = targets[step]
            out_ret[:, i] = cum
            out_dd[i] = float(np.mean(worst_dd < log_thr))
    return out_ret, out_dd


# --------------------------------------------------------------------------- #
# Training set
# --------------------------------------------------------------------------- #

def target_dates(t: pd.Timestamp, n: int = HORIZON_QUARTERS) -> list[pd.Timestamp]:
    """Ends of the n quarters after the quarter containing t."""
    q = pd.Period(pd.Timestamp(t), freq="Q")
    return [(q + k).end_time.normalize() for k in range(1, n + 1)]


def business_days_between(t: pd.Timestamp, d: pd.Timestamp) -> int:
    return len(pd.bdate_range(pd.Timestamp(t) + pd.Timedelta(days=1), d))


def feature_frame(d: PointInTimeData, dates: pd.DatetimeIndex, regimes: pd.DataFrame) -> pd.DataFrame:
    """Decision-time features for each date (target-independent part)."""
    pf = price_features(d.prices, d.vix, d.tnx)
    pos = pf.index.searchsorted(dates, side="right") - 1
    base = pf.iloc[np.clip(pos, 0, None)].set_axis(dates)
    base[pos < 0] = np.nan
    mf = macro_features(d.macro, dates)
    feats = pd.concat([base, mf], axis=1)
    feats["real_yield_10y"] = feats["yield_10y"] - feats["cpi_yoy"]
    rpos = regimes.index.searchsorted(dates, side="right") - 1
    reg = regimes.iloc[np.clip(rpos, 0, None)].set_axis(dates)
    reg[rpos < 0] = np.nan
    feats = pd.concat([feats, reg], axis=1)
    feats["news_sentiment_30d"] = sentiment_feature(d.sentiment, dates).to_numpy()
    return feats


def _risk(rp: pd.Series | None, target: pd.Timestamp) -> float:
    if rp is None:
        return np.nan
    q = pd.Period(target, freq="Q")
    return float(rp.get(q, np.nan))


def build_training_set(d: PointInTimeData, T: pd.Timestamp, regimes: pd.DataFrame,
                       min_history: int = 252) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Month-end decision dates x 4 targets, purged to targets realised by T.

    Returns (X, y, meta) where meta has decision_date and target_date.
    """
    close = d.prices.dropna().sort_index()
    if len(close) <= min_history:
        empty = pd.DataFrame(columns=feature_names())
        return empty, pd.Series(dtype=float), pd.DataFrame(columns=["decision_date", "target_date"])
    s = pd.Series(close.index, index=close.index)
    month_ends = pd.DatetimeIndex(s.groupby(close.index.to_period("M")).last().to_numpy())
    month_ends = month_ends[month_ends >= close.index[min_history]]

    rows, ys, meta = [], [], []
    feats = feature_frame(d, month_ends, regimes)
    for t in month_ends:
        p_t = close.loc[t]
        for D in target_dates(t):
            if D > T:  # purge: outcome not yet known at T
                continue
            p_d = close.loc[:D]
            if p_d.empty or p_d.index[-1] <= pd.Period(D, freq="Q").start_time:
                continue
            row = feats.loc[t].copy()
            row["risk_density"] = _risk(d.risk_points, D)
            row["days_to_target"] = (D - t).days
            rows.append(row)
            ys.append(np.log(p_d.iloc[-1] / p_t))
            meta.append({"decision_date": t, "target_date": D})
    X = pd.DataFrame(rows, columns=feature_names()).reset_index(drop=True).astype(float)
    return X, pd.Series(ys, dtype=float, name="log_return"), pd.DataFrame(meta)


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EnsembleConfig:
    n_paths: int = 10_000
    seed: int = 42
    hmm_n_init: int = 3
    hmm_n_iter: int = 200
    hmm_refit_every: int = 4        # quarters between expanding-window refits
    hmm_min_obs: int = 750          # ~3 years of daily data before the first fit
    hmm_fit_window: int | None = None  # live HMM/GARCH fit: last N obs (None = all)
    w_mc: float = 0.5
    w_lgb: float = 0.5
    min_train_rows: int = 60
    drawdown_threshold: float = 0.10
    lgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    ))


@dataclass(frozen=True)
class EnsembleResult:
    forecast: Forecast              # blended P10/P50/P90 prices
    table: pd.DataFrame             # per quarter: component quantiles, blend, drawdown prob
    prob_drawdown: float            # P(>threshold drawdown within the 4-quarter horizon)
    regime: dict                    # current filtered probs, drifts, transition matrix
    garch: GarchParams
    drivers: pd.DataFrame           # SHAP values per quarter x quantile x feature
    weights: dict[str, float]
    n_train_rows: int
    as_of: pd.Timestamp

    def top_drivers(self, n: int = 5, quantile: float = 0.5) -> pd.DataFrame:
        if self.drivers.empty:
            return self.drivers
        d = self.drivers[self.drivers["quantile"] == quantile].copy()
        d["abs"] = d["shap"].abs()
        d = d.sort_values(["quarter", "abs"], ascending=[True, False])
        return d.groupby("quarter").head(n).drop(columns="abs").reset_index(drop=True)


def _fit_lgb(X: pd.DataFrame, y: pd.Series, cfg: EnsembleConfig) -> dict[float, object]:
    import lightgbm as lgb

    models = {}
    for q in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=cfg.seed,
                              deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1,
                              **cfg.lgb_params)
        m.fit(X, y)
        models[q] = m
    return models


def _shap(models: dict[float, object], X_live: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    import shap

    rows = []
    for q, m in models.items():
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(X_live)
        base = float(np.ravel(ex.expected_value)[0])
        pred = m.predict(X_live)  # raw (pre-sort) prediction = base + sum(shap)
        for i, lab in enumerate(labels):
            for j, f in enumerate(X_live.columns):
                rows.append({"quarter": lab, "quantile": q, "feature": f,
                             "value": X_live.iloc[i, j], "shap": float(sv[i, j]), "base": base,
                             "prediction": float(pred[i])})
    return pd.DataFrame(rows)


def ensemble_forecast(data: PointInTimeData, asof: pd.Timestamp | None = None,
                      config: EnsembleConfig = EnsembleConfig(),
                      regimes: pd.DataFrame | None = None) -> EnsembleResult:
    """Forecast as of `asof`.

    `regimes`: optional precomputed regime_feature_table for the same data
    and config (e.g. built once for a backtest). Only rows <= asof are used.
    Rows <= T are identical to a fresh computation on data cut at T, which
    test_regime_table_has_no_lookahead checks, so this is a pure speed-up.
    """
    cfg = config
    T = pd.Timestamp(asof if asof is not None else data.prices.index.max()).normalize()
    d = data.asof(T)                                   # <- the one door to the data
    close = d.prices.dropna().sort_index()
    if close.empty:
        raise ValueError("No prices on or before the as-of date")
    p_T = float(close.iloc[-1])
    labels = future_quarter_labels(T)
    targets = target_dates(T)
    steps = [business_days_between(T, D) for D in targets]

    # 1) regimes (live fit on data <= T)
    obs = regime_observations(close, d.vix)
    if len(obs) < cfg.hmm_min_obs:
        raise ValueError(f"Need >= {cfg.hmm_min_obs} daily observations with VIX, got {len(obs)}")
    fit_obs = obs if cfg.hmm_fit_window is None else obs.iloc[-cfg.hmm_fit_window:]
    rm = RegimeModel(cfg.hmm_n_init, cfg.hmm_n_iter, cfg.seed).fit(fit_obs.to_numpy())
    filt = rm.filter(fit_obs.to_numpy())
    gamma = rm.smooth(fit_obs.to_numpy())
    r = fit_obs["ret"].to_numpy()
    drift = (gamma * r[:, None]).sum(0) / np.maximum(gamma.sum(0), 1e-12)
    p_now = filt[-1]

    # 2) GARCH on regime-demeaned returns + Monte Carlo
    garch = fit_garch(r - gamma @ drift)
    rng = np.random.default_rng(cfg.seed)
    mc_ret, mc_dd = simulate_paths(drift, rm.transmat_, p_now, garch, steps, cfg.n_paths, rng,
                                   cfg.drawdown_threshold)
    mc_q = np.quantile(mc_ret, QUANTILES, axis=0).T            # [n_targets, 3]

    # 3) LightGBM quantiles on purged training rows
    if regimes is None:
        regimes = regime_feature_table(obs, T, cfg.hmm_refit_every, cfg.hmm_min_obs,
                                       cfg.hmm_n_init, cfg.hmm_n_iter, cfg.seed)
    else:
        regimes = regimes.loc[:T]
    X, y, meta = build_training_set(d, T, regimes)
    assert meta.empty or (meta["target_date"] <= T).all(), "look-ahead: unrealised target in training"

    X_live = feature_frame(d, pd.DatetimeIndex([T] * len(targets)), regimes)
    X_live["risk_density"] = [_risk(d.risk_points, D) for D in targets]
    X_live["days_to_target"] = [(D - T).days for D in targets]
    X_live = X_live[feature_names()].reset_index(drop=True).astype(float)

    use_lgb = len(X) >= cfg.min_train_rows and cfg.w_lgb > 0
    if use_lgb:
        models = _fit_lgb(X, y, cfg)
        lgb_q = np.sort(np.column_stack([models[q].predict(X_live) for q in QUANTILES]), axis=1)
        drivers = _shap(models, X_live, labels)
        w = {"mc": cfg.w_mc / (cfg.w_mc + cfg.w_lgb), "lgb": cfg.w_lgb / (cfg.w_mc + cfg.w_lgb)}
    else:
        lgb_q = np.full_like(mc_q, np.nan)
        drivers = pd.DataFrame(columns=["quarter", "quantile", "feature", "value", "shap", "base",
                                        "prediction"])
        w = {"mc": 1.0, "lgb": 0.0}

    blend = mc_q * w["mc"] + (np.nan_to_num(lgb_q) * w["lgb"] if use_lgb else 0)
    blend = np.sort(blend, axis=1)
    prices = p_T * np.exp(blend)

    table = pd.DataFrame({
        "quarter": labels, "target_date": targets, "trading_days": steps,
        **{f"mc_q{int(q * 100)}": mc_q[:, i] for i, q in enumerate(QUANTILES)},
        **{f"lgb_q{int(q * 100)}": lgb_q[:, i] for i, q in enumerate(QUANTILES)},
        **{f"blend_q{int(q * 100)}": blend[:, i] for i, q in enumerate(QUANTILES)},
        "p10": prices[:, 0], "p50": prices[:, 1], "p90": prices[:, 2],
        "prob_drawdown": mc_dd,
    }).set_index("quarter")

    fc = Forecast(
        ticker=data.ticker or str(close.name or ""),
        as_of=T,
        last_close=p_T,
        model=f"ensemble(mc={w['mc']:.2f}, lgb={w['lgb']:.2f}, paths={cfg.n_paths})",
        quarters=tuple(QuarterRange(lab, *map(float, prices[i])) for i, lab in enumerate(labels)),
    )
    regime = {
        "current": dict(zip(REGIMES, map(float, p_now))),
        "drift_annualized": dict(zip(REGIMES, map(float, drift * TRADING_DAYS))),
        "transmat": pd.DataFrame(rm.transmat_, index=REGIMES, columns=REGIMES),
    }
    return EnsembleResult(fc, table, float(mc_dd[-1]), regime, garch, drivers, w, len(X), T)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def load_data(ticker: str, settings, on_warn=print) -> PointInTimeData:  # pragma: no cover
    """Assemble point-in-time inputs from the project's caches/APIs."""
    from forecaster.cache import Cache
    from forecaster.calendar import MACRO_RELEASES, fred_release_events
    from forecaster.market_data import ParquetStore
    from forecaster.news import MARKET_TAG, NewsStore, daily_index
    from forecaster.sources.fred import FredClient

    store = ParquetStore(settings.cache_dir / "ohlcv")
    close = store.get(ticker)["Close"]
    vix = store.get("^VIX")["Close"]
    try:
        tnx = store.get("^TNX")["Close"]
    except Exception as exc:
        on_warn(f"^TNX unavailable: {exc}")
        tnx = None

    macro: dict[str, pd.DataFrame] = {}
    risk = None
    if settings.fred_api_key:
        fred = FredClient(settings.fred_api_key, Cache(settings.cache_dir, settings.cache_ttl_hours * 3600))
        for spec in INDICATORS:
            try:
                macro[spec.series_id] = pit_indicator(fred.get_first_release(spec.series_id), spec)
            except Exception as exc:
                on_warn(f"{spec.series_id}: {exc}")
        try:
            ev = fred_release_events(fred, close.index.min(), pd.Timestamp.today() + pd.DateOffset(years=2),
                                     MACRO_RELEASES, on_error=lambda c, e: on_warn(f"{c}: {e}"),
                                     limit=2000)  # decades of monthly releases
            s = pd.Series([e.points for e in ev], index=pd.PeriodIndex([e.date for e in ev], freq="Q"))
            risk = s.groupby(level=0).sum()
        except Exception as exc:
            on_warn(f"risk density: {exc}")
    else:
        on_warn("FRED_API_KEY not set: macro and risk-density features will be missing")

    sentiment = None
    news_db = settings.cache_dir / "news.sqlite"
    if news_db.exists():
        di = daily_index(NewsStore(news_db).scored_frame())
        if MARKET_TAG in di.index.get_level_values("tag"):
            sentiment = di.loc[MARKET_TAG]["sentiment"]
    return PointInTimeData(close, vix, tnx, macro, sentiment, risk, ticker)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    import argparse
    import sys

    from forecaster.config import load_settings

    ap = argparse.ArgumentParser(description="Ensemble 4-quarter forecast")
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--asof", help="forecast as of this date (YYYY-MM-DD), for backtests")
    ap.add_argument("--paths", type=int, default=10_000)
    args = ap.parse_args(argv)

    settings = load_settings()
    warn = lambda m: print(f"[warn] {m}", file=sys.stderr)  # noqa: E731
    for t in args.tickers or settings.tickers:
        try:
            res = ensemble_forecast(load_data(t, settings, warn), args.asof,
                                    EnsembleConfig(n_paths=args.paths))
        except Exception as exc:
            warn(f"{t}: {exc}")
            continue
        print(f"\n=== {t}  as of {res.as_of.date()}  last {res.forecast.last_close:,.2f}  "
              f"weights {res.weights}  train rows {res.n_train_rows}")
        print("regime now:", {k: round(v, 2) for k, v in res.regime["current"].items()})
        print(res.table[["target_date", "p10", "p50", "p90", "prob_drawdown"]].round(3))
        print(f"P(>{EnsembleConfig().drawdown_threshold:.0%} drawdown within 4 quarters): {res.prob_drawdown:.1%}")
        if not res.drivers.empty:
            print("top P50 drivers:")
            print(res.top_drivers(3)[["quarter", "feature", "value", "shap"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
