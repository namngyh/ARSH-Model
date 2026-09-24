"""ARSH v0.5 — lock VN30F1M data, return horizon, distribution and K."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import html
from itertools import combinations
import json
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from runtime import RegimeRuntime, emission_log_prob
from student_t_hmm import GaussianSequenceHMM, StudentTHMM


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "ohlc_export.csv"
HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90)
K_VALUES = (2, 3, 4, 5, 6, 7)
STUDENT_FAMILIES = ("student_t_shared", "student_t_state")
ALL_HMM_FAMILIES = ("gaussian_hmm",) + STUDENT_FAMILIES
SESSIONS = (("AM", "09:00", "11:30"), ("PM", "13:00", "14:30"))
POLICIES = ("continuous_carry", "daily_sequence", "session_sequence")
BACKENDS = ("cpu", "cuda")
RUN_DEADLINE = None


class TimeBudgetReached(RuntimeError):
    pass


def check_time_budget():
    if RUN_DEADLINE is not None and time.monotonic() >= RUN_DEADLINE:
        raise TimeBudgetReached


def emit_progress(message: str, cache_dir: Path | None = None):
    line=f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line,flush=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True,exist_ok=True)
        with (cache_dir.parent/"progress.log").open("a",encoding="utf8") as handle:
            handle.write(line+"\n")


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_joblib_dump(value, path: Path):
    """Write checkpoints atomically so an interrupted write is never treated as complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def load_minutes(path: Path):
    raw = pd.read_csv(path, dtype={"TRADING_DATE": str})
    required = {"SYMBOL", "TRADING_DATE", "TRADING_TIME", "OPEN_PX", "HIGH_PX", "LOW_PX", "CLOSE_PX"}
    if not required <= set(raw):
        raise ValueError(f"Missing columns: {sorted(required-set(raw))}")
    raw = raw.loc[raw.SYMBOL.eq("VN30F1M")].copy()
    raw.index = pd.DatetimeIndex(pd.to_datetime(
        raw.TRADING_DATE + " " + raw.TRADING_TIME,
        format="%Y%m%d %H:%M:%S", errors="raise"), name="datetime")
    raw = raw.sort_index()
    prices = raw[["OPEN_PX", "HIGH_PX", "LOW_PX", "CLOSE_PX"]]
    bad = (~np.isfinite(prices).all(axis=1) | prices.le(0).any(axis=1)
           | raw.HIGH_PX.lt(prices.max(axis=1)) | raw.LOW_PX.gt(prices.min(axis=1)))
    if raw.empty or raw.index.has_duplicates or bad.any():
        raise ValueError(f"Data validation failed: duplicates={raw.index.duplicated().sum()}, bad={bad.sum()}")
    audit = {"source": str(path.resolve()), "source_sha256": sha256(path), "rows": len(raw),
             "first": str(raw.index[0]), "last": str(raw.index[-1]),
             "days": int(raw.index.normalize().nunique()), "missing_cells": int(raw.isna().sum().sum())}
    return raw, audit


def source_structure_audit(raw: pd.DataFrame):
    """Audit timestamp regimes and likely front-contract rollover effects."""
    daily=raw.groupby(raw.index.normalize()).agg(
        rows=("CLOSE_PX","size"),day_open=("OPEN_PX","first"),
        day_close=("CLOSE_PX","last"),first_time=("TRADING_TIME","min"),
        last_time=("TRADING_TIME","max"))
    for clock in ("11:30:00","14:30:00","14:45:00"):
        present=raw.TRADING_TIME.eq(clock).groupby(raw.index.normalize()).any()
        daily[f"has_{clock[:5].replace(':','')}"]=present.reindex(daily.index,fill_value=False)
    daily["timestamp_pattern"]=(daily[["has_1130","has_1430","has_1445"]]
                                .astype(int).astype(str).agg("".join,axis=1))
    daily["previous_date"]=daily.index.to_series().shift()
    daily["previous_close"]=daily.day_close.shift()
    daily["overnight_log_gap"]=np.log(daily.day_open/daily.previous_close)
    daily["previous_is_third_thursday"]=(daily.previous_date.dt.weekday.eq(3)
                                           & daily.previous_date.dt.day.between(15,21))
    daily["absolute_gap"]=daily.overnight_log_gap.abs()
    expiry=daily[daily.previous_is_third_thursday].absolute_gap.dropna()
    ordinary=daily[~daily.previous_is_third_thursday].absolute_gap.dropna()
    before=daily[daily.index<pd.Timestamp("2025-05-05")]
    after=daily[daily.index>=pd.Timestamp("2025-05-05")]
    summary={
        "single_continuous_symbol_only":bool(raw.SYMBOL.nunique()==1 and raw.SYMBOL.iloc[0]=="VN30F1M"),
        "contract_identifier_available":False,
        "duplicate_timestamps":int(raw.index.duplicated().sum()),
        "modal_rows_per_day_before_2025_05_05":int(before.rows.mode().iloc[0]),
        "modal_rows_per_day_from_2025_05_05":int(after.rows.mode().iloc[0]),
        "modal_pattern_before_2025_05_05":str(before.timestamp_pattern.mode().iloc[0]),
        "modal_pattern_from_2025_05_05":str(after.timestamp_pattern.mode().iloc[0]),
        "pattern_definition":"has_1130,has_1430,has_1445",
        "expiry_transition_count":int(len(expiry)),
        "expiry_abs_gap_median":float(expiry.median()),
        "ordinary_abs_gap_median":float(ordinary.median()),
        "expiry_abs_gap_p95":float(expiry.quantile(.95)),
        "ordinary_abs_gap_p95":float(ordinary.quantile(.95)),
        "inference":"Timestamp structure changes at 2025-05-05; expiry-adjacent gaps are larger. Provider convention cannot be proven without contract-level/tick data."
    }
    return daily.reset_index(names="date"),summary


def _stamp(day, clock):
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {clock}")


def build_returns(raw: pd.DataFrame, horizons=HORIZONS, overlapping=False,
                  allow_one_internal_missing=False):
    records = {h: [] for h in horizons}; rejected = []; gap_rows = []
    grouped = list(raw.groupby(raw.index.normalize()))
    previous_close = None; previous_day = None
    for day, group in grouped:
        close = group.CLOSE_PX
        open_price = close.get(_stamp(day, "09:00"), np.nan)
        if previous_close is not None and np.isfinite(open_price):
            gap_rows.append({"date": day, "gap_type": "overnight_open",
                             "log_return": float(np.log(open_price / previous_close))})
        for session, start_clock, end_clock in SESSIONS:
            anchor, session_end = _stamp(day, start_clock), _stamp(day, end_clock)
            expected_session = pd.date_range(anchor, session_end, freq="min")
            session_close = close.reindex(expected_session)
            valid = session_close.notna().to_numpy().astype(int)
            cumulative = np.r_[0, np.cumsum(valid)]
            for horizon in horizons:
                step=1 if overlapping else horizon
                starts = np.arange(0, len(expected_session)-horizon, step, dtype=int)
                ends = starts+horizon
                observed = cumulative[ends+1]-cumulative[starts]
                endpoints_present = np.isfinite(p0 := session_close.iloc[starts].to_numpy()) & \
                                    np.isfinite(p1 := session_close.iloc[ends].to_numpy())
                missing_count = (horizon+1)-observed
                accepted = endpoints_present & (missing_count == 0)
                if allow_one_internal_missing:
                    # Sensitivity only: a close-to-close return needs its two
                    # endpoints, so one absent interior quote can be admitted
                    # without imputing a price.
                    accepted |= endpoints_present & (missing_count == 1)
                start_time = expected_session[starts]; end_time = expected_session[ends]
                remaining = (len(expected_session)-1)-ends
                bucket = np.where(starts<30,"open_30m",np.where(remaining<30,"close_30m","middle"))
                if accepted.any():
                    rollover_adjacent = bool(previous_day is not None
                        and pd.Timestamp(previous_day).weekday() == 3
                        and 15 <= pd.Timestamp(previous_day).day <= 21)
                    frame = pd.DataFrame({"datetime":end_time[accepted],"date":day,"session":session,
                        "window_start":start_time[accepted],"horizon_min":horizon,
                        "bar_number":np.arange(len(starts))[accepted],
                        "time_bucket":[f"{session}_{x}" for x in bucket[accepted]],
                        "rollover_adjacent":rollover_adjacent,
                        "internal_missing_minutes":missing_count[accepted],
                        "start_close":p0[accepted],"close":p1[accepted],
                        "log_return":np.log(p1[accepted]/p0[accepted])})
                    records[horizon].append(frame)
                for j in np.flatnonzero(~accepted):
                    rejected.append({"date":day,"session":session,"horizon_min":horizon,
                                     "start":start_time[j],"end":end_time[j],
                                     "observed_prices":int(observed[j]),"expected_prices":horizon+1,
                                     "reason":"incomplete_strict_window"})
        endpoints = {clock: close.get(_stamp(day, clock), np.nan)
                     for clock in ("11:30", "13:00", "14:30", "14:45")}
        for name, left, right in (("lunch", "11:30", "13:00"),
                                  ("closing_auction", "14:30", "14:45")):
            if np.isfinite(endpoints[left]) and np.isfinite(endpoints[right]):
                gap_rows.append({"date": day, "gap_type": name,
                                 "log_return": float(np.log(endpoints[right] / endpoints[left]))})
        if np.isfinite(endpoints["14:45"]):
            previous_close = endpoints["14:45"]
        elif np.isfinite(endpoints["14:30"]):
            previous_close = endpoints["14:30"]
        previous_day = day
    frames = {h: pd.concat(rows,ignore_index=True).set_index("datetime").sort_index()
              for h, rows in records.items()}
    audits = []
    for h, frame in frames.items():
        x = frame.log_return
        audits.append({"horizon_min": h, "accepted_returns": len(frame),
                       "rejected_windows": sum(row["horizon_min"] == h for row in rejected),
                       "first": str(frame.index.min()), "last": str(frame.index.max()),
                       "zero_rate": float(x.eq(0).mean()), "mean": float(x.mean()),
                       "std": float(x.std(ddof=1)), "skew": float(x.skew()),
                       "excess_kurtosis": float(x.kurt()), "q01": float(x.quantile(.01)),
                       "q99": float(x.quantile(.99)),
                       "outliers_abs_gt_5std": int((x.abs() > 5*x.std(ddof=1)).sum())})
    return frames, pd.DataFrame(audits), pd.DataFrame(rejected), pd.DataFrame(gap_rows)


def intraday_adjust_split(train, validation, test, shrinkage=20.0):
    """Remove train-estimated intraday volatility seasonality without leakage."""
    keys=["session","bar_number"]
    overall=float(train.log_return.std(ddof=0))
    if not np.isfinite(overall) or overall<=0:raise ValueError("invalid training volatility")
    grouped=train.groupby(keys).log_return.agg(["std","count"])
    grouped["variance"]=(grouped["count"]*grouped["std"].fillna(overall).pow(2)
                         +shrinkage*overall**2)/(grouped["count"]+shrinkage)
    grouped["factor"]=(np.sqrt(grouped["variance"])/overall).clip(.25,4.0)
    factors=grouped["factor"]
    adjusted=[]
    for frame in (train,validation,test):
        out=frame.copy();idx=pd.MultiIndex.from_frame(out[keys])
        factor=factors.reindex(idx).fillna(1.0).to_numpy()
        out["raw_log_return"]=out.log_return
        out["intraday_volatility_factor"]=factor
        out["log_return"]=out.log_return/factor
        adjusted.append(out)
    return tuple(adjusted)


def robust_filter_split(train, validation, test, exclude_zero=False,
                        outlier_policy="keep", exclude_rollover_adjacent=False):
    """Apply sensitivity-only filters using thresholds learned from train."""
    if outlier_policy not in {"keep", "exclude_train_5sigma"}:
        raise ValueError(f"unknown outlier policy: {outlier_policy}")
    cutoff = np.inf
    if outlier_policy == "exclude_train_5sigma":
        cutoff = 5.0 * float(train.log_return.std(ddof=1))
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("invalid train-only outlier cutoff")
    result=[]
    for frame in (train,validation,test):
        keep=np.ones(len(frame),dtype=bool)
        if exclude_zero:keep &= frame.log_return.to_numpy()!=0
        if np.isfinite(cutoff):keep &= frame.log_return.abs().to_numpy()<=cutoff
        if exclude_rollover_adjacent:
            if "rollover_adjacent" not in frame:raise ValueError("rollover flag unavailable")
            keep &= ~frame.rollover_adjacent.to_numpy(dtype=bool)
        result.append(frame.loc[keep].copy())
    if min(map(len,result))<100:
        raise ValueError("sensitivity filter leaves insufficient observations")
    return tuple(result), {"exclude_zero":bool(exclude_zero),
                           "outlier_policy":outlier_policy,
                           "train_outlier_cutoff":None if not np.isfinite(cutoff) else float(cutoff),
                           "exclude_rollover_adjacent":bool(exclude_rollover_adjacent)}


def make_folds(reference: pd.DataFrame, train_years=3, block_months=6,expanding=False):
    first = reference.index.min().normalize()
    last = reference.index.max().normalize() + pd.Timedelta(1, unit="D")
    folds = []; cursor = first; number = 0
    while True:
        start=first if expanding else cursor
        train_end = cursor + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(months=block_months)
        test_end = val_end + pd.DateOffset(months=block_months)
        if test_end > last: break
        counts = [((reference.index >= a) & (reference.index < b)).sum()
                  for a, b in ((start, train_end), (train_end, val_end), (val_end, test_end))]
        if min(counts) >= 100:
            folds.append(Fold(number, start, train_end, val_end, test_end)); number += 1
        cursor += pd.DateOffset(months=block_months)
    if len(folds) < 3: raise ValueError("Need at least three complete folds")
    return folds


def slice_fold(frame, fold):
    return tuple(frame.loc[(frame.index >= a) & (frame.index < b)].copy()
                 for a, b in ((fold.train_start, fold.train_end),
                              (fold.train_end, fold.validation_end),
                              (fold.validation_end, fold.test_end)))


def sequence_starts(frame: pd.DataFrame, policy: str) -> np.ndarray:
    """Return causal reset flags for the configured sequence policy."""
    if policy not in POLICIES:
        raise ValueError(f"unknown sequence policy: {policy}")
    if frame.empty:
        return np.empty(0, dtype=bool)
    # A rejected/removed window is always a hard boundary.  Otherwise deleting
    # an observation would create a transition over an unknown return.
    discontinuity = np.zeros(len(frame), dtype=bool)
    if {"date", "session", "bar_number"} <= set(frame):
        same_context = (pd.to_datetime(frame["date"]).dt.normalize().to_numpy()[1:]
                        == pd.to_datetime(frame["date"]).dt.normalize().to_numpy()[:-1])
        same_context &= frame["session"].astype(str).to_numpy()[1:] == frame["session"].astype(str).to_numpy()[:-1]
        discontinuity[1:] = same_context & (np.diff(frame["bar_number"].to_numpy()) != 1)
    if policy == "continuous_carry":
        return discontinuity
    date = pd.to_datetime(frame["date"]).dt.normalize().to_numpy()
    starts = np.ones(len(frame), dtype=bool)
    if policy == "daily_sequence":
        starts[1:] = date[1:] != date[:-1]
    else:
        session=frame["session"].astype(str).to_numpy()
        starts[1:] = (date[1:] != date[:-1]) | (session[1:] != session[:-1])
    return starts | discontinuity


def sequence_lengths(frame: pd.DataFrame, policy: str):
    """Lengths passed to EM so forbidden boundaries never create transitions."""
    starts = sequence_starts(frame, policy)
    if len(starts) == 0:
        return None
    positions = np.r_[0, np.flatnonzero(starts[1:])+1]
    if len(positions) == 1:
        return None
    return np.diff(np.r_[positions, len(starts)]).astype(int)


def gaussian_variance(model):
    return np.diagonal(model.covars_, axis1=1, axis2=2)[:, 0]


def fit_one_hmm(x, family, k, iterations, seed, lengths=None,
                backend="cpu", device="cuda:0"):
    if family == "gaussian_hmm":
        model = GaussianSequenceHMM(k, n_iter=iterations, tol=.01, random_state=seed)
    else:
        model = StudentTHMM(k, df_mode="shared" if family == "student_t_shared" else "state",
                            n_iter=iterations, tol=.01, random_state=seed)
    if backend == "cpu":
        model.fit(x, lengths)
    elif backend == "cuda":
        from cuda_hmm import fit_cuda_hmm
        model = fit_cuda_hmm(model, x, lengths=lengths, device=device)
    else:
        raise ValueError(f"unknown backend: {backend}")
    scale2 = gaussian_variance(model) if family == "gaussian_hmm" else model.scale2_
    if not np.isfinite(scale2).all() or (scale2 < 1e-8).any():
        raise ValueError("degenerate variance/scale")
    history = list(model.monitor_.history)
    delta = float(history[-1]-history[-2]) if len(history)>1 else np.inf
    model.last_delta_ = delta
    model.delta_per_observation_ = delta/len(x)
    model.strict_converged_ = bool(model.monitor_.converged)
    model.practical_converged_ = bool(model.monitor_.converged or
                                      (delta >= -1e-8 and delta/len(x) < 1e-5))
    return model, float(model.score(x, lengths)), model.practical_converged_, int(model.monitor_.iter)


def fit_hmm_restarts(x, family, k, iterations, restarts, seed0, horizon, phase,
                     cache_dir: Path | None = None, lengths=None,
                     backend="cpu", device="cuda:0"):
    fitted = []; diagnostics = []
    for seed in range(seed0, seed0 + restarts):
        check_time_budget()
        cache = None if cache_dir is None else cache_dir/f"{phase}__h{horizon}__{family}__k{k}__seed{seed}.joblib"
        if cache is not None and cache.exists():
            saved=joblib.load(cache)
            if saved[0] is not None:fitted.append(saved[0])
            diagnostics.append(saved[1])
            emit_progress(f"CACHE {phase} | h={horizon}m | {family} | K={k} | seed={seed}",cache_dir)
            continue
        started=time.monotonic()
        emit_progress(f"START {phase} | h={horizon}m | {family} | K={k} | seed={seed}",cache_dir)
        try:
            model, ll, converged, n_used = fit_one_hmm(
                x, family, k, iterations, seed, lengths, backend, device)
            fitted_item=(converged,ll,model)
            diagnostic={"horizon_min": horizon, "phase": phase, "family": family,
                                "k": k, "seed": seed, "train_ll": ll,
                                "iterations": n_used, "converged": converged,
                                "strict_converged": model.strict_converged_,
                                "last_delta": model.last_delta_,
                                "delta_per_observation": model.delta_per_observation_,
                                "backend": backend, "device": device,
                                "sequences": 1 if lengths is None else len(lengths)}
            fitted.append(fitted_item);diagnostics.append(diagnostic)
            if cache is not None:atomic_joblib_dump((fitted_item,diagnostic),cache)
            emit_progress(f"DONE  {phase} | h={horizon}m | {family} | K={k} | seed={seed} | {time.monotonic()-started:.1f}s | converged={converged}",cache_dir)
        except Exception as exc:
            diagnostic={"horizon_min": horizon, "phase": phase, "family": family,
                        "k": k, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            diagnostics.append(diagnostic)
            if cache is not None:atomic_joblib_dump((None,diagnostic),cache)
            emit_progress(f"FAIL  {phase} | h={horizon}m | {family} | K={k} | seed={seed} | {type(exc).__name__}: {exc}",cache_dir)
    if not fitted: raise ValueError(f"all fits failed: h={horizon}, {family}, K={k}")
    converged = [item for item in fitted if item[0]]
    return max(converged or fitted, key=lambda item: item[1])[2], diagnostics


def causal_filter(model, x, previous_posterior=None, starts=None):
    values = np.asarray(x, dtype=float).reshape(-1)
    logb = np.vstack([emission_log_prob(model, value) for value in values])
    posterior = np.empty((len(values), model.n_components)); scores = np.empty(len(values))
    starts = np.zeros(len(values), dtype=bool) if starts is None else np.asarray(starts, dtype=bool)
    if len(starts) != len(values):
        raise ValueError("starts must have one flag per observation")
    for i in range(len(values)):
        prior = (model.startprob_ if starts[i] or (i == 0 and previous_posterior is None) else
                 previous_posterior @ model.transmat_ if i == 0 else
                 posterior[i-1] @ model.transmat_)
        joint = np.log(np.maximum(prior, 1e-300)) + logb[i]
        scores[i] = logsumexp(joint); posterior[i] = np.exp(joint - scores[i])
    return posterior, scores


def fit_iid(x):
    x = np.asarray(x).reshape(-1)
    gaussian = {"mean": float(x.mean()), "std": float(x.std(ddof=0))}
    df, loc, scale = stats.t.fit(x)
    # A Student-t density is valid for df>0. Short-horizon VN30F1M returns can
    # estimate df<=2 (undefined theoretical variance), which is itself useful
    # evidence of heavy tails and must not invalidate the density benchmark.
    if not all(np.isfinite([df, loc, scale])) or df <= 0 or scale <= 0:
        raise ValueError("invalid Student-t fit")
    return gaussian, {"df": float(df), "loc": float(loc), "scale": float(scale)}


def iid_scores(parameters, x, family):
    x = np.asarray(x).reshape(-1)
    return (stats.norm.logpdf(x, parameters["mean"], parameters["std"])
            if family == "gaussian" else
            stats.t.logpdf(x, parameters["df"], parameters["loc"], parameters["scale"]))


def validate_horizon(train, validation, horizon, iterations, restarts, cache_dir=None,
                     policy="continuous_carry", backend="cpu", device="cuda:0"):
    scaler = StandardScaler().fit(train[["log_return"]])
    xt = scaler.transform(train[["log_return"]]); xv = scaler.transform(validation[["log_return"]])
    jac = float(np.log(scaler.scale_[0])); gaussian, student = fit_iid(xt)
    rows = [{"horizon_min": horizon, "family": "gaussian", "k": 1,
             "validation_log_density": float(iid_scores(gaussian, xv, "gaussian").mean()-jac),
             "validation_min_soft_share": np.nan, "eligible_1pct": True},
            {"horizon_min": horizon, "family": "student_t", "k": 1,
             "validation_log_density": float(iid_scores(student, xv, "student_t").mean()-jac),
             "validation_min_soft_share": np.nan, "eligible_1pct": True}]
    for k in K_VALUES:
        emit_progress(f"START screen_selection | h={horizon}m | gmm | K={k}",cache_dir)
        started=time.monotonic()
        gmm = GaussianMixture(k, n_init=restarts, max_iter=iterations, random_state=42).fit(xt)
        gmm_share=float(gmm.predict_proba(xv).mean(0).min())
        rows.append({"horizon_min": horizon, "family": "gmm", "k": k,
                     "validation_log_density": float(gmm.score_samples(xv).mean()-jac),
                     "validation_min_soft_share": gmm_share,
                     "eligible_1pct": gmm_share>=.01})
        emit_progress(f"DONE  screen_selection | h={horizon}m | gmm | K={k} | {time.monotonic()-started:.1f}s",cache_dir)
    diagnostics = []
    train_lengths = sequence_lengths(train, policy)
    train_starts = sequence_starts(train, policy)
    validation_starts = sequence_starts(validation, policy)
    for k in K_VALUES:
        model, diag = fit_hmm_restarts(xt, "gaussian_hmm", k, iterations, restarts,
                                       42, horizon, "screen_selection", cache_dir,
                                       train_lengths, backend, device)
        diagnostics.extend(diag)
        prior = causal_filter(model, xt, starts=train_starts)[0][-1]
        prob, score = causal_filter(model, xv, prior, validation_starts)
        share = float(prob.mean(0).min())
        rows.append({"horizon_min": horizon, "family": "gaussian_hmm", "k": k,
                     "validation_log_density": float(score.mean()-jac),
                     "validation_min_soft_share": share, "eligible_1pct": share >= .01})
    candidate = pd.DataFrame(rows)
    student_score = float(candidate.loc[candidate.family.eq("student_t"), "validation_log_density"].iloc[0])
    candidate["validation_gain_vs_student"] = candidate.validation_log_density - student_score
    selected = {"gmm_k": int(candidate[candidate.family.eq("gmm") & candidate.eligible_1pct].sort_values(
        ["validation_log_density", "k"], ascending=[False, True]).iloc[0].k)}
    eligible = candidate[candidate.family.eq("gaussian_hmm") & candidate.eligible_1pct]
    if eligible.empty: raise ValueError(f"no eligible Gaussian HMM at {horizon}m")
    best = eligible.sort_values(["validation_gain_vs_student", "k"], ascending=[False, True]).iloc[0]
    selected["gaussian_hmm_k"] = int(best.k)
    selected["screen_gain"] = float(best.validation_gain_vs_student)
    return selected, candidate, diagnostics


def validate_student(train, validation, horizon, family, iterations, restarts,
                     student_score, cache_dir=None, policy="continuous_carry",
                     backend="cpu", device="cuda:0"):
    scaler = StandardScaler().fit(train[["log_return"]])
    xt = scaler.transform(train[["log_return"]]); xv = scaler.transform(validation[["log_return"]])
    jac = float(np.log(scaler.scale_[0])); rows = []; diagnostics = []
    train_lengths = sequence_lengths(train, policy)
    train_starts = sequence_starts(train, policy)
    validation_starts = sequence_starts(validation, policy)
    for k in K_VALUES:
        model, diag = fit_hmm_restarts(xt, family, k, iterations, restarts,
                                       62, horizon, "confirm_selection", cache_dir,
                                       train_lengths, backend, device)
        diagnostics.extend(diag)
        prior = causal_filter(model, xt, starts=train_starts)[0][-1]
        prob, score = causal_filter(model, xv, prior, validation_starts)
        share = float(prob.mean(0).min())
        logdensity = float(score.mean()-jac)
        rows.append({"horizon_min": horizon, "family": family, "k": k,
                     "validation_log_density": logdensity,
                     "validation_gain_vs_student": logdensity-student_score,
                     "validation_min_soft_share": share, "eligible_1pct": share >= .01})
    result = pd.DataFrame(rows); eligible = result[result.eligible_1pct]
    if eligible.empty: raise ValueError(f"no eligible {family} at {horizon}m")
    best = eligible.sort_values(["validation_gain_vs_student", "k"], ascending=[False, True]).iloc[0]
    return int(best.k), result, diagnostics


def distribution(model, family, scaler):
    means_z = model.means_[:, 0] if family == "gaussian_hmm" else model.means_
    scale_z = np.sqrt(gaussian_variance(model)) if family == "gaussian_hmm" else np.sqrt(model.scale2_)
    means = (means_z*scaler.scale_[0]+scaler.mean_[0])*100; scales = scale_z*scaler.scale_[0]*100
    if family == "gaussian_hmm":
        dfs = np.repeat(np.inf, model.n_components); stds = scales
    else:
        dfs = model.df_.copy(); stds = scales*np.sqrt(dfs/(dfs-2))
    return means, scales, stds, dfs


def duration_summary(test, hard, state, horizon, starts=None):
    starts=np.zeros(len(hard),dtype=bool) if starts is None else np.asarray(starts,dtype=bool)
    runs = []; start = 0
    for i in range(1, len(hard)+1):
        if i == len(hard) or starts[i] or hard[i] != hard[i-1]:
            if hard[i-1] == state:
                bars = i-start; trading = bars*horizon
                wall = (test.index[i-1] - test.window_start.iloc[start]).total_seconds()/60
                runs.append((bars, trading, wall))
            start = i
    if not runs: return {k: np.nan for k in ("mean_run_bars", "median_run_bars", "p90_run_bars",
                                               "mean_trading_minutes", "mean_calendar_minutes")}
    a = np.asarray(runs)
    return {"mean_run_bars": float(a[:,0].mean()), "median_run_bars": float(np.median(a[:,0])),
            "p90_run_bars": float(np.quantile(a[:,0], .9)),
            "mean_trading_minutes": float(a[:,1].mean()),
            "mean_calendar_minutes": float(a[:,2].mean())}


def prepare_test_context(train, validation, test, policy="continuous_carry"):
    development = pd.concat([train, validation]).sort_index()
    scaler = StandardScaler().fit(development[["log_return"]])
    xd = scaler.transform(development[["log_return"]]); xt = scaler.transform(test[["log_return"]])
    jac = float(np.log(scaler.scale_[0])); gaussian, student = fit_iid(xd)
    return (development, scaler, xd, xt, jac, gaussian, student,
            iid_scores(student, xt, "student_t")-jac,
            sequence_lengths(development, policy), sequence_starts(development, policy),
            sequence_starts(test, policy))


def fit_score_model(test, horizon, family, k, iterations, restarts, context,
                    cache_dir=None, backend="cpu", device="cuda:0"):
    (development, scaler, xd, xt, jac, gaussian, student, student_test,
     development_lengths, development_starts, test_starts) = context
    if family == "gaussian": scores = iid_scores(gaussian, xt, "gaussian")-jac; return scores, None, None, [], [], scaler
    if family == "student_t": return student_test, None, None, [], [], scaler
    if family == "gmm":
        model = GaussianMixture(k, n_init=restarts, max_iter=iterations, random_state=142).fit(xd)
        return model.score_samples(xt)-jac, None, model, [], [], scaler
    model, diagnostics = fit_hmm_restarts(xd, family, k, iterations, restarts,
                                           142, horizon, "refit", cache_dir,
                                           development_lengths, backend, device)
    prior = causal_filter(model, xd, starts=development_starts)[0][-1]
    probability, scores = causal_filter(model, xt, prior, test_starts)
    means, scales, stds, dfs = distribution(model, family, scaler); order = np.argsort(stds)
    display_probability = probability[:, order]; hard = display_probability.argmax(1)
    profiles = []
    for display_state, internal_state in enumerate(order):
        mask = hard == display_state
        profiles.append({"horizon_min": horizon, "family": family, "model_key": f"h{horizon}__{family}",
            "k": k, "internal_state": int(internal_state), "display_state": int(display_state),
            "model_mean_return_pct": float(means[internal_state]), "model_scale_pct": float(scales[internal_state]),
            "model_std_return_pct": float(stds[internal_state]),
            "model_df": np.nan if np.isinf(dfs[internal_state]) else float(dfs[internal_state]),
            "self_transition": float(model.transmat_[internal_state, internal_state]),
            "expected_duration_observations": float(1/max(1-model.transmat_[internal_state,internal_state],1e-12)),
            "expected_duration_trading_minutes": float(horizon/max(1-model.transmat_[internal_state,internal_state],1e-12)),
            "test_count": int(mask.sum()), "test_share": float(mask.mean()),
            "test_mean_return_pct": float(test.log_return.to_numpy()[mask].mean()*100) if mask.any() else np.nan,
            "test_std_return_pct": float(test.log_return.to_numpy()[mask].std(ddof=1)*100) if mask.sum()>1 else np.nan,
            "mean_confidence": float(display_probability[mask].max(1).mean()) if mask.any() else np.nan,
            **duration_summary(test, hard, display_state, horizon,test_starts)})
    state_output = pd.DataFrame({"state": hard, "confidence": display_probability.max(1),
                                 "sequence_start":test_starts}, index=test.index)
    return scores-jac, student_test, model, profiles, diagnostics, scaler, state_output


def process_fold(frames, fold, iterations, restarts, selection_iterations,
                 selection_restarts, top_n, total,checkpoint_root=None,
                 return_variant="raw", policy="continuous_carry",
                 backend="cpu", device="cuda:0", exclude_zero=False,
                 outlier_policy="keep", exclude_rollover_adjacent=False):
    with threadpool_limits(limits=1):
        emit_progress(f"FOLD {fold.fold+1}/{total} START")
        fold_cache=None if checkpoint_root is None else Path(checkpoint_root)/f"fold_{fold.fold:02d}_tasks"
        if fold_cache is not None:fold_cache.mkdir(parents=True,exist_ok=True)
        fit_cache=None if fold_cache is None else fold_cache/"fits"
        selected = {}; candidates = []; diagnostics = []; split = {}
        for h in HORIZONS:
            check_time_budget()
            emit_progress(f"SCREEN horizon {h}m START",fold_cache)
            train, val, test = slice_fold(frames[h], fold); split[h] = (train, val, test)
            (train,val,test),_=robust_filter_split(
                train,val,test,exclude_zero,outlier_policy,exclude_rollover_adjacent)
            split[h]=(train,val,test)
            if return_variant=="intraday_adjusted":
                train,val,test=intraday_adjust_split(train,val,test);split[h]=(train,val,test)
            task=None if fold_cache is None else fold_cache/f"screen_h{h}.joblib"
            if task is not None and task.exists():choice,table,diag=joblib.load(task)
            else:
                choice, table, diag = validate_horizon(
                    train, val, h, selection_iterations, selection_restarts,
                    fit_cache, policy, backend, device)
                if task is not None:atomic_joblib_dump((choice,table,diag),task)
            selected[h] = choice; candidates.append(table); diagnostics.extend(diag)
            emit_progress(f"SCREEN horizon {h}m DONE | Gaussian-HMM K={choice['gaussian_hmm_k']} | gain={choice['screen_gain']:.6f}",fold_cache)
        top = sorted(HORIZONS, key=lambda h: selected[h]["screen_gain"], reverse=True)[:top_n]
        for h in top:
            train, val, _ = split[h]
            student_score = float(pd.concat(candidates).query(
                "horizon_min == @h and family == 'student_t'").validation_log_density.iloc[0])
            for family in STUDENT_FAMILIES:
                check_time_budget()
                emit_progress(f"CONFIRM horizon {h}m | {family} START",fold_cache)
                task=None if fold_cache is None else fold_cache/f"confirm_h{h}_{family}.joblib"
                if task is not None and task.exists():k,table,diag=joblib.load(task)
                else:
                    k, table, diag = validate_student(
                        train, val, h, family, selection_iterations,
                        selection_restarts, student_score, fit_cache,
                        policy, backend, device)
                    if task is not None:atomic_joblib_dump((k,table,diag),task)
                selected[h][f"{family}_k"] = k; candidates.append(table); diagnostics.extend(diag)
                emit_progress(f"CONFIRM horizon {h}m | {family} DONE | K={k}",fold_cache)
        candidate = pd.concat(candidates, ignore_index=True)
        candidate["fold"] = fold.fold; candidate["screen_top2"] = candidate.horizon_min.isin(top)
        eligible_hmm = candidate[candidate.family.isin(ALL_HMM_FAMILIES) & candidate.eligible_1pct]
        champion_row = eligible_hmm.sort_values(
            ["validation_gain_vs_student", "k"], ascending=[False, True]).iloc[0]
        champion = (int(champion_row.horizon_min), str(champion_row.family), int(champion_row.k))
        predictions = []; profiles = []; models = {"fold_bounds": asdict(fold), "selection": selected,
                                                    "screen_top_horizons": top, "champion": champion}
        for h in HORIZONS:
            train, val, test = split[h]
            context = prepare_test_context(train, val, test, policy)
            families = [("student_t", 1), ("gaussian", 1), ("gmm", selected[h]["gmm_k"]),
                        ("gaussian_hmm", selected[h]["gaussian_hmm_k"])]
            if h in top:
                families += [(family, selected[h][f"{family}_k"]) for family in STUDENT_FAMILIES]
            student_scores = None
            for family, k in families:
                check_time_budget()
                emit_progress(f"REFIT horizon {h}m | {family} | K={k} START",fold_cache)
                task=None if fold_cache is None else fold_cache/f"refit_h{h}_{family}_k{k}.joblib"
                if task is not None and task.exists():output=joblib.load(task)
                else:
                    output = fit_score_model(test, h, family, k, iterations,
                                             restarts, context, fit_cache,
                                             backend, device)
                    if task is not None:atomic_joblib_dump(output,task)
                scores, baseline, model, state_profiles, diag, scaler = output[:6]
                state_output = output[6] if len(output) == 7 else None
                if family == "student_t": student_scores = scores
                if baseline is not None: student_scores = baseline
                if student_scores is None:
                    # Student is processed before every HMM/GMM in the family list.
                    raise RuntimeError("student baseline unavailable")
                part = test[["date", "session", "window_start", "time_bucket",
                             "bar_number", "rollover_adjacent", "log_return"]].copy()
                part["fold"] = fold.fold; part["horizon_min"] = h; part["family"] = family
                part["model_key"] = f"h{h}__{family}"; part["log_density"] = scores
                part["student_log_density"] = student_scores
                part["gain_vs_student"] = scores-student_scores
                part["state"] = state_output.state if state_output is not None else np.nan
                part["confidence"] = state_output.confidence if state_output is not None else np.nan
                part["sequence_start"] = state_output.sequence_start if state_output is not None else False
                predictions.append(part); diagnostics.extend(diag)
                for row in state_profiles: row["fold"] = fold.fold
                profiles.extend(state_profiles)
                models[f"h{h}__{family}"] = model if model is not None else {"parameters": "iid"}
                if family in ALL_HMM_FAMILIES: models[f"h{h}__{family}__scaler"] = scaler
                emit_progress(f"REFIT horizon {h}m | {family} | K={k} DONE",fold_cache)
        fold_row = {"fold": fold.fold, "train_start": str(fold.train_start.date()),
                    "train_end_exclusive": str(fold.train_end.date()),
                    "validation_end_exclusive": str(fold.validation_end.date()),
                    "test_end_exclusive": str(fold.test_end.date()),
                    "screen_top_horizons": ",".join(map(str, top)),
                    "champion_horizon": champion[0], "champion_family": champion[1], "champion_k": champion[2]}
        emit_progress(f"FOLD {fold.fold+1}/{total} DONE",fold_cache)
        return pd.concat(predictions), candidate, pd.DataFrame(profiles), diagnostics, fold_row, models


def alignment_cost(old, new):
    ref = max((old.model_std_return_pct+new.model_std_return_pct)/2, .02)
    df_old = 200 if pd.isna(old.model_df) else old.model_df; df_new = 200 if pd.isna(new.model_df) else new.model_df
    return float(abs(old.model_mean_return_pct-new.model_mean_return_pct)/ref
                 + .6*abs(np.log(max(old.model_std_return_pct,1e-6)/max(new.model_std_return_pct,1e-6)))
                 + .3*abs(old.self_transition-new.self_transition)
                 + .2*abs(old.test_share-new.test_share)+.15*abs(np.log(df_old/df_new)))


def align_states(states, threshold=1.5):
    states = states.copy(); states["stable_state_id"] = ""; states["alignment_cost"] = np.nan
    states["alignment_status"] = ""; events = []
    for key, group in states.groupby("model_key", sort=False):
        previous = None; counter = 0
        for fold in sorted(group.fold.unique()):
            current_idx = states.index[(states.model_key == key)&(states.fold == fold)].tolist()
            if previous is None:
                for idx in current_idx:
                    counter += 1; sid = f"{key}::R{counter:03d}"
                    states.loc[idx,["stable_state_id","alignment_status"]] = [sid,"new_reference"]
                    events.append({"model_key":key,"fold":fold,"event":"new","stable_state_id":sid,"cost":np.nan})
            else:
                old=states.loc[previous]; new=states.loc[current_idx]
                cost=np.array([[alignment_cost(o,n) for _,n in new.iterrows()] for _,o in old.iterrows()])
                rr,cc=linear_sum_assignment(cost); matched_new=set(); matched_old=set()
                for r,c in zip(rr,cc):
                    if cost[r,c] <= threshold:
                        oi,ni=previous[r],current_idx[c]; sid=states.loc[oi,"stable_state_id"]
                        states.loc[ni,["stable_state_id","alignment_cost","alignment_status"]]=[sid,cost[r,c],"matched"]
                        matched_new.add(ni);matched_old.add(oi)
                        events.append({"model_key":key,"fold":fold,"event":"matched","stable_state_id":sid,"cost":cost[r,c]})
                for idx in current_idx:
                    if idx not in matched_new:
                        counter+=1;sid=f"{key}::R{counter:03d}"
                        states.loc[idx,["stable_state_id","alignment_status"]]=[sid,"new_unmatched"]
                        events.append({"model_key":key,"fold":fold,"event":"new","stable_state_id":sid,"cost":np.nan})
                for idx in previous:
                    if idx not in matched_old:
                        events.append({"model_key":key,"fold":fold,"event":"retired",
                                       "stable_state_id":states.loc[idx,"stable_state_id"],"cost":np.nan})
            previous=current_idx
    return states,pd.DataFrame(events)


def bootstrap_daily(values, block_lengths=(1,5,10,20), iterations=2000, seed=404):
    x=np.asarray(values,float);x=x[np.isfinite(x)];n=len(x);rng=np.random.default_rng(seed);rows=[]
    for block in block_lengths:
        need=int(np.ceil(n/block));starts=rng.integers(0,n,size=(iterations,need))
        idx=(starts[:,:,None]+np.arange(block)[None,None,:])%n
        draws=x[idx.reshape(iterations,-1)[:,:n]].mean(1)
        rows.append({"days":n,"block_days":block,"iterations":iterations,"mean_difference":float(x.mean()),
                     "ci_95_low":float(np.quantile(draws,.025)),"ci_95_high":float(np.quantile(draws,.975)),
                     "probability_mean_positive":float((draws>0).mean())})
    return rows


def bootstrap_results(predictions):
    daily=(predictions.groupby(["fold","horizon_min","family","date"],as_index=False)
           .gain_vs_student.mean()); rows=[]
    for (h,family),group in daily.groupby(["horizon_min","family"]):
        for row in bootstrap_daily(group.gain_vs_student):
            rows.append({"comparison":"gain_vs_student","left":f"h{h}__{family}","right":"zero",**row})
    gaussian=daily[daily.family.eq("gaussian_hmm")]
    means=gaussian.groupby("horizon_min").gain_vs_student.mean().sort_values(ascending=False)
    winner=int(means.index[0]);runner=int(means.index[1])
    wide=gaussian.pivot_table(index=["fold","date"],columns="horizon_min",values="gain_vs_student").dropna()
    for other in HORIZONS:
        if other==winner:continue
        for row in bootstrap_daily(wide[winner]-wide[other]):
            rows.append({"comparison":"horizon_gain_difference","left":f"h{winner}__gaussian_hmm",
                         "right":f"h{other}__gaussian_hmm",**row})
    # Direct paired family comparisons answer whether heavy tails improve on
    # Gaussian HMM, rather than comparing both only against a shared baseline.
    hmm_daily=(predictions[predictions.family.isin(ALL_HMM_FAMILIES)]
               .groupby(["fold","horizon_min","family","date"],as_index=False)
               .log_density.mean())
    for h,group in hmm_daily.groupby("horizon_min"):
        wide_family=group.pivot_table(index=["fold","date"],columns="family",
                                      values="log_density").dropna(axis=1,how="all")
        for left,right in combinations(sorted(wide_family.columns),2):
            paired=wide_family[[left,right]].dropna()
            if paired.empty:continue
            for row in bootstrap_daily(paired[left]-paired[right]):
                rows.append({"comparison":"hmm_family_difference",
                             "left":f"h{h}__{left}","right":f"h{h}__{right}",**row})
    return pd.DataFrame(rows),winner,runner


def choose_hmm_family(metrics, bootstraps, horizon):
    subset=metrics[(metrics.horizon_min==horizon)&metrics.family.isin(ALL_HMM_FAMILIES)]
    summary=(subset.groupby("family").agg(mean_log_density=("log_density","mean"),
                                           folds=("fold","nunique")).sort_values("mean_log_density",ascending=False))
    best=str(summary.index[0]);reference="gaussian_hmm";low=high=np.nan
    if best!=reference:
        pair=bootstraps[(bootstraps.comparison=="hmm_family_difference")
                        & (bootstraps.block_days==20)]
        direct=pair[(pair.left==f"h{horizon}__{best}")&(pair.right==f"h{horizon}__{reference}")]
        reverse=pair[(pair.left==f"h{horizon}__{reference}")&(pair.right==f"h{horizon}__{best}")]
        if len(direct):low,high=float(direct.iloc[0].ci_95_low),float(direct.iloc[0].ci_95_high)
        elif len(reverse):low,high=-float(reverse.iloc[0].ci_95_high),-float(reverse.iloc[0].ci_95_low)
    conclusive=bool(best==reference or (np.isfinite(low) and low>0))
    preferred=best if conclusive else reference
    return {"best_mean_hmm_family":best,"preferred_hmm_family":preferred,
            "family_gain_vs_gaussian_ci20":[None if not np.isfinite(low) else low,
                                             None if not np.isfinite(high) else high],
            "heavy_tail_replacement_conclusive":bool(best!=reference and conclusive),
            "family_fold_counts":{str(k):int(v) for k,v in summary.folds.items()}}


def occupancy_sensitivity(candidates):
    rows=[]
    hmm=candidates[candidates.family.isin(ALL_HMM_FAMILIES)]
    for (fold,h,family),g in hmm.groupby(["fold","horizon_min","family"]):
        for threshold in (.005,.01,.02,.05):
            eligible=g[g.validation_min_soft_share>=threshold]
            best=None if eligible.empty else eligible.sort_values(
                ["validation_gain_vs_student","k"],ascending=[False,True]).iloc[0]
            rows.append({"fold":fold,"horizon_min":h,"family":family,"threshold":threshold,
                         "selected_k":np.nan if best is None else int(best.k),
                         "eligible_candidates":len(eligible)})
    result=pd.DataFrame(rows);ref=result[result.threshold.eq(.01)].set_index(
        ["fold","horizon_min","family"]).selected_k
    result["same_k_as_1pct"]=[bool(r.selected_k==ref.get((r.fold,r.horizon_min,r.family),np.nan))
                              for r in result.itertuples()]
    return result


def refit_convergence_audit(models):
    rows=[]
    for fold,item in models.items():
        for key,model in item.items():
            if not (isinstance(key,str) and key.startswith("h") and "__" in key
                    and not key.endswith("__scaler") and hasattr(model,"monitor_")):
                continue
            horizon=int(key.split("__",1)[0][1:]);family=key.split("__",1)[1]
            history=list(model.monitor_.history)
            delta=float(history[-1]-history[-2]) if len(history)>1 else np.inf
            n=int(sum(model.lengths_));per_observation=delta/n
            rows.append({"fold":fold,"horizon_min":horizon,"family":family,
                         "k":model.n_components,"iterations":len(history),
                         "strict_converged":bool(model.monitor_.converged),
                         "last_delta":delta,"delta_per_observation":per_observation,
                         "practical_converged_1e_5":bool(model.monitor_.converged or
                                                         (delta>=-1e-8 and per_observation<1e-5))})
    return pd.DataFrame(rows)


def state_dynamics(predictions, confidence_threshold=.60):
    rows=[]
    hmm=predictions[predictions.family.isin(ALL_HMM_FAMILIES)&predictions.state.notna()]
    for (fold,h,family),g in hmm.groupby(["fold","horizon_min","family"]):
        g=g.sort_index();state=g.state.to_numpy();confidence=g.confidence.to_numpy();n=len(g)
        reset=g.sequence_start.fillna(False).to_numpy(dtype=bool)
        changes=(state[1:]!=state[:-1]) & ~reset[1:]
        run_break=(state[1:]!=state[:-1]) | reset[1:]
        starts=np.r_[0,np.flatnonzero(run_break)+1];ends=np.r_[starts[1:],n]
        lengths=ends-starts;delays=[]
        for start,end in zip(starts[1:],ends[1:]):
            confirmed=np.flatnonzero(confidence[start:end]>=confidence_threshold)
            if len(confirmed):delays.append(int(confirmed[0]))
        rows.append({"fold":fold,"horizon_min":h,"family":family,"observations":n,
                     "state_changes":int(changes.sum()),
                     "sequence_resets":int(reset.sum()),
                     "changes_per_100_bars":100*changes.sum()/n,
                     "one_bar_run_rate":float((lengths==1).mean()),
                     "mean_run_bars":float(lengths.mean()),
                     "mean_confirmation_delay_bars":float(np.mean(delays)) if delays else np.nan,
                     "mean_confirmation_delay_trading_minutes":float(np.mean(delays)*h) if delays else np.nan,
                     "confidence_threshold":confidence_threshold})
    return pd.DataFrame(rows)


def one_missing_sensitivity(raw, rejected, winner):
    close=raw.CLOSE_PX;rows=[]
    subset=rejected[rejected.horizon_min.eq(winner)]
    for row in subset.itertuples():
        series=close.reindex(pd.date_range(row.start,row.end,freq="min"))
        missing=int(series.isna().sum());endpoints=bool(pd.notna(series.iloc[0]) and pd.notna(series.iloc[-1]))
        category=("recoverable_one_internal_missing" if missing==1 and endpoints else
                  "endpoint_missing" if not endpoints else "more_than_one_internal_missing")
        rows.append({"horizon_min":winner,"start":row.start,"end":row.end,
                     "missing_minutes":missing,"endpoints_present":endpoints,"category":category})
    return pd.DataFrame(rows)


def write_report(folder,audit,horizon_audit,folds,candidates,metrics,bootstraps,states,alignments,
                 intraday,dynamics,sensitivity,missing_sensitivity,decision,fit_diag,
                 refit_audit,manifest,iterations,restarts,policy):
    os.environ.setdefault("MPLCONFIGDIR",str(HERE/".mplconfig"));import matplotlib
    matplotlib.use("Agg");import matplotlib.pyplot as plt
    gaussian=metrics[metrics.family.eq("gaussian_hmm")].groupby("horizon_min").gain_vs_student.mean().sort_index()
    fig,ax=plt.subplots(figsize=(9,5));gaussian.plot.bar(ax=ax,color="#2563eb")
    ax.axhline(0,color="#333",lw=.8);ax.set(title="Gaussian HMM gain over Student-t i.i.d.",ylabel="Test log-density gain",xlabel="Return horizon (minutes)")
    ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(folder/"horizon_comparison.png",dpi=170);plt.close(fig)
    fold_gaussian=metrics[metrics.family.eq("gaussian_hmm")].pivot(index="fold",columns="horizon_min",values="gain_vs_student")
    fig,ax=plt.subplots(figsize=(10,5));fold_gaussian.plot(marker="o",ax=ax)
    ax.axhline(0,color="#333",lw=.8);ax.set(title="Out-of-sample gain by fold",ylabel="Gain vs Student-t i.i.d.")
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/"fold_horizon_comparison.png",dpi=170);plt.close(fig)
    def tbl(df):return df.to_html(index=False,border=0,float_format=lambda x:f"{x:.6f}")
    summary=metrics.groupby(["horizon_min","family"]).agg(mean_log_density=("log_density","mean"),
        mean_gain_vs_student=("gain_vs_student","mean"),folds=("fold","nunique")).reset_index().sort_values("mean_gain_vs_student",ascending=False)
    selection=pd.DataFrame(folds)[["fold","screen_top_horizons","champion_horizon","champion_family","champion_k"]]
    sens=sensitivity.groupby("threshold").agg(comparisons=("same_k_as_1pct","size"),
        same_k_rate=("same_k_as_1pct","mean"),no_candidate=("selected_k",lambda x:int(x.isna().sum()))).reset_index()
    diag=pd.DataFrame(fit_diag);success=diag.train_ll.notna();conv=diag[success].groupby("family").agg(
        fits=("train_ll","count"),convergence_rate=("converged","mean"),mean_iterations=("iterations","mean")).reset_index()
    refit_summary=refit_audit.groupby(["horizon_min","family"]).agg(
        models=("fold","size"),strict_rate=("strict_converged","mean"),
        practical_rate=("practical_converged_1e_5","mean"),
        median_delta_per_observation=("delta_per_observation","median")).reset_index()
    sample=horizon_audit[["horizon_min","accepted_returns","rejected_windows","zero_rate","std","excess_kurtosis"]]
    page=f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>ARSH v0.5</title>
<style>@page{{size:A4;margin:14mm}}body{{font:14px/1.5 Arial;max-width:1250px;margin:30px auto;color:#172033}}h1,h2{{color:#123b57}}.lead{{background:#e0f2fe;padding:16px;border-left:4px solid #0284c7}}.warn{{background:#fff7ed;padding:14px;border-left:4px solid #ea580c}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:10px;margin:12px 0}}th{{background:#164e6b;color:#fff}}th,td{{border:1px solid #ccd6dd;padding:4px;text-align:right}}td:first-child{{text-align:left}}code{{font-size:11px}}</style>
<h1>ARSH v0.5</h1><p><strong>Adaptive Regime-Switching — Hieu</strong></p>
<div class="lead"><b>Kết luận.</b> Gaussian HMM có gain trung bình cao nhất tại khung <b>{decision['best_gaussian_horizon']} phút</b>. So với khung đứng thứ hai {decision['runner_up_horizon']} phút, bootstrap block 20 ngày có CI [{decision['winner_vs_runner_ci20'][0]:.6f}, {decision['winner_vs_runner_ci20'][1]:.6f}]. <b>{html.escape(decision['decision_vi'])}</b></div>
<p>Họ HMM ưu tiên: <b>{html.escape(decision['preferred_hmm_family'])}</b>; họ có mean cao nhất: {html.escape(decision['best_mean_hmm_family'])}. Student-t chỉ thay Gaussian khi paired block bootstrap kết luận rõ.</p>
<h2>Thiết kế</h2><p>Tạo riêng lợi suất log cho các horizon {html.escape(', '.join(map(str, sorted(horizon_audit.horizon_min.unique()))))} phút, bar không chồng lấn trong 09:00–11:30 và 13:00–14:30. Không tạo return xuyên trưa, ATC hay qua đêm. Policy <code>{html.escape(policy)}</code> được dùng nhất quán trong EM và causal filtering; cửa sổ thiếu luôn tạo ranh giới cứng. Mọi horizon dùng cùng {len(folds)} mốc walk-forward; validation chọn horizon/K/model, test chỉ đánh giá.</p>
<p>Nguồn có {audit['rows']:,} dòng phút; SHA-256 <code>{audit['source_sha256']}</code>. SHA-256 model <code>{manifest['selected_fold_models.joblib']}</code>.</p>
<h2>Audit theo horizon</h2>{tbl(sample)}<img src="horizon_comparison.png"><img src="fold_horizon_comparison.png">
<h2>Kết quả test</h2>{tbl(summary)}<h2>Lựa chọn trong từng fold</h2>{tbl(selection)}
<h2>Bootstrap</h2>{tbl(bootstraps[bootstraps.block_days.eq(20)])}
<h2>Thời lượng state</h2><p>Báo cáo đồng thời số bar, phút giao dịch và phút lịch. Có {len(states)} state profile; {(states.test_count==0).sum()} state không có hard assignment. Alignment: {(alignments.event=='matched').sum()} matched, {(alignments.event=='new').sum()} new, {(alignments.event=='retired').sum()} retired.</p>
{tbl(states.groupby(['horizon_min','family']).agg(mean_run_bars=('mean_run_bars','mean'),mean_trading_minutes=('mean_trading_minutes','mean'),mean_calendar_minutes=('mean_calendar_minutes','mean'),mean_confidence=('mean_confidence','mean')).reset_index())}
<h2>Theo thời điểm trong phiên</h2>{tbl(intraday)}<h2>Độ nhạy occupancy</h2>{tbl(sens)}
<h2>Flicker và độ trễ xác nhận</h2><p>Độ trễ là số bar từ lúc hard state đổi đến lần đầu confidence đạt 60%; đây là proxy vận hành, không phải độ trễ so với ground truth chưa quan sát được.</p>{tbl(dynamics.groupby(['horizon_min','family']).agg(changes_per_100_bars=('changes_per_100_bars','mean'),one_bar_run_rate=('one_bar_run_rate','mean'),mean_run_bars=('mean_run_bars','mean'),confirmation_delay_minutes=('mean_confirmation_delay_trading_minutes','mean')).reset_index())}
<h2>Độ nhạy một phút thiếu</h2>{tbl(missing_sensitivity.groupby('category').size().reset_index(name='windows')) if len(missing_sensitivity) else '<p>Không có cửa sổ bị loại.</p>'}
<h2>Chẩn đoán fit</h2><p>{int(success.sum())} fit thành công; refit dùng tối đa {iterations} vòng và {restarts} seed. `strict_rate` dùng ngưỡng likelihood tổng 0,01 của v0.3; `practical_rate` dùng thay đổi nhỏ hơn 1e-5 trên mỗi quan sát để so sánh công bằng giữa các horizon có cỡ mẫu khác nhau.</p>{tbl(conv)}{tbl(refit_summary)}
<h2>Giới hạn</h2><ul><li>Test cũ là research benchmark, chưa phải final untouched holdout.</li><li>Rollover và timestamp chỉ được audit từ chuỗi tổng hợp; chưa có mã hợp đồng gốc/tick data để xác minh tuyệt đối.</li><li>V0.5 chọn horizon đơn biến; chưa ghép đa khung.</li><li>Không có online parameter learning, HireVAE, tín hiệu hoặc paper trading.</li></ul>
<div class="warn">Khung {decision['best_gaussian_horizon']} phút là horizon ưu tiên theo bằng chứng hiện tại. Chưa khóa cấu hình cho v1 nếu K thường chạm biên tìm kiếm hoặc convergence/state chưa đạt cổng chất lượng.</div></html>'''
    (folder/"report.html").write_text(page,encoding="utf8")


def main():
    global HORIZONS, K_VALUES, RUN_DEADLINE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DEFAULT_DATA);parser.add_argument("--output",type=Path,default=HERE/"outputs")
    parser.add_argument("--iterations",type=int,default=300);parser.add_argument("--restarts",type=int,default=5)
    parser.add_argument("--selection-iterations",type=int,default=100)
    parser.add_argument("--selection-restarts",type=int,default=3)
    parser.add_argument("--jobs",type=int,default=1);parser.add_argument("--top-horizons",type=int,default=3)
    parser.add_argument("--max-folds",type=int)
    parser.add_argument("--horizons",type=int,nargs="+",default=list(HORIZONS))
    parser.add_argument("--k-values",type=int,nargs="+",default=list(K_VALUES))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--time-budget-minutes",type=float)
    parser.add_argument("--return-variant",choices=("raw","intraday_adjusted"),default="raw")
    parser.add_argument("--overlapping",action="store_true")
    parser.add_argument("--train-years",type=int,default=3)
    parser.add_argument("--expanding-window",action="store_true")
    parser.add_argument("--policy", choices=POLICIES, default="continuous_carry")
    parser.add_argument("--backend", choices=BACKENDS, default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, nargs="+",
                        help="Original zero-based fold IDs for an independent shard")
    parser.add_argument("--exclude-zero", action="store_true",
                        help="Sensitivity only: remove exact-zero returns")
    parser.add_argument("--outlier-policy", choices=("keep","exclude_train_5sigma"),
                        default="keep")
    parser.add_argument("--exclude-rollover-adjacent", action="store_true",
                        help="Sensitivity only: remove days following likely expiry Thursday")
    parser.add_argument("--allow-one-internal-missing", action="store_true",
                        help="Sensitivity only: admit windows with both endpoints and one missing interior minute")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Never fit; require every selected fold checkpoint (CPU merge host)")
    args=parser.parse_args()
    HORIZONS=tuple(dict.fromkeys(args.horizons));K_VALUES=tuple(dict.fromkeys(args.k_values))
    if min(HORIZONS)<1 or min(K_VALUES)<2:parser.error("invalid horizon or K")
    if args.jobs != 1:
        parser.error("v0.5 checkpoint runner currently requires --jobs 1")
    if args.backend == "cuda" and not args.finalize_only:
        try:
            from cuda_hmm import cuda_available
            if not cuda_available(args.device):
                parser.error(f"CUDA backend requested but unavailable on {args.device}")
        except ImportError as exc:
            parser.error(f"CUDA backend requires PyTorch with CUDA: {exc}")
    if not args.data.is_file():
        parser.error(f"data file not found: {args.data}. Copy ohlc_export.csv into data/ or pass --data PATH")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error("output folder is not empty; use --resume or choose another folder")
    args.output.mkdir(parents=True,exist_ok=True)
    emit_progress("LOAD source data START")
    raw,audit=load_minutes(args.data);emit_progress(f"LOAD source data DONE | rows={len(raw):,}")
    emit_progress("AUDIT timestamp/rollover START")
    source_daily,source_summary=source_structure_audit(raw)
    emit_progress("AUDIT timestamp/rollover DONE")
    emit_progress(f"BUILD returns START | horizons={list(HORIZONS)}")
    frames,horizon_audit,rejected,gaps=build_returns(
        raw,HORIZONS,overlapping=args.overlapping,
        allow_one_internal_missing=args.allow_one_internal_missing)
    emit_progress("BUILD returns DONE | "+", ".join(f"{h}m={len(frames[h]):,}" for h in HORIZONS))
    reference_horizon=60 if 60 in frames else max(frames)
    folds=make_folds(frames[reference_horizon],train_years=args.train_years,
                     expanding=args.expanding_window)
    if args.folds is not None:
        requested=set(args.folds);available={fold.fold for fold in folds}
        unknown=sorted(requested-available)
        if unknown:parser.error(f"unknown fold IDs: {unknown}; available={sorted(available)}")
        folds=[fold for fold in folds if fold.fold in requested]
    folds=folds[:args.max_folds] if args.max_folds else folds
    if not folds:parser.error("no folds selected")
    checkpoint_dir=args.output/"_checkpoints";checkpoint_dir.mkdir(exist_ok=True)
    worker=partial(process_fold,frames,iterations=args.iterations,restarts=args.restarts,
                   selection_iterations=args.selection_iterations,
                   selection_restarts=args.selection_restarts,
                   top_n=args.top_horizons,total=len(folds),checkpoint_root=checkpoint_dir,
                   return_variant=args.return_variant,policy=args.policy,
                   backend=args.backend,device=args.device,
                   exclude_zero=args.exclude_zero,outlier_policy=args.outlier_policy,
                   exclude_rollover_adjacent=args.exclude_rollover_adjacent)
    run_identity={"version":"0.5","source_sha256":audit["source_sha256"],
                  "code_sha256":sha256(Path(__file__)),
                  "horizons":list(HORIZONS),"k_values":list(K_VALUES),
                  "iterations":args.iterations,"restarts":args.restarts,
                  "selection_iterations":args.selection_iterations,
                  "selection_restarts":args.selection_restarts,"top_horizons":args.top_horizons,
                  "return_variant":args.return_variant,"overlapping":args.overlapping,
                  "train_years":args.train_years,
                  "expanding_window":args.expanding_window,
                  "policy":args.policy,"backend":args.backend,"device":args.device,
                  "exclude_zero":args.exclude_zero,"outlier_policy":args.outlier_policy,
                  "exclude_rollover_adjacent":args.exclude_rollover_adjacent,
                  "allow_one_internal_missing":args.allow_one_internal_missing,
                  "selected_fold_ids":[fold.fold for fold in folds],
                  "folds":[asdict(f) for f in folds]}
    identity_path=checkpoint_dir/"run_identity.joblib"
    if identity_path.exists():
        if joblib.load(identity_path)!=run_identity:
            parser.error("checkpoint configuration/data differs; use a new output folder")
    else:atomic_joblib_dump(run_identity,identity_path)
    horizon_audit.to_csv(args.output/"horizon_data_audit.csv",index=False,encoding="utf-8-sig")
    rejected.to_csv(args.output/"rejected_windows.csv",index=False,encoding="utf-8-sig")
    gaps.to_csv(args.output/"excluded_gap_returns.csv",index=False,encoding="utf-8-sig")
    source_daily.to_csv(args.output/"source_structure_daily.csv",index=False,encoding="utf-8-sig")
    (args.output/"data_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf8")
    (args.output/"source_structure_audit.json").write_text(
        json.dumps(source_summary,indent=2,ensure_ascii=False),encoding="utf8")
    deadline=None if args.time_budget_minutes is None else time.monotonic()+60*args.time_budget_minutes
    RUN_DEADLINE=deadline
    results=[]
    with threadpool_limits(limits=1):
        for fold in folds:
            checkpoint=checkpoint_dir/f"fold_{fold.fold:02d}.joblib"
            if checkpoint.exists():
                print(f"Resume fold {fold.fold+1}/{len(folds)}",flush=True);result=joblib.load(checkpoint)
            else:
                if args.finalize_only:
                    raise RuntimeError(f"finalize-only requires checkpoint: {checkpoint}")
                if deadline is not None and time.monotonic()>=deadline:break
                try:result=worker(fold)
                except TimeBudgetReached:break
                atomic_joblib_dump(result,checkpoint)
            results.append(result)
            progress={"completed_folds":len(results),"total_folds":len(folds),
                      "complete":len(results)==len(folds),"last_update":pd.Timestamp.now().isoformat()}
            (args.output/"progress.json").write_text(json.dumps(progress,indent=2),encoding="utf8")
    if len(results)<len(folds):
        print(f"Safe stop: {len(results)}/{len(folds)} folds complete. Run again with --resume.",flush=True)
        return
    predictions=pd.concat([x[0] for x in results]).sort_index();candidates=pd.concat([x[1] for x in results],ignore_index=True)
    states=pd.concat([x[2] for x in results],ignore_index=True);fit_diag=sum([x[3] for x in results],[])
    fold_rows=[x[4] for x in results];models={fold.fold:x[5] for fold,x in zip(folds,results)}
    states,alignments=align_states(states)
    for row in states.itertuples():
        mask=(predictions.fold.eq(row.fold)&predictions.model_key.eq(row.model_key)&predictions.state.eq(row.display_state))
        predictions.loc[mask,"stable_state_id"]=row.stable_state_id
    metrics=predictions.groupby(["fold","horizon_min","family","model_key"],as_index=False).agg(
        log_density=("log_density","mean"),gain_vs_student=("gain_vs_student","mean"),observations=("log_density","size"))
    bootstraps,winner,runner=bootstrap_results(predictions)
    pair=bootstraps[(bootstraps.left==f"h{winner}__gaussian_hmm")&(bootstraps.right==f"h{runner}__gaussian_hmm")&bootstraps.block_days.eq(20)]
    if pair.empty:
        # winner was compared against every other; runner is therefore present.
        raise RuntimeError("winner/runner bootstrap missing")
    pair=pair.iloc[0];conclusive=bool(pair.ci_95_low>0)
    refit_audit=refit_convergence_audit(models)
    winner_refit=refit_audit[(refit_audit.horizon_min==winner)&(refit_audit.family=="gaussian_hmm")]
    practical_rate=float(winner_refit.practical_converged_1e_5.mean())
    gaussian_candidates=candidates[(candidates.family=="gaussian_hmm")&candidates.eligible_1pct]
    chosen=(gaussian_candidates.sort_values(
        ["fold","horizon_min","validation_gain_vs_student","k"],ascending=[True,True,False,True])
        .groupby(["fold","horizon_min"],as_index=False).first())
    boundary_rate=float((chosen[chosen.horizon_min==winner].k==max(K_VALUES)).mean())
    ready_for_v1=bool(conclusive and practical_rate>=.8 and boundary_rate<=.5)
    decision={"best_gaussian_horizon":winner,"runner_up_horizon":runner,
              "winner_vs_runner_mean":float(pair.mean_difference),
              "winner_vs_runner_ci20":[float(pair.ci_95_low),float(pair.ci_95_high)],
              "bootstrap_horizon_advantage_conclusive":conclusive,
              "winner_practical_convergence_rate":practical_rate,
              "winner_k_upper_boundary_rate":boundary_rate,
              "preferred_research_horizon":winner,
              "lock_configuration_for_v1":ready_for_v1,
              "decision_vi":(f"Ưu tiên khung {winner} phút, nhưng chưa khóa cấu hình cho v1 vì cổng K/hội tụ chưa đạt."
                             if conclusive and not ready_for_v1 else
                             f"Khung {winner} phút đủ điều kiện làm cấu hình kế tiếp."
                             if ready_for_v1 else
                             "Chưa đủ bằng chứng chọn một horizon; giữ hai ứng viên đầu cho v0.5.")}
    decision.update(choose_hmm_family(metrics,bootstraps,winner))
    sensitivity=occupancy_sensitivity(candidates)
    dynamics=state_dynamics(predictions)
    missing_sensitivity=one_missing_sensitivity(raw,rejected,winner)
    intraday=(predictions[predictions.family.isin(ALL_HMM_FAMILIES)].groupby(
        ["horizon_min","family","time_bucket"],as_index=False).agg(
        observations=("log_density","size"),mean_gain_vs_student=("gain_vs_student","mean"),
        mean_confidence=("confidence","mean")))
    folder=args.output
    predictions.to_csv(folder/"test_predictions_long.csv",encoding="utf-8-sig")
    candidates.to_csv(folder/"validation_candidates.csv",index=False,encoding="utf-8-sig")
    metrics.to_csv(folder/"walk_forward_metrics.csv",index=False,encoding="utf-8-sig")
    horizon_audit.to_csv(folder/"horizon_data_audit.csv",index=False,encoding="utf-8-sig")
    rejected.to_csv(folder/"rejected_windows.csv",index=False,encoding="utf-8-sig")
    gaps.to_csv(folder/"excluded_gap_returns.csv",index=False,encoding="utf-8-sig")
    states.to_csv(folder/"state_profiles_aligned.csv",index=False,encoding="utf-8-sig")
    alignments.to_csv(folder/"state_alignment_events.csv",index=False,encoding="utf-8-sig")
    bootstraps.to_csv(folder/"bootstrap_comparisons.csv",index=False,encoding="utf-8-sig")
    sensitivity.to_csv(folder/"occupancy_sensitivity.csv",index=False,encoding="utf-8-sig")
    intraday.to_csv(folder/"intraday_analysis.csv",index=False,encoding="utf-8-sig")
    dynamics.to_csv(folder/"state_dynamics.csv",index=False,encoding="utf-8-sig")
    missing_sensitivity.to_csv(folder/"one_missing_window_sensitivity.csv",index=False,encoding="utf-8-sig")
    refit_audit.to_csv(folder/"refit_convergence_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(folder/"folds.csv",index=False,encoding="utf-8-sig")
    joblib.dump(models,folder/"selected_fold_models.joblib")
    (folder/"data_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf8")
    config={"name":"Adaptive Regime-Switching — Hieu","version":"0.5","horizons_min":list(HORIZONS),
            "sessions":[list(x) for x in SESSIONS],"non_overlapping":not args.overlapping,"interpolation":False,
            "return_variant":args.return_variant,
            "policy":args.policy,"backend":args.backend,"device":args.device,
            "exclude_zero":args.exclude_zero,"outlier_policy":args.outlier_policy,
            "exclude_rollover_adjacent":args.exclude_rollover_adjacent,
            "allow_one_internal_missing":args.allow_one_internal_missing,
            "selected_fold_ids":[fold.fold for fold in folds],
            "k_values":list(K_VALUES),"occupancy_gate":.01,
            "train_years":args.train_years,"validation_months":6,"test_months":6,"step_months":6,
            "expanding_window":args.expanding_window,
            "iterations":args.iterations,"restarts":args.restarts,"top_horizons":args.top_horizons,
            "selection_iterations":args.selection_iterations,"selection_restarts":args.selection_restarts,
            "online_parameter_learning":False,"python":platform.python_version(),"numpy":np.__version__,
            "pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__}
    (folder/"run_config.json").write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding="utf8")
    (folder/"fit_diagnostics.json").write_text(json.dumps(fit_diag,indent=2,ensure_ascii=False,default=str),encoding="utf8")
    (folder/"horizon_decision.json").write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding="utf8")
    last=max(models);h,family,k=models[last]["champion"];key=f"h{h}__{family}"
    tr,va,te=slice_fold(frames[h],Fold(**models[last]["fold_bounds"]))
    (tr,va,te),_=robust_filter_split(tr,va,te,args.exclude_zero,
                                     args.outlier_policy,args.exclude_rollover_adjacent)
    if args.return_variant=="intraday_adjusted":
        tr,va,te=intraday_adjust_split(tr,va,te)
    dev=pd.concat([tr,va]).sort_index()
    scaler=models[last][f"{key}__scaler"];x=scaler.transform(dev[["log_return"]])
    posterior=causal_filter(models[last][key],x,starts=sequence_starts(dev,args.policy))[0][-1]
    profile=states[(states.fold==last)&(states.model_key==key)];stable=dict(zip(profile.internal_state.astype(int),profile.stable_state_id))
    runtime=RegimeRuntime(models[last][key],scaler,args.policy,f"ARSH-v0.5-h{h}-research",stable,
                          posterior=posterior,last_timestamp=dev.index[-1]);runtime.save(folder/"runtime_research_candidate.joblib")
    manifest={p.name:sha256(p) for p in (folder/"selected_fold_models.joblib",folder/"runtime_research_candidate.joblib",
        HERE/"arsh_v05.py",HERE/"student_t_hmm.py",HERE/"runtime.py")}
    (folder/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    write_report(folder,audit,horizon_audit,fold_rows,candidates,metrics,bootstraps,states,alignments,
                 intraday,dynamics,sensitivity,missing_sensitivity,decision,fit_diag,
                 refit_audit,manifest,args.iterations,args.restarts,args.policy)
    print(json.dumps(decision,ensure_ascii=True,indent=2),flush=True);print(f"Report: {(folder/'report.html').resolve()}",flush=True)


if __name__=="__main__":main()
