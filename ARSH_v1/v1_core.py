"""Causal ARSH v1 replay and historical diagnostics.

The historical data contract is the same as the frozen v0.6/v0.5 return
builder. Model files are loaded only after their published SHA-256 is checked.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
from arsh_v05 import build_returns, load_minutes, sequence_starts  # noqa: E402

EXPECTED_HASHES = {
    "strict": "4c67a274092bdcde4cd7c00af89b7308268ac8360285c3346450b30e3c86d69f",
    "original": "9071bb38709abb3ba91012b1d4a7e18d3285e0e1de7bf2c37e8a186091cf46e1",
}
MODEL_PATHS = {
    "strict": ROOT / "models" / "strict_k7.joblib",
    "original": ROOT / "models" / "original_k7.joblib",
}
STATES = tuple(f"S{i}" for i in range(7))
CONTINUOUS = (("AM", "09:00", "11:30"), ("PM", "13:00", "14:30"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_for(name: str):
    path = MODEL_PATHS[name]
    actual = sha256(path)
    if actual != EXPECTED_HASHES[name]:
        raise ValueError(f"{name} artifact hash mismatch: {actual}")
    runtime = joblib.load(path)
    if runtime.policy != "daily_sequence" or runtime.model.n_components != 7:
        raise ValueError("Artifact is not the locked K=7 daily model")
    if runtime.model.__class__.__name__ != "StudentTHMM":
        raise ValueError("Artifact has an unexpected emission family")
    mapped = {int(i): str(label).split("::")[-1] for i, label in runtime.stable_state_ids.items()}
    if set(mapped.values()) != set(STATES) or set(mapped) != set(range(7)):
        raise ValueError(f"State mapping is not a permutation of S0..S6: {mapped}")
    stable_to_internal = np.array([next(i for i, label in mapped.items() if label == state)
                                   for state in STATES], dtype=int)
    return runtime, actual, stable_to_internal


def expand_sources(sources: list[Path]) -> list[Path]:
    files = []
    for source in sources:
        source = source.resolve()
        if source.is_file():
            files.append(source)
        elif source.is_dir():
            # v06_fetch_db writes immutable day versions. Use latest version
            # for each day, but never silently resolve collisions across inputs.
            latest: dict[str, tuple[int, Path]] = {}
            for path in source.glob("VN30F1M_*_v*.csv"):
                match = re.fullmatch(r"VN30F1M_(\d{4}-\d{2}-\d{2})_v(\d+)\.csv", path.name)
                if match:
                    day, version = match.group(1), int(match.group(2))
                    if day not in latest or version > latest[day][0]:
                        latest[day] = (version, path)
            if not latest:
                raise ValueError(f"No versioned VN30F1M day CSVs in {source}")
            files.extend(path for _, path in sorted(latest.values(), key=lambda x: x[1].name))
        else:
            raise FileNotFoundError(source)
    if not files:
        raise ValueError("No source CSVs supplied")
    return files


def load_sources(sources: list[Path]):
    files = expand_sources(sources)
    frames = []
    audit = []
    for path in files:
        raw, info = load_minutes(path)
        if "VOL" in raw:
            raw["VOL"] = pd.to_numeric(raw["VOL"], errors="coerce")
        frames.append(raw)
        audit.append({
            "path": str(path), "sha256": info["source_sha256"],
            "rows": int(info["rows"]), "first": info["first"],
            "last": info["last"], "days": int(info["days"]),
            "columns": list(raw.columns),
        })
    raw = pd.concat(frames).sort_index()
    if raw.index.has_duplicates:
        duplicated = raw.index[raw.index.duplicated()].unique()[:5]
        raise ValueError(f"Overlapping source files: duplicate bars {list(map(str, duplicated))}")
    return raw, audit


def make_observations(raw: pd.DataFrame):
    frames, return_audit, rejected, _ = build_returns(
        raw, horizons=(1,), overlapping=False, allow_one_internal_missing=False
    )
    obs = frames[1]
    if obs.empty:
        raise ValueError("No accepted one-minute returns")
    starts = sequence_starts(obs, "daily_sequence")
    # Each return ends at the close of its endpoint bar. Historical CSV times
    # label bar starts, so the bar is complete one minute later.
    obs = obs.copy()
    obs["volume"] = pd.to_numeric(raw["VOL"].reindex(obs.index), errors="coerce") if "VOL" in raw else np.nan
    obs["event_bar_end"] = obs.index + pd.Timedelta(minutes=1)
    return obs, starts, {
        "accepted_returns": len(obs),
        "rejected_windows": len(rejected),
        "return_audit": return_audit.to_dict(orient="records"),
    }


def period_name(timestamp) -> str:
    day = pd.Timestamp(timestamp).date()
    if day < pd.Timestamp("2026-08-01").date():
        return "through_2026_07_model_development"
    if day < pd.Timestamp("2026-09-30").date():
        return "aug_sep_seen_before_strict_lock"
    return "from_2026_09_30_post_lock_calendar"


def filter_observations(runtime, stable_to_internal: np.ndarray,
                        obs: pd.DataFrame, starts: np.ndarray):
    values = obs["log_return"].to_numpy(dtype=float)
    feature_name = str(runtime.scaler.feature_names_in_[0]) if hasattr(
        runtime.scaler, "feature_names_in_") else "log_return"
    x = runtime.scaler.transform(pd.DataFrame({feature_name: values})).reshape(-1)
    log_b = runtime.model.emission_log_prob(x)
    if log_b.shape != (len(obs), 7) or not np.isfinite(log_b).all():
        raise ValueError("Invalid emission log probabilities")
    trans = np.asarray(runtime.model.transmat_, dtype=float)
    start = np.asarray(runtime.model.startprob_, dtype=float)
    if not np.allclose(trans.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Transition rows do not sum to one")
    posterior = np.empty((len(obs), 7), dtype=float)
    score = np.empty(len(obs), dtype=float)
    soft_trans = np.zeros((7, 7), dtype=float)
    prev = None
    for n in range(len(obs)):
        prior = start if starts[n] else prev @ trans
        joint = np.log(np.maximum(prior, 1e-300)) + log_b[n]
        log_score = float(logsumexp(joint))
        prev = np.exp(joint - log_score)
        posterior[n] = prev
        score[n] = log_score - math.log(float(runtime.scaler.scale_[0]))
        if n and not starts[n]:
            # Filtered pair probability given observations through this bar.
            log_weights = (
                np.log(np.maximum(posterior[n - 1, :, None], 1e-300))
                + np.log(np.maximum(trans, 1e-300))
                + log_b[n][None, :]
            )
            soft_trans += np.exp(log_weights - logsumexp(log_weights))
    if not np.allclose(posterior.sum(axis=1), 1, atol=1e-9):
        raise ValueError("Posterior normalization failed")
    return posterior[:, stable_to_internal], score, soft_trans[np.ix_(stable_to_internal, stable_to_internal)]


def entropy_stats(posterior: np.ndarray):
    entropy = -np.sum(posterior * np.log(np.maximum(posterior, 1e-300)), axis=1)
    normalized = entropy / math.log(7)
    return entropy, {
        "count": int(len(entropy)),
        "mean_nats": float(entropy.mean()),
        "median_nats": float(np.median(entropy)),
        "mean_normalized": float(normalized.mean()),
        "p90_normalized": float(np.quantile(normalized, 0.9)),
        "fraction_normalized_above_0_8": float(np.mean(normalized > 0.8)),
    }


def overlap_jsd(runtime, stable_to_internal: np.ndarray):
    """Equal-prior emission overlap and JSD; numerical tails are bounded."""
    model = runtime.model
    means = np.asarray(model.means_, dtype=float)[stable_to_internal]
    scales = np.sqrt(np.asarray(model.scale2_, dtype=float)[stable_to_internal])
    dfs = np.asarray(model.df_, dtype=float)[stable_to_internal]
    overlap = np.eye(7)
    jsd = np.zeros((7, 7), dtype=float)
    tail_quantile = 1e-7
    for i in range(7):
        for j in range(i + 1, 7):
            left = min(student_t.ppf(tail_quantile, dfs[q], loc=means[q], scale=scales[q])
                       for q in (i, j))
            right = max(student_t.ppf(1 - tail_quantile, dfs[q], loc=means[q], scale=scales[q])
                        for q in (i, j))
            grid = np.linspace(left, right, 8001)
            f = student_t.pdf(grid, dfs[i], loc=means[i], scale=scales[i])
            g = student_t.pdf(grid, dfs[j], loc=means[j], scale=scales[j])
            f /= np.trapezoid(f, grid)
            g /= np.trapezoid(g, grid)
            m = (f + g) / 2
            overlap[i, j] = overlap[j, i] = float(np.trapezoid(np.minimum(f, g), grid))
            divergence = 0.5 * np.trapezoid(f * np.log(np.maximum(f / m, 1e-300)), grid)
            divergence += 0.5 * np.trapezoid(g * np.log(np.maximum(g / m, 1e-300)), grid)
            jsd[i, j] = jsd[j, i] = float(divergence)
    pairs = [
        {"states": [STATES[i], STATES[j]],
         "overlap": float(overlap[i, j]), "jsd_nats": float(jsd[i, j])}
        for i in range(7) for j in range(i + 1, 7)
    ]
    return {
        "overlap_matrix": overlap.tolist(),
        "jsd_nats_matrix": jsd.tolist(),
        "pairs": sorted(pairs, key=lambda row: row["overlap"], reverse=True),
        "method": "equal-prior Student-t emission density, standardized scale",
        "tail_quantile_each_side": tail_quantile,
        "warning": "Emission separation alone does not prove useful market regimes.",
    }


def transition_duration(runtime, stable_to_internal: np.ndarray,
                        posterior: np.ndarray, starts: np.ndarray):
    theoretical = np.asarray(runtime.model.transmat_)[np.ix_(stable_to_internal, stable_to_internal)]
    labels = np.argmax(posterior, axis=1)
    counts = np.zeros((7, 7), dtype=int)
    runs: list[list[int]] = [[] for _ in range(7)]
    current = int(labels[0])
    length = 1
    for n in range(1, len(labels)):
        if not starts[n]:
            counts[labels[n - 1], labels[n]] += 1
        if starts[n] or labels[n] != current:
            runs[current].append(length)
            current, length = int(labels[n]), 1
        else:
            length += 1
    runs[current].append(length)
    empirical = np.divide(counts, counts.sum(axis=1, keepdims=True),
                          out=np.zeros_like(counts, dtype=float),
                          where=counts.sum(axis=1, keepdims=True) > 0)
    return {
        "states": list(STATES),
        "theoretical_transition": theoretical.tolist(),
        "theoretical_duration_observations": [
            float(1 / max(1 - theoretical[i, i], 1e-12)) for i in range(7)
        ],
        "observed_argmax_transition_counts": counts.tolist(),
        "observed_argmax_transition_rates": empirical.tolist(),
        "observed_argmax_runs": [
            {"state": STATES[i], "run_count": len(runs[i]),
             "mean_observations": float(np.mean(runs[i])) if runs[i] else None,
             "median_observations": float(np.median(runs[i])) if runs[i] else None,
             "one_observation_fraction": float(np.mean(np.asarray(runs[i]) == 1)) if runs[i] else None}
            for i in range(7)
        ],
        "warning": "HMM expected duration and runs of argmax labels measure different things.",
    }


def economic_profiles(obs: pd.DataFrame, posterior: np.ndarray, starts: np.ndarray):
    returns = obs["log_return"].to_numpy(dtype=float)
    volume = obs["volume"].to_numpy(dtype=float)
    session = obs["session"].astype(str).to_numpy()
    next_valid = np.zeros(len(obs), dtype=bool)
    next_valid[:-1] = (
        ~starts[1:]
        & (np.diff(obs.index.to_numpy()) == np.timedelta64(1, "m"))
        & (session[1:] == session[:-1])
    )
    next_returns = np.full(len(obs), np.nan)
    next_returns[:-1] = returns[1:]
    profiles = []
    hard = np.argmax(posterior, axis=1)
    for i, state in enumerate(STATES):
        weights = posterior[:, i]
        total = float(weights.sum())
        if total <= 0:
            profiles.append({"state": state, "soft_weight": 0.0, "hard_count": 0})
            continue
        mean = float(np.average(returns, weights=weights))
        variance = float(np.average((returns - mean) ** 2, weights=weights))
        future_weight = weights * next_valid
        volume_valid = np.isfinite(volume)
        profiles.append({
            "state": state, "soft_weight": total, "soft_share": total / len(obs),
            "hard_count": int(np.sum(hard == i)),
            "contemporaneous_mean_return_pct": mean * 100,
            "contemporaneous_std_return_pct": math.sqrt(variance) * 100,
            "contemporaneous_mean_abs_return_pct": float(np.average(np.abs(returns), weights=weights)) * 100,
            "next_observation_mean_return_pct": (
                float(np.nansum(next_returns * future_weight) / future_weight.sum() * 100)
                if future_weight.sum() else None
            ),
            "next_observation_mean_abs_return_pct": (
                float(np.nansum(np.abs(next_returns) * future_weight) / future_weight.sum() * 100)
                if future_weight.sum() else None
            ),
            "am_soft_share_within_state": float(weights[session == "AM"].sum() / total),
            "mean_volume": (
                float(np.average(volume[volume_valid], weights=weights[volume_valid]))
                if volume_valid.any() and weights[volume_valid].sum() else None
            ),
        })
    return {
        "profiles": profiles,
        "warning": "Profiles are descriptive. Next-observation values use only contiguous observations; they do not establish tradable predictability.",
    }


def analyze_periods(runtime, stable_to_internal: np.ndarray,
                    obs: pd.DataFrame, starts: np.ndarray, posterior: np.ndarray,
                    score: np.ndarray):
    periods = np.asarray([period_name(t) for t in obs.index])
    report = {}
    for period in dict.fromkeys(periods):
        mask = periods == period
        sub_obs = obs.iloc[np.flatnonzero(mask)]
        sub_p = posterior[mask]
        sub_starts = starts[mask].copy()
        sub_starts[0] = True
        _, entropy = entropy_stats(sub_p)
        report[period] = {
            "first": str(sub_obs.index[0]), "last": str(sub_obs.index[-1]),
            "observations": len(sub_obs), "days": int(sub_obs.index.normalize().nunique()),
            "mean_predictive_log_density": float(np.mean(score[mask])),
            "posterior_entropy": entropy,
            "transition_and_duration": transition_duration(runtime, stable_to_internal, sub_p, sub_starts),
            "economic_profiles": economic_profiles(sub_obs, sub_p, sub_starts),
        }
    return report


def records_frame(obs: pd.DataFrame, starts: np.ndarray, posterior: np.ndarray,
                  score: np.ndarray, model_name: str, model_hash: str,
                  model_version: str):
    entropy, _ = entropy_stats(posterior)
    best = np.argmax(posterior, axis=1)
    result = pd.DataFrame({
        "timestamp_label": obs.index.astype(str),
        "event_bar_end": obs["event_bar_end"].astype(str).to_numpy(),
        # Historical CSVs have no per-bar arrival time. A live adapter must
        # fill and verify available_at before this output is joined to Modus.
        "available_at": "",
        "availability_status": "unknown_historical_csv",
        "date": pd.to_datetime(obs["date"]).dt.date.astype(str).to_numpy(),
        "session": obs["session"].to_numpy(),
        "sequence_reset": starts,
        "log_return": obs["log_return"].to_numpy(),
        "volume": obs["volume"].to_numpy(),
        "model_name": model_name,
        "model_version": model_version,
        "model_sha256": model_hash,
        "state_id": [STATES[i] for i in best],
        "confidence": posterior[np.arange(len(obs)), best],
        "posterior_entropy_nats": entropy,
        "predictive_log_density": score,
        "period_class": [period_name(t) for t in obs.index],
    })
    for i, state in enumerate(STATES):
        result[f"p_{state}"] = posterior[:, i]
    return result
