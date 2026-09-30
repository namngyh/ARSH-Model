"""Experimental causal parameter adaptation for replay only.

This is a conservative exponentially weighted moment and transition updater,
not the exact online EM algorithm for a Student-t HMM. It is deliberately
unable to replace or overwrite the frozen artifact.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from v1_core import STATES


def run_shadow(runtime, stable_to_internal: np.ndarray, obs: pd.DataFrame,
               starts: np.ndarray, frozen_score: np.ndarray, shadow_start: str,
               decay: float = 0.9999, prior_strength: float = 1000.0):
    if not 0.9 <= decay < 1 or prior_strength < 10:
        raise ValueError("Unsafe shadow updater configuration")
    threshold = pd.Timestamp(shadow_start)
    active = np.asarray(obs.index >= threshold, dtype=bool)
    if not active.any():
        return None, {"status": "no_observations_on_or_after_shadow_start",
                      "shadow_start": shadow_start, "updates": 0}
    model = copy.deepcopy(runtime.model)
    original_mean = np.asarray(model.means_, dtype=float).copy()
    original_scale2 = np.asarray(model.scale2_, dtype=float).copy()
    original_trans = np.asarray(model.transmat_, dtype=float).copy()
    df = np.asarray(model.df_, dtype=float)
    original_variance = original_scale2 * df / np.maximum(df - 2, 0.05)
    effective_n = np.full(7, prior_strength, dtype=float)
    first = effective_n * original_mean
    second = effective_n * (original_variance + original_mean ** 2)
    trans_counts = prior_strength * original_trans
    feature_name = str(runtime.scaler.feature_names_in_[0]) if hasattr(
        runtime.scaler, "feature_names_in_") else "log_return"
    x = runtime.scaler.transform(
        pd.DataFrame({feature_name: obs["log_return"].to_numpy()})
    ).reshape(-1)
    scale_log = math.log(float(runtime.scaler.scale_[0]))
    result_p = np.empty((len(obs), 7))
    result_score = np.empty(len(obs))
    prev = None
    updates = 0
    for n, value in enumerate(x):
        log_b = model.emission_log_prob(np.array([value]))[0]
        prior = model.startprob_ if starts[n] else prev @ model.transmat_
        joint = np.log(np.maximum(prior, 1e-300)) + log_b
        log_score = float(logsumexp(joint))
        current = np.exp(joint - log_score)
        result_p[n] = current
        result_score[n] = log_score - scale_log
        if active[n]:
            # The current observation is scored BEFORE any update from it.
            effective_n *= decay
            first *= decay
            second *= decay
            trans_counts *= decay
            effective_n += current
            first += current * value
            second += current * value * value
            if n and not starts[n]:
                log_xi = (
                    np.log(np.maximum(prev[:, None], 1e-300))
                    + np.log(np.maximum(model.transmat_, 1e-300))
                    + log_b[None, :]
                )
                trans_counts += np.exp(log_xi - logsumexp(log_xi))
            new_mean = first / np.maximum(effective_n, 1e-12)
            new_variance = np.maximum(second / np.maximum(effective_n, 1e-12) - new_mean ** 2, 1e-8)
            new_scale2 = new_variance * np.maximum(df - 2, 0.05) / df
            # Bound drift so this research replay cannot silently relabel a
            # state or collapse an emission into a near point mass.
            mean_bound = 0.5 * np.sqrt(original_variance)
            model.means_ = np.clip(new_mean, original_mean - mean_bound,
                                   original_mean + mean_bound)
            model.scale2_ = np.clip(new_scale2, 0.25 * original_scale2,
                                    4.0 * original_scale2)
            model.transmat_ = trans_counts / trans_counts.sum(axis=1, keepdims=True)
            updates += 1
        prev = current
    comparison = result_score[active] - frozen_score[active]
    report = {
        "status": "experimental_shadow_only",
        "method": "prequential filtered posterior, exponentially weighted moment/transition update; df and scaler fixed",
        "not_exact_online_em": True,
        "shadow_start": shadow_start, "updates": updates,
        "first_active_observation": str(obs.index[np.flatnonzero(active)[0]]),
        "last_active_observation": str(obs.index[np.flatnonzero(active)[-1]]),
        "mean_log_density_frozen": float(np.mean(frozen_score[active])),
        "mean_log_density_shadow": float(np.mean(result_score[active])),
        "mean_shadow_minus_frozen": float(np.mean(comparison)),
        "fraction_shadow_better_per_observation": float(np.mean(comparison > 0)),
        "max_standardized_mean_shift": float(np.max(np.abs(model.means_ - original_mean))),
        "max_scale_ratio": float(np.max(model.scale2_ / original_scale2)),
        "min_scale_ratio": float(np.min(model.scale2_ / original_scale2)),
        "decay": decay, "prior_strength_per_state": prior_strength,
        "warning": "Research comparison only; no promotion rule, no production parameter update.",
    }
    records = pd.DataFrame({
        "timestamp_label": obs.index.astype(str),
        "shadow_active": active,
        "frozen_log_density": frozen_score,
        "shadow_log_density": result_score,
        "shadow_state_id": [STATES[i] for i in np.argmax(result_p[:, stable_to_internal], axis=1)],
    })
    for i, state in enumerate(STATES):
        records[f"shadow_p_{state}"] = result_p[:, stable_to_internal[i]]
    return records, report
