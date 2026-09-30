"""Serializable causal state engine shared by ARSH v0.3 and future v1."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp


def emission_log_prob(model, x: float) -> np.ndarray:
    if hasattr(model, "emission_log_prob"):
        return model.emission_log_prob(np.array([x]))[0]
    means = model.means_[:, 0]
    var = np.diagonal(model.covars_, axis1=1, axis2=2)[:, 0]
    return -.5 * (np.log(2 * np.pi * var) + (x - means) ** 2 / var)


@dataclass
class RegimeRuntime:
    model: object
    scaler: object
    policy: str
    model_version: str
    stable_state_ids: dict[int, str]
    posterior: np.ndarray | None = None
    last_timestamp: pd.Timestamp | None = None
    audit_log: list[dict] = field(default_factory=list)

    def _reset_key(self, timestamp: pd.Timestamp):
        if self.policy == "continuous_carry":
            return "continuous"
        if self.policy == "daily_sequence":
            return str(timestamp.date())
        if self.policy == "session_sequence":
            return f"{timestamp.date()}-{'AM' if timestamp.hour <= 11 else 'PM'}"
        raise ValueError(f"unknown policy: {self.policy}")

    def predict_one(self, log_return: float, timestamp) -> dict:
        timestamp = pd.Timestamp(timestamp)
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("timestamp must be strictly newer than the previous observation")
        if hasattr(self.scaler, "feature_names_in_"):
            sample = pd.DataFrame([[float(log_return)]], columns=self.scaler.feature_names_in_)
        else:
            sample = [[float(log_return)]]
        x = float(self.scaler.transform(sample)[0, 0])
        reset = self.posterior is None
        if self.last_timestamp is not None:
            reset |= self._reset_key(timestamp) != self._reset_key(self.last_timestamp)
        prior = self.model.startprob_ if reset else self.posterior @ self.model.transmat_
        log_joint = np.log(np.maximum(prior, 1e-300)) + emission_log_prob(self.model, x)
        log_score_std = float(logsumexp(log_joint))
        posterior = np.exp(log_joint - log_score_std)
        state = int(np.argmax(posterior))
        result = {
            "timestamp": str(timestamp), "model_version": self.model_version,
            "policy": self.policy, "reset": bool(reset),
            "predictive_log_density": log_score_std - float(np.log(self.scaler.scale_[0])),
            "internal_state": state,
            "stable_state_id": self.stable_state_ids.get(state, f"UNMAPPED_{state}"),
            "confidence": float(posterior[state]),
            "posterior": posterior.tolist(),
        }
        self.posterior = posterior
        self.last_timestamp = timestamp
        self.audit_log.append(result)
        return result

    def reset(self):
        self.posterior = None
        self.last_timestamp = None

    def save(self, path: Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "RegimeRuntime":
        return joblib.load(path)
