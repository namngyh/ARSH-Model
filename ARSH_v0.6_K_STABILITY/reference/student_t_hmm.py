"""Small univariate Student-t HMM used by ARSH v0.3.

The implementation is intentionally narrow: univariate emissions, EM fitting,
multiple independent sequences through ``lengths``, and either a shared or a
state-specific degrees-of-freedom parameter.  It mirrors the pieces of
hmmlearn.GaussianHMM needed by the ARSH causal filter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import digamma, gammaln, logsumexp
from sklearn.cluster import KMeans
from hmmlearn import _hmmc


def _normalize(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 1e-300)
    return values / values.sum(axis=axis, keepdims=axis is not None)


def _slices(n: int, lengths=None):
    if lengths is None:
        return [slice(0, n)]
    lengths = np.asarray(lengths, dtype=int)
    if lengths.ndim != 1 or (lengths <= 0).any() or lengths.sum() != n:
        raise ValueError("lengths must be positive and sum to n_samples")
    ends = np.cumsum(lengths)
    starts = np.r_[0, ends[:-1]]
    return [slice(int(a), int(b)) for a, b in zip(starts, ends)]


@dataclass
class Monitor:
    history: list[float]
    iter: int = 0
    converged: bool = False


class StudentTHMM:
    """Univariate Student-t hidden Markov model fitted by EM/ECM."""

    def __init__(self, n_components: int, df_mode: str = "shared", n_iter: int = 300,
                 tol: float = .01, random_state: int | None = None,
                 min_scale2: float = 1e-6, min_df: float = 2.05, max_df: float = 200.):
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if df_mode not in {"shared", "state"}:
            raise ValueError("df_mode must be 'shared' or 'state'")
        self.n_components = int(n_components)
        self.df_mode = df_mode
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.min_scale2 = float(min_scale2)
        self.min_df = float(min_df)
        self.max_df = float(max_df)
        self.monitor_ = Monitor([])

    def _initialize(self, values: np.ndarray, lengths=None):
        rng = np.random.RandomState(self.random_state)
        km = KMeans(n_clusters=self.n_components, n_init=10,
                    random_state=self.random_state).fit(values[:, None])
        labels = km.labels_
        self.means_ = km.cluster_centers_[:, 0].astype(float)
        global_var = max(float(np.var(values)), self.min_scale2)
        self.scale2_ = np.asarray([
            max(float(np.var(values[labels == k])), self.min_scale2)
            if np.sum(labels == k) > 1 else global_var
            for k in range(self.n_components)], dtype=float)
        # A neutral fixed start avoids running a separate costly Student-t MLE
        # for every K/seed.  EM estimates df from this reproducible value.
        initial_df = 8.
        self.df_ = np.full(self.n_components, initial_df, dtype=float)
        # Derive the Markov parameters from K-means labels within each allowed
        # sequence. Small seeded jitter retains meaningful restart diversity.
        start_counts = np.ones(self.n_components)
        trans_counts = np.ones((self.n_components, self.n_components))
        for segment in _slices(len(values), lengths):
            sequence = labels[segment]
            start_counts[sequence[0]] += 1
            if len(sequence) > 1:
                np.add.at(trans_counts, (sequence[:-1], sequence[1:]), 1)
        start_counts += rng.gamma(1., .01, self.n_components)
        trans_counts += rng.gamma(1., .01, trans_counts.shape)
        self.startprob_ = _normalize(start_counts)
        self.transmat_ = _normalize(trans_counts, axis=1)

    def emission_log_prob(self, x) -> np.ndarray:
        values = np.asarray(x, dtype=float).reshape(-1)
        df = self.df_[None, :]; scale2 = self.scale2_[None, :]
        standardized2 = (values[:, None]-self.means_[None, :])**2/scale2
        return (gammaln((df+1)/2)-gammaln(df/2)
                -.5*(np.log(df*np.pi)+np.log(scale2))
                -(df+1)/2*np.log1p(standardized2/df))

    def _e_step(self, values: np.ndarray, lengths=None):
        logb = self.emission_log_prob(values)
        gamma = np.zeros((len(values), self.n_components))
        start_counts = np.zeros(self.n_components)
        trans_counts = np.zeros((self.n_components, self.n_components))
        total_ll = 0.
        # Group equally sized independent sequences and process every group in
        # one vectorized forward-backward pass.  Daily/session policies create
        # thousands of length-1/2/3 sequences, for which a Python loop around
        # a compiled routine is substantially slower than this batched form.
        segments = _slices(len(values), lengths)
        if len(segments) == 1:
            b = logb[segments[0]]
            ll, alpha = _hmmc.forward_log(self.startprob_, self.transmat_, b)
            beta = _hmmc.backward_log(self.startprob_, self.transmat_, b)
            log_gamma = alpha + beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            g = np.exp(log_gamma)
            gamma[segments[0]] = g
            start_counts += g[0]
            if len(b) > 1:
                trans_counts += np.exp(_hmmc.compute_log_xi_sum(
                    alpha, self.transmat_, beta, b))
            return float(ll), gamma, start_counts, trans_counts
        groups: dict[int, list[slice]] = {}
        for segment in segments:
            groups.setdefault(segment.stop - segment.start, []).append(segment)
        log_start = np.log(np.maximum(self.startprob_, 1e-300))
        log_trans = np.log(np.maximum(self.transmat_, 1e-300))
        for length, same_length in groups.items():
            indices = np.asarray([np.arange(s.start, s.stop) for s in same_length])
            b = logb[indices]                         # sequence x time x state
            count = len(same_length)
            alpha = np.empty((count, length, self.n_components))
            alpha[:, 0] = log_start + b[:, 0]
            for t in range(1, length):
                alpha[:, t] = logsumexp(
                    alpha[:, t - 1, :, None] + log_trans[None, :, :], axis=1) + b[:, t]
            ll = logsumexp(alpha[:, -1], axis=1)
            beta = np.zeros_like(alpha)
            for t in range(length - 2, -1, -1):
                beta[:, t] = logsumexp(
                    log_trans[None, :, :] + b[:, t + 1, None, :]
                    + beta[:, t + 1, None, :], axis=2)
            g = np.exp(alpha + beta - ll[:, None, None])
            g /= g.sum(axis=2, keepdims=True)
            gamma[indices.ravel()] = g.reshape(-1, self.n_components)
            start_counts += g[:, 0].sum(axis=0)
            for t in range(length - 1):
                log_xi = (alpha[:, t, :, None] + log_trans[None, :, :]
                          + b[:, t + 1, None, :] + beta[:, t + 1, None, :]
                          - ll[:, None, None])
                trans_counts += np.exp(log_xi).sum(axis=0)
            total_ll += float(ll.sum())
        return total_ll, gamma, start_counts, trans_counts

    def _update_df(self, gamma, expected_u, expected_log_u):
        def solve(summary: float, old: float) -> float:
            def equation(nu):
                return np.log(nu / 2.) - digamma(nu / 2.) + 1. + summary
            lo, hi = self.min_df, self.max_df
            flo, fhi = equation(lo), equation(hi)
            if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0:
                return float(optimize.brentq(equation, lo, hi, maxiter=100))
            result = optimize.minimize_scalar(lambda nu: equation(nu) ** 2,
                                              bounds=(lo, hi), method="bounded")
            return float(result.x) if result.success else float(old)

        if self.df_mode == "shared":
            summary = float(np.sum(gamma * (expected_log_u - expected_u)) / np.sum(gamma))
            value = solve(summary, float(np.mean(self.df_)))
            return np.full(self.n_components, value)
        result = np.empty(self.n_components)
        for k in range(self.n_components):
            weight = max(float(gamma[:, k].sum()), 1e-12)
            summary = float(np.sum(gamma[:, k] * (expected_log_u[:, k] - expected_u[:, k])) / weight)
            result[k] = solve(summary, self.df_[k])
        return result

    def fit(self, x, lengths=None):
        values = np.asarray(x, dtype=float).reshape(-1)
        if len(values) < max(20, 5 * self.n_components) or not np.isfinite(values).all():
            raise ValueError("invalid or insufficient observations")
        segments = _slices(len(values), lengths)
        self._initialize(values, lengths)
        history: list[float] = []
        for iteration in range(1, self.n_iter + 1):
            ll, gamma, start_counts, trans_counts = self._e_step(values, lengths)
            history.append(ll)
            delta = (values[:, None] - self.means_[None, :]) ** 2 / self.scale2_[None, :]
            expected_u = (self.df_[None, :] + 1.) / (self.df_[None, :] + delta)
            expected_log_u = digamma((self.df_[None, :] + 1.) / 2.) - np.log((self.df_[None, :] + delta) / 2.)
            weights = gamma * expected_u
            denom = np.maximum(weights.sum(axis=0), 1e-12)
            new_means = (weights * values[:, None]).sum(axis=0) / denom
            nk = np.maximum(gamma.sum(axis=0), 1e-12)
            new_scale2 = (weights * (values[:, None] - new_means[None, :]) ** 2).sum(axis=0) / nk
            self.startprob_ = _normalize(start_counts)
            self.transmat_ = _normalize(trans_counts + 1e-12, axis=1)
            self.means_ = new_means
            self.scale2_ = np.maximum(new_scale2, self.min_scale2)
            self.df_ = np.clip(self._update_df(gamma, expected_u, expected_log_u),
                               self.min_df, self.max_df)
            if iteration > 1:
                improvement = history[-1] - history[-2]
                if improvement >= -1e-6 and improvement < self.tol:
                    self.monitor_ = Monitor(history, iteration, True)
                    break
        else:
            self.monitor_ = Monitor(history, self.n_iter, False)
        self.n_features = 1
        self.lengths_ = [segment.stop - segment.start for segment in segments]
        return self

    def score(self, x, lengths=None) -> float:
        return float(self._e_step(np.asarray(x).reshape(-1), lengths)[0])

    @property
    def covars_(self):
        """Compatibility view; Student-t scale squared is not its variance."""
        return self.scale2_[:, None, None]


class GaussianSequenceHMM(StudentTHMM):
    """Univariate Gaussian HMM with the same batched sequence semantics."""

    def __init__(self, n_components: int, n_iter: int = 300, tol: float = .01,
                 random_state: int | None = None, min_scale2: float = 1e-6):
        super().__init__(n_components, df_mode="shared", n_iter=n_iter, tol=tol,
                         random_state=random_state, min_scale2=min_scale2)

    def emission_log_prob(self, x) -> np.ndarray:
        values = np.asarray(x, dtype=float).reshape(-1)
        means = self.means_[:, 0]
        return -.5 * (np.log(2 * np.pi * self.scale2_[None, :])
                      + (values[:, None] - means[None, :]) ** 2
                      / self.scale2_[None, :])

    def fit(self, x, lengths=None):
        values = np.asarray(x, dtype=float).reshape(-1)
        if len(values) < max(20, 5 * self.n_components) or not np.isfinite(values).all():
            raise ValueError("invalid or insufficient observations")
        segments = _slices(len(values), lengths)
        self._initialize(values, lengths)
        history: list[float] = []
        for iteration in range(1, self.n_iter + 1):
            ll, gamma, start_counts, trans_counts = self._e_step(values, lengths)
            history.append(ll)
            nk = np.maximum(gamma.sum(axis=0), 1e-12)
            means = (gamma * values[:, None]).sum(axis=0) / nk
            scale2 = (gamma * (values[:, None] - means[None, :]) ** 2).sum(axis=0) / nk
            self.startprob_ = _normalize(start_counts)
            self.transmat_ = _normalize(trans_counts + 1e-12, axis=1)
            self.means_ = means
            self.scale2_ = np.maximum(scale2, self.min_scale2)
            if iteration > 1:
                improvement = history[-1] - history[-2]
                if improvement >= -1e-6 and improvement < self.tol:
                    self.monitor_ = Monitor(history, iteration, True)
                    break
        else:
            self.monitor_ = Monitor(history, self.n_iter, False)
        self.n_features = 1
        self.lengths_ = [segment.stop - segment.start for segment in segments]
        return self

    def score(self, x, lengths=None) -> float:
        return float(self._e_step(np.asarray(x).reshape(-1), lengths)[0])

    @property
    def means_(self):
        return self._means[:, None]

    @means_.setter
    def means_(self, values):
        self._means = np.asarray(values, dtype=float).reshape(-1)
