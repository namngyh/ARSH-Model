"""Causal display-label confirmation. Raw posterior and predictive scores stay intact."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConfirmationRule:
    name: str
    min_probability: float = .6
    min_margin: float = .1
    confirm_observations: int = 2

    def __post_init__(self):
        if not 0 <= self.min_probability <= 1 or not 0 <= self.min_margin <= 1:
            raise ValueError('Probability and margin must be between zero and one')
        if not isinstance(self.confirm_observations, int) or self.confirm_observations < 1:
            raise ValueError('confirm_observations must be a positive integer')


def confirm_labels(posterior, starts, rule):
    p = np.asarray(posterior, dtype=float)
    starts = np.asarray(starts, dtype=bool)
    if p.ndim != 2 or len(p) == 0 or starts.shape != (len(p),):
        raise ValueError('Posterior must be nonempty n x K with n reset flags')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-9, rtol=0):
        raise ValueError('Invalid probability rows')
    raw = p.argmax(1)
    displayed = np.empty(len(p), dtype=int)
    waits = np.zeros(len(p), dtype=int)
    confirmation_start = np.full(len(p),-1,dtype=int)
    changed = np.zeros(len(p), dtype=bool)
    current, pending, count, pending_start = None, None, 0, None
    for t, candidate in enumerate(raw):
        if current is None or starts[t]:
            current, pending, count, pending_start = int(candidate), None, 0, None
        elif candidate == current:
            pending, count, pending_start = None, 0, None
        elif p[t, candidate] >= rule.min_probability and p[t, candidate] - p[t, current] >= rule.min_margin:
            if pending == candidate:
                count += 1
            else:
                pending, count, pending_start = int(candidate), 1, t
            if count >= rule.confirm_observations:
                waits[t] = count - 1
                changed[t] = True
                confirmation_start[t] = pending_start
                current, pending, count, pending_start = int(candidate), None, 0, None
        else:
            pending, count, pending_start = None, 0, None
        displayed[t] = current
    display_probability = p[np.arange(len(p)), displayed]
    uncertain = (p.max(1) < rule.min_probability) | (display_probability < rule.min_probability)
    return {
        'raw_label': raw, 'confirmed_label':displayed,
        'display_label':np.where(uncertain,-1,displayed),
        'raw_confidence': p.max(1), 'display_probability': display_probability,
        'uncertain':uncertain,
        'changed': changed, 'confirmation_wait_observations': waits,
        'confirmation_start_index':confirmation_start,
    }


def label_metrics(result, starts):
    starts = np.asarray(starts, dtype=bool)
    raw, display = result['raw_label'], result['confirmed_label']
    def runs(labels):
        boundaries = np.r_[0, np.flatnonzero(starts[1:] | (labels[1:] != labels[:-1])) + 1, len(labels)]
        lengths = np.diff(boundaries)
        switches = np.count_nonzero((labels[1:] != labels[:-1]) & ~starts[1:])
        return {'switches': int(switches), 'run_count': int(len(lengths)),
                'mean_run': float(lengths.mean()), 'median_run': float(np.median(lengths)),
                'one_observation_run_share': float((lengths == 1).mean())}
    accepted = result['confirmation_wait_observations'][result['changed']]
    return {
        'raw': runs(raw), 'display': runs(display),
        'display_definition':'Run metrics track confirmation memory. Unsupported labels are emitted as UNCERTAIN, so these durations are not evidence of actual regime duration.',
        'confirmed_label_coverage':float((~result['uncertain']).mean()),
        'disagreement_with_raw_share': float((raw != display).mean()),
        'uncertain_share': float(result['uncertain'].mean()),
        'held_label_probability_below_10pct_share': float((result['display_probability'] < .1).mean()),
        'mean_display_probability': float(result['display_probability'].mean()),
        'mean_wait_after_first_qualifying_observation': float(accepted.mean()) if len(accepted) else None,
        'delay_definition': 'Wait after the first consecutive qualifying observation; not delay against a true market regime.',
    }
