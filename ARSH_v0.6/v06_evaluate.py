"""Out-of-sample check of a locked v0.6 model, exactly as V06_TRAINING_PLAN.md section 5 states.

Holdout: 2026-08-01 through the last complete source day on or before the lock date in
artifact_manifest.json. Parameters, scaler and configuration are read from the locked files and
never changed. Written before the model existed; the pass rule below is the plan's rule.

Pass (both required):
  1. mean predictive log density per observation > the iid Student-t baseline fitted on the same
     3-year final window (student_baseline.json);
  2. smallest soft occupancy (mean posterior) of the 7 states >= 1%.
Everything else is descriptive, including the v0.5 artifact on the same observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOLDOUT_START = date(2026, 8, 1)
V05_SHARD = Path("runs_v05_revised/policy_daily/policy_daily__f08_09")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_stats(labels, starts):
    """Mean length of runs of the displayed state, and share of runs lasting one observation."""
    runs, length = [], 0
    for index, label in enumerate(labels):
        if index == 0 or starts[index] or label != labels[index - 1]:
            if length:
                runs.append(length)
            length = 0
        length += 1
    runs.append(length)
    return sum(runs) / len(runs), sum(r == 1 for r in runs) / len(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=HERE / "models" / "v06_k7_cut20260801")
    args = parser.parse_args()
    repo, folder = args.repo.resolve(), args.model.resolve()
    manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact = folder / "runtime_research_candidate.joblib"
    if sha256(artifact) != manifest[artifact.name]:
        raise SystemExit("model artifact differs from its lock manifest")
    lock_day = date.fromisoformat(manifest["locked_at"][:10])

    sys.path.insert(0, str(repo))
    import numpy as np
    import pandas as pd
    import arsh_v05 as arsh
    from runtime import RegimeRuntime
    import v06_fetch_db as fetch

    stored = [m for m in fetch.sync(HOLDOUT_START, lock_day) if m["stored"]]
    if not stored:
        raise SystemExit("no complete holdout days")
    raw = pd.concat(arsh.load_minutes(fetch.DAYS_DIR / m["csv"])[0] for m in stored).sort_index()
    frames, _, rejected, _ = arsh.build_returns(raw, horizons=(1,), overlapping=False,
                                                allow_one_internal_missing=False)
    obs = frames[1]
    starts = arsh.sequence_starts(obs, "daily_sequence")
    values = obs[["log_return"]]

    def score(runtime):
        x = runtime.scaler.transform(values)
        posterior, logscore = arsh.causal_filter(runtime.model, x, starts=starts)
        density = logscore - float(np.log(runtime.scaler.scale_[0]))
        labels = posterior.argmax(1)
        mean_run, one_share = run_stats(labels, starts)
        shares = posterior.mean(0)
        return density, x, {
            "mean_log_density": float(density.mean()),
            "soft_share_by_state": {runtime.stable_state_ids.get(i, str(i)): float(s) for i, s in enumerate(shares)},
            "min_soft_share": float(shares.min()),
            "mean_confidence": float(posterior.max(1).mean()),
            "mean_run_observations": float(mean_run),
            "one_observation_run_share": float(one_share),
        }

    new = RegimeRuntime.load(artifact)
    density, x_new, new_stats = score(new)
    baseline = json.loads((folder / "student_baseline.json").read_text(encoding="utf-8"))["parameters"]
    student = arsh.iid_scores(baseline, x_new, "student_t") - float(np.log(new.scaler.scale_[0]))
    gain = float(density.mean() - student.mean())
    months = pd.Series(density - student, index=obs.index).groupby(obs.index.to_period("M")).mean()
    _, _, old_stats = score(RegimeRuntime.load(repo / V05_SHARD / "runtime_research_candidate.joblib"))

    passed_gain = gain > 0
    passed_share = new_stats["min_soft_share"] >= 0.01
    result = {
        "plan": "V06_TRAINING_PLAN.md section 5",
        "model_version": new.model_version,
        "model_sha256": manifest[artifact.name],
        "locked_at": manifest["locked_at"],
        "holdout_first_day": stored[0]["date"], "holdout_last_day": stored[-1]["date"],
        "holdout_days": len(stored), "observations": int(len(obs)), "rejected_windows": int(len(rejected)),
        "days_with_warnings": {m["date"]: m["warnings"] for m in stored if m["warnings"]},
        "criterion_1_gain_vs_student": gain, "criterion_1_pass": passed_gain,
        "criterion_2_min_soft_share": new_stats["min_soft_share"], "criterion_2_pass": passed_share,
        "result": "PASS" if passed_gain and passed_share else "FAIL",
        "descriptive": {"gain_vs_student_by_month": {str(k): float(v) for k, v in months.items()},
                        "student_baseline_mean_log_density": float(student.mean()),
                        "new_model": new_stats, "v05_artifact_same_observations": old_stats},
    }
    out = folder / "holdout_evaluation.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
