"""Train the v0.6 K=7 model on data before the cut date, following V06_TRAINING_PLAN.md exactly.

Reuses the unmodified v0.5 functions (returns, HMM fits, causal filter, runtime) so the only
differences from v0.5 are the dates and the 3-year final window. All rules come from
v06_training_config.json; this script only executes them and records hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "v06_training_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="v0.5 code folder")
    parser.add_argument("--out", type=Path, default=HERE / "models" / "v06_k7_cut20260801")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    if (out / "artifact_manifest.json").exists():
        raise SystemExit(f"{out} already holds a locked model; refusing to overwrite")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan = HERE / cfg["plan"]

    sys.path.insert(0, str(repo))
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import arsh_v05 as arsh
    from runtime import RegimeRuntime
    import v06_fetch_db as fetch

    k, horizon, policy = cfg["k"], cfg["horizon_min"], cfg["policy"]
    cut = pd.Timestamp(cfg["data_before"])
    sel_start, val_start = pd.Timestamp(cfg["selection_train_start"]), pd.Timestamp(cfg["selection_validation_start"])
    final_start = pd.Timestamp(cfg["final_train_start"])
    for seeds in (cfg["selection_seeds"], cfg["final_seeds"]):
        if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
            raise SystemExit("seeds must be consecutive (v0.5 fit_hmm_restarts convention)")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "_checkpoints"
    cache.mkdir(exist_ok=True)

    # 1. Verified complete days before the cut, from the database adapter.
    first_day = min(sel_start, final_start).date()
    metas = fetch.sync(first_day, (cut - pd.Timedelta(days=1)).date())
    incomplete = [m["date"] for m in metas if not m["stored"]]
    stored = [m for m in metas if m["stored"]]
    if not stored:
        raise SystemExit("no complete days before the cut date")
    last_day = stored[-1]["date"]
    training_csv = out / f"training_data_{first_day}_{last_day}.csv"
    with training_csv.open("w", encoding="utf-8", newline="") as target:
        for index, meta in enumerate(stored):
            lines = (fetch.DAYS_DIR / meta["csv"]).read_text(encoding="utf-8").splitlines(keepends=True)
            target.writelines(lines if index == 0 else lines[1:])
    arsh.emit_progress(f"DATA {len(stored)} complete days {first_day}..{last_day}; incomplete skipped: {incomplete}")

    raw, data_audit = arsh.load_minutes(training_csv)
    frames, _, rejected, _ = arsh.build_returns(raw, horizons=(horizon,), overlapping=False,
                                                allow_one_internal_missing=False)
    frame = frames[horizon]
    if frame.index.max() >= cut:
        raise SystemExit("training returns reach the cut date")

    # 2. Family selection on validation.
    train = frame[(frame.index >= sel_start) & (frame.index < val_start)]
    valid = frame[(frame.index >= val_start) & (frame.index < cut)]
    scaler = StandardScaler().fit(train[["log_return"]])
    xt, xv = scaler.transform(train[["log_return"]]), scaler.transform(valid[["log_return"]])
    jac = float(np.log(scaler.scale_[0]))
    _, student = arsh.fit_iid(xt)
    student_valid = float((arsh.iid_scores(student, xv, "student_t") - jac).mean())
    rows, diagnostics = [], []
    for family in cfg["families_simplest_first"]:
        model, diag = arsh.fit_hmm_restarts(
            xt, family, k, cfg["selection_iterations"], len(cfg["selection_seeds"]),
            cfg["selection_seeds"][0], horizon, "v06_selection", cache,
            arsh.sequence_lengths(train, policy), args.backend, "cuda:0")
        diagnostics.extend(diag)
        prior = arsh.causal_filter(model, xt, starts=arsh.sequence_starts(train, policy))[0][-1]
        prob, score = arsh.causal_filter(model, xv, prior, arsh.sequence_starts(valid, policy))
        density = float(score.mean() - jac)
        share = float(prob.mean(0).min())
        rows.append({"family": family, "validation_log_density": density,
                     "validation_gain_vs_student": density - student_valid,
                     "validation_min_soft_share": share,
                     "eligible": share >= cfg["min_soft_share"],
                     "converged": bool(model.practical_converged_)})
    selection = pd.DataFrame(rows)
    eligible = selection[selection.eligible]
    if eligible.empty:
        selection.to_csv(out / "selection.csv", index=False, encoding="utf-8-sig")
        raise SystemExit("no eligible family; stopping without a model (plan section 3)")
    best = float(eligible.validation_log_density.max())
    tied = eligible[eligible.validation_log_density >= best - cfg["tie_margin"]]
    order = cfg["families_simplest_first"]
    chosen = min(tied.family, key=order.index)
    selection["best_minus_family"] = best - selection.validation_log_density
    selection["within_tie_margin"] = selection.family.isin(tied.family)
    selection["chosen"] = selection.family.eq(chosen)
    selection.to_csv(out / "selection.csv", index=False, encoding="utf-8-sig")
    arsh.emit_progress(f"SELECTED {chosen}\n{selection.to_string(index=False)}")

    # 3. Final fit on the last 3 years before the cut.
    dev = frame[(frame.index >= final_start) & (frame.index < cut)]
    scaler = StandardScaler().fit(dev[["log_return"]])
    xd = scaler.transform(dev[["log_return"]])
    model, diag = arsh.fit_hmm_restarts(
        xd, chosen, k, cfg["final_iterations"], len(cfg["final_seeds"]), cfg["final_seeds"][0],
        horizon, "v06_final", cache, arsh.sequence_lengths(dev, policy), args.backend, "cuda:0")
    diagnostics.extend(diag)
    _, _, stds, _ = arsh.distribution(model, chosen, scaler)
    stable = {int(internal): f"v06::S{rank}" for rank, internal in enumerate(np.argsort(stds))}
    posterior = arsh.causal_filter(model, xd, starts=arsh.sequence_starts(dev, policy))[0][-1]
    runtime = RegimeRuntime(model, scaler, policy, cfg["model_version"], stable,
                            posterior=posterior, last_timestamp=dev.index[-1])
    runtime.save(out / "runtime_research_candidate.joblib")
    _, final_student = arsh.fit_iid(xd)
    (out / "student_baseline.json").write_text(json.dumps(
        {"note": "iid Student-t fitted on the final window, in scaler space; holdout baseline (plan 5.1)",
         "parameters": final_student}, indent=2, default=float) + "\n", encoding="utf-8")

    # 4. Metadata in the layout v06_replay_eod.py expects, then the lock (manifest).
    (out / "fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "policy": policy, "return_variant": "raw", "non_overlapping": True,
        "allow_one_internal_missing": False, "k": k, "horizon_min": horizon, "family": chosen,
        "selection_train": [str(sel_start.date()), str(val_start.date())],
        "selection_validation": [str(val_start.date()), str(cut.date())],
        "final_train": [str(final_start.date()), str(cut.date())],
        "last_training_day": last_day, "incomplete_days_skipped": incomplete,
        "final_train_observations": int(len(dev)), "rejected_windows": int(len(rejected)),
        "final_converged": bool(model.practical_converged_),
        "model_version": cfg["model_version"], "backend": args.backend,
        "online_parameter_learning": False}, indent=2), encoding="utf-8")
    with (out / "folds.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fold", "train_start", "train_end_exclusive", "validation_end_exclusive",
                         "test_end_exclusive", "champion_horizon", "champion_family", "champion_k"])
        writer.writerow([0, final_start.date(), cut.date(), cut.date(), "", horizon, chosen, k])
    manifest = {
        "runtime_research_candidate.joblib": sha256(out / "runtime_research_candidate.joblib"),
        training_csv.name: sha256(training_csv),
        "V06_TRAINING_PLAN.md": sha256(plan),
        "v06_training_config.json": sha256(CONFIG),
        "v06_train.py": sha256(Path(__file__).resolve()),
        "arsh_v05.py": sha256(repo / "arsh_v05.py"),
        "student_t_hmm.py": sha256(repo / "student_t_hmm.py"),
        "runtime.py": sha256(repo / "runtime.py"),
        "locked_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "last_training_day": last_day,
    }
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    arsh.emit_progress(f"LOCKED {out} | family={chosen} | last training day {last_day}")


if __name__ == "__main__":
    main()
