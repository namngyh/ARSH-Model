"""Run one historical VN30F1M day through a saved K=7 model and write an EOD report.

This is an engineering replay using the v0.5 CSV convention. A live provider needs
its own timestamp/availability adapter before the result is treated as live data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


DEFAULT_SHARD = Path("runs_v05_revised/policy_daily/policy_daily__f08_09")
EXPECTED_POLICY = "daily_sequence"
HMM_FAMILIES = ("gaussian_hmm", "student_t_shared", "student_t_state")
EXPECTED_HORIZON = 1
EXPECTED_K = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="ARSH-Model clone or v0.5 portable folder")
    parser.add_argument("--data", type=Path, help="v0.5-format VN30F1M CSV; default: REPO/data/ohlc_export.csv")
    parser.add_argument("--shard", type=Path, help="saved K=7 shard; default: latest daily shard")
    parser.add_argument("--date", help="YYYY-MM-DD; default: latest date present in the CSV")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs" / "eod")
    parser.add_argument("--source-meta", type=Path,
                        help="JSON sidecar from v06_fetch_db.py; sets completeness status and data warnings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    data = (args.data or repo / "data" / "ohlc_export.csv").resolve()
    shard = (args.shard or repo / DEFAULT_SHARD).resolve()
    for path in (repo / "runtime.py", repo / "student_t_hmm.py", repo / "arsh_v05.py",
                 data, shard / "runtime_research_candidate.joblib", shard / "artifact_manifest.json",
                 shard / "run_config.json", shard / "folds.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    sys.path.insert(0, str(repo))  # joblib models reference these v0.5 modules.
    import numpy as np
    import pandas as pd
    from arsh_v05 import build_returns, load_minutes, sequence_starts
    from runtime import RegimeRuntime

    config = json.loads((shard / "run_config.json").read_text(encoding="utf-8"))
    if config["policy"] != EXPECTED_POLICY or config["return_variant"] != "raw" \
            or config["non_overlapping"] is not True or config["allow_one_internal_missing"] is not False:
        raise ValueError("Shard configuration does not match the v0.6 engineering replay contract")
    with (shard / "folds.csv").open(encoding="utf-8-sig", newline="") as stream:
        folds = list(csv.DictReader(stream))
    last_fold = max(folds, key=lambda row: int(row["fold"]))
    family = last_fold["champion_family"]
    if (int(last_fold["champion_k"]), int(last_fold["champion_horizon"])) != (EXPECTED_K, EXPECTED_HORIZON) \
            or family not in HMM_FAMILIES:
        raise ValueError("Latest fold is not a K=7, 1-minute HMM candidate")

    artifact = shard / "runtime_research_candidate.joblib"
    manifest = json.loads((shard / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifact_hash = sha256(artifact)
    if artifact_hash != manifest[artifact.name]:
        raise ValueError("Model artifact hash differs from the shard manifest")
    runtime = RegimeRuntime.load(artifact)
    if runtime.model.n_components != EXPECTED_K or runtime.policy != EXPECTED_POLICY:
        raise ValueError("Loaded model has the wrong number of states or sequence policy")
    runtime.reset()  # Saved runtime includes an old posterior; replay starts afresh.

    raw, data_audit = load_minutes(data)
    day = pd.Timestamp(args.date).normalize() if args.date else raw.index.max().normalize()
    if pd.isna(day):
        raise ValueError("Invalid replay date")
    fitted_until = pd.Timestamp(last_fold["validation_end_exclusive"]).normalize()
    if day < fitted_until:
        raise ValueError(f"Date {day.date()} precedes this model's fit cutoff {fitted_until.date()}")
    day_raw = raw.loc[raw.index.normalize() == day].copy()
    if day_raw.empty:
        raise ValueError(f"No VN30F1M candles found on {day.date()}")
    day_hash = hashlib.sha256(day_raw.to_csv(index=True).encode("utf-8")).hexdigest()

    frames, _, rejected, _ = build_returns(day_raw, horizons=(EXPECTED_HORIZON,), overlapping=False,
                                           allow_one_internal_missing=False)
    observations = frames[EXPECTED_HORIZON]
    starts = sequence_starts(observations, EXPECTED_POLICY)
    if len(observations) == 0:
        raise ValueError(f"No valid return windows on {day.date()}")
    detailed = []
    for index, (timestamp, row) in enumerate(observations.iterrows()):
        if starts[index] and index > 0:
            runtime.reset()  # A rejected window is a hard boundary inside the day.
        result = runtime.predict_one(float(row["log_return"]), timestamp)
        posterior = np.asarray(result["posterior"], dtype=float)
        if len(posterior) != EXPECTED_K or not np.isfinite(posterior).all() \
                or (posterior < 0).any() or not math.isclose(float(posterior.sum()), 1.0, abs_tol=1e-9):
            raise ValueError(f"Invalid K=7 posterior at {timestamp}")
        detailed.append({
            "date": str(day.date()), "timestamp": str(timestamp),
            "window_start": str(row["window_start"]), "log_return": float(row["log_return"]),
            "session": row["session"], "sequence_reset": bool(starts[index]),
            "internal_state": result["internal_state"],
            "state_id": result["stable_state_id"], "confidence": result["confidence"],
            **{f"p_state_{state}": float(posterior[state]) for state in range(EXPECTED_K)},
        })

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    version_tag = f"{day_hash[:12]}_{artifact_hash[:12]}"
    detail_path = out / f"states_{day.date()}_{version_tag}.csv"
    with detail_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detailed[0]))
        writer.writeheader()
        writer.writerows(detailed)

    final = detailed[-1]
    status = "engineering_replay; source_day_completeness_unverified"
    convention = "v0.5 historical CSV convention; provider semantics unverified"
    source_meta = None
    if args.source_meta:
        source_meta = json.loads(args.source_meta.read_text(encoding="utf-8"))
        if source_meta["date"] != str(day.date()):
            raise ValueError("Source sidecar date differs from the replayed day")
        status = "replay; source_day_complete" if source_meta["complete"] else "replay; source_day_INCOMPLETE"
        convention = source_meta["timestamp_convention"]
    report = {
        "status": status,
        "date": str(day.date()),
        "symbol": "VN30F1M",
        "source_csv": str(data),
        "source_sha256": data_audit["source_sha256"],
        "source_day_sha256": day_hash,
        "source_timestamp_convention": convention,
        "source_day_version": source_meta["version"] if source_meta else None,
        "source_available_at": source_meta["available_at"] if source_meta else None,
        "data_warnings": source_meta["warnings"] if source_meta else [],
        "model_artifact": str(artifact),
        "model_sha256": artifact_hash,
        "model_version": runtime.model_version,
        "model_fit_cutoff_exclusive": str(fitted_until.date()),
        "policy": runtime.policy,
        "horizon_min": EXPECTED_HORIZON,
        "family": family,
        "k": EXPECTED_K,
        "observations": len(detailed),
        "rejected_windows": int(len(rejected)),
        "first_observation": detailed[0]["timestamp"],
        "last_observation": final["timestamp"],
        "final_posterior": [final[f"p_state_{state}"] for state in range(EXPECTED_K)],
        "final_internal_state": final["internal_state"],
        "final_state_id": final["state_id"],
        "final_confidence": final["confidence"],
        "detail_csv": str(detail_path),
        "note": "Descriptive state output only; no orders or parameter learning.",
    }
    report_path = out / f"report_{day.date()}_{version_tag}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Processed {len(detailed)} observations for {day.date()}")
    print(f"Detailed states: {detail_path}")
    print(f"End-of-day report: {report_path}")


if __name__ == "__main__":
    main()
