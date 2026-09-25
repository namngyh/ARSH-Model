"""Compare K on saved validation folds; never use test data for selection."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "ARSH_v0.5_PAUSED_2026-09-24" / "runs_v05_revised"
OUTPUT = HERE / "outputs" / "starting_point" / "k_validation_comparison.csv"
POLICIES = {"policy_daily": "daily_sequence", "policy_session": "session_sequence"}
HMM_FAMILIES = {"gaussian_hmm", "student_t_shared", "student_t_state"}
FOLDS = set(range(10))
K_VALUES = range(2, 8)
HORIZONS = (1, 2)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    output_rows = []
    for policy_folder, policy in POLICIES.items():
        candidates = []
        shards = sorted(path for path in (RUNS / policy_folder).iterdir() if path.is_dir())
        if len(shards) != 5:
            raise ValueError(f"{policy}: expected five completed shards")
        for shard in shards:
            candidates.extend(read_csv(shard / "validation_candidates.csv"))

        by_key: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
        for row in candidates:
            if row["family"] in HMM_FAMILIES:
                key = (int(row["fold"]), int(row["horizon_min"]), int(row["k"]))
                by_key[key].append(row)

        for horizon in HORIZONS:
            best_by_k: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
            for fold in FOLDS:
                for k in K_VALUES:
                    rows = by_key[fold, horizon, k]
                    if {row["family"] for row in rows} != HMM_FAMILIES or len(rows) != 3:
                        raise ValueError(f"Incomplete family coverage: {policy}, {fold}, {horizon}, {k}")
                    eligible = [row for row in rows if row["eligible_1pct"].lower() == "true"]
                    if eligible:
                        best_by_k[k][fold] = max(
                            eligible, key=lambda row: float(row["validation_log_density"])
                        )

            for k in K_VALUES:
                own = best_by_k[k]
                paired_folds = sorted(own.keys() & best_by_k[7].keys())
                gaps = [
                    float(best_by_k[7][fold]["validation_gain_vs_student"])
                    - float(own[fold]["validation_gain_vs_student"])
                    for fold in paired_folds
                ]
                output_rows.append({
                    "policy": policy,
                    "horizon_min": horizon,
                    "k": k,
                    "eligible_folds": len(own),
                    "mean_validation_gain_vs_student": mean(
                        float(row["validation_gain_vs_student"]) for row in own.values()
                    ) if own else "",
                    "paired_folds_vs_k7": len(paired_folds),
                    "mean_k7_minus_k_gain": mean(gaps) if gaps else "",
                    "median_k7_minus_k_gain": median(gaps) if gaps else "",
                    "folds_k7_better": sum(gap > 0 for gap in gaps),
                    "median_min_soft_share": median(
                        float(row["validation_min_soft_share"]) for row in own.values()
                    ) if own else "",
                })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
