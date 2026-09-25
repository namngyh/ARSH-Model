"""Inventory v0.5 policy results for v0.6 without refitting or selecting on test."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "ARSH_v0.5_PAUSED_2026-09-24" / "runs_v05_revised"
OUTPUT = HERE / "outputs" / "starting_point"
POLICIES = ("policy_daily", "policy_session")
HMM_FAMILIES = {"gaussian_hmm", "student_t_shared", "student_t_state"}
EXPECTED_FOLDS = set(range(10))
EXPECTED_DATA_SHA = "bf84b23d6fa48b9fd90c477ef6788cc0aca19e86a0480d91600fa77363e71096"
EXPECTED_CODE_SHA = "b5e19665d071b0e8097f35db56ee54008ed193b4c0e8dc80219be64d00c60efc"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def eligible(row: dict[str, str]) -> bool:
    return row["eligible_1pct"].strip().lower() == "true"


def main() -> None:
    selected: list[dict] = []
    champions: list[dict] = []
    profiles: list[dict] = []
    coverage: dict[str, list[int]] = {}

    for policy_dir in POLICIES:
        folder = RUNS / policy_dir
        shards = sorted(path for path in folder.iterdir() if path.is_dir())
        if len(shards) != 5:
            raise ValueError(f"{policy_dir}: expected 5 completed shards, found {len(shards)}")
        fold_ids: set[int] = set()

        for shard in shards:
            progress = read_json(shard / "progress.json")
            audit = read_json(shard / "data_audit.json")
            manifest = read_json(shard / "artifact_manifest.json")
            config = read_json(shard / "run_config.json")
            if not progress.get("complete"):
                raise ValueError(f"Incomplete shard: {shard}")
            if audit.get("source_sha256") != EXPECTED_DATA_SHA:
                raise ValueError(f"Data hash mismatch: {shard}")
            if manifest.get("arsh_v05.py") != EXPECTED_CODE_SHA:
                raise ValueError(f"Code hash mismatch: {shard}")
            expected_policy = "daily_sequence" if policy_dir == "policy_daily" else "session_sequence"
            if config.get("policy") != expected_policy:
                raise ValueError(f"Unexpected policy: {shard}")

            candidates = read_csv(shard / "validation_candidates.csv")
            by_fold_horizon: dict[tuple[int, int], list[dict]] = defaultdict(list)
            for row in candidates:
                if row["family"] in HMM_FAMILIES and eligible(row):
                    key = (int(row["fold"]), int(row["horizon_min"]))
                    by_fold_horizon[key].append(row)
            for (fold, horizon), rows in by_fold_horizon.items():
                best = max(rows, key=lambda row: (float(row["validation_log_density"]), -int(row["k"])))
                selected.append({
                    "policy": config["policy"],
                    "fold": fold,
                    "horizon_min": horizon,
                    "family": best["family"],
                    "k": int(best["k"]),
                    "validation_gain_vs_student": float(best["validation_gain_vs_student"]),
                    "validation_min_soft_share": float(best["validation_min_soft_share"]),
                    "shard": shard.name,
                })

            all_profiles = read_csv(shard / "state_profiles_aligned.csv")
            for fold_row in read_csv(shard / "folds.csv"):
                fold = int(fold_row["fold"])
                if fold in fold_ids:
                    raise ValueError(f"Duplicate fold {fold} in {policy_dir}")
                fold_ids.add(fold)
                horizon = int(fold_row["champion_horizon"])
                family = fold_row["champion_family"]
                k = int(fold_row["champion_k"])
                matching = [row for row in candidates if int(row["fold"]) == fold
                            and int(row["horizon_min"]) == horizon
                            and row["family"] == family and int(row["k"]) == k]
                if len(matching) != 1 or not eligible(matching[0]):
                    raise ValueError(f"Champion missing or ineligible: {shard}, fold {fold}")
                champions.append({
                    "policy": config["policy"],
                    "fold": fold,
                    "horizon_min": horizon,
                    "family": family,
                    "k": k,
                    "validation_gain_vs_student": float(matching[0]["validation_gain_vs_student"]),
                    "validation_min_soft_share": float(matching[0]["validation_min_soft_share"]),
                    "shard": shard.name,
                })

                state_rows = [row for row in all_profiles if int(row["fold"]) == fold
                              and int(row["horizon_min"]) == horizon
                              and row["family"] == family and int(row["k"]) == k]
                if len(state_rows) != k:
                    raise ValueError(f"Expected {k} state profiles: {shard}, fold {fold}; got {len(state_rows)}")
                for row in state_rows:
                    profiles.append({
                        "policy": config["policy"], "fold": fold,
                        "horizon_min": horizon, "family": family, "k": k,
                        "display_state": row["display_state"],
                        "model_mean_return_pct": row["model_mean_return_pct"],
                        "model_std_return_pct": row["model_std_return_pct"],
                        "model_df": row["model_df"],
                        "self_transition": row["self_transition"],
                        "test_share_hard": row["test_share"],
                        "mean_confidence_hard_group": row["mean_confidence"],
                        "mean_run_bars_hard": row["mean_run_bars"],
                        "shard": shard.name,
                    })

        if fold_ids != EXPECTED_FOLDS:
            raise ValueError(f"{policy_dir}: fold coverage {sorted(fold_ids)}")
        coverage[policy_dir] = sorted(fold_ids)

    selected.sort(key=lambda row: (row["policy"], row["fold"], row["horizon_min"]))
    champions.sort(key=lambda row: (row["policy"], row["fold"]))
    profiles.sort(key=lambda row: (row["policy"], row["fold"], int(row["display_state"])))

    summary = {}
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_policy[row["policy"]].append(row)
    for policy, rows in by_policy.items():
        own_champions = [row for row in champions if row["policy"] == policy]
        summary[policy] = {
            "folds": len(own_champions),
            "mean_validation_gain_1m_2m": sum(row["validation_gain_vs_student"] for row in rows) / len(rows),
            "champion_horizons": dict(Counter(str(row["horizon_min"]) for row in own_champions)),
            "champion_families": dict(Counter(row["family"] for row in own_champions)),
            "champion_k": dict(Counter(str(row["k"]) for row in own_champions)),
        }
    daily = {(row["fold"], row["horizon_min"]): row["validation_gain_vs_student"]
             for row in selected if row["policy"] == "daily_sequence"}
    session = {(row["fold"], row["horizon_min"]): row["validation_gain_vs_student"]
               for row in selected if row["policy"] == "session_sequence"}
    if daily.keys() != session.keys() or len(daily) != 20:
        raise ValueError("Daily/session validation pairs do not cover 10 folds × 2 horizons")
    differences = [daily[key] - session[key] for key in sorted(daily)]

    report = {
        "source": "v0.5 revised policy shards; historical research benchmark",
        "data_sha256": EXPECTED_DATA_SHA,
        "code_sha256": EXPECTED_CODE_SHA,
        "coverage": coverage,
        "policy_summary": summary,
        "daily_minus_session_mean_validation_gain": sum(differences) / len(differences),
        "daily_better_pairs": sum(value > 0 for value in differences),
        "paired_fold_horizon_count": len(differences),
        "continuous_carry_status": "missing completed fold results",
        "selection_status": "K=7 fixed for v0.6 by project decision; policy, horizon, HMM family and model parameters remain under evaluation",
        "state_profile_note": "Test profiles are descriptive hard-label summaries; state IDs are not aligned across shards.",
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    chosen_fields = list(champions[0])
    write_csv(OUTPUT / "validation_best_per_fold_horizon.csv", selected, chosen_fields)
    write_csv(OUTPUT / "fold_champions.csv", champions, chosen_fields)
    write_csv(OUTPUT / "champion_state_profiles.csv", profiles, list(profiles[0]))
    (OUTPUT / "starting_point.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT}: {len(champions)} fold champions, {len(profiles)} state profiles")


if __name__ == "__main__":
    main()
