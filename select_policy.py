"""Choose the sequence policy from completed policy experiments, validation first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders",type=Path,nargs=3)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();rows=[]
    for folder in args.folders:
        progress=json.loads((folder/"progress.json").read_text(encoding="utf8"))
        if not progress["complete"]:parser.error(f"incomplete policy experiment: {folder}")
        config=json.loads((folder/"run_config.json").read_text(encoding="utf8"));policy=config["policy"]
        candidates=pd.read_csv(folder/"validation_candidates.csv")
        hmm=candidates[(candidates.family.isin(["gaussian_hmm","student_t_shared","student_t_state"]))
                       & candidates.eligible_1pct]
        # Selection remains entirely validation-based: best eligible HMM per fold/horizon.
        chosen=(hmm.sort_values(["fold","horizon_min","validation_log_density","k"],
                                ascending=[True,True,False,True])
                .groupby(["fold","horizon_min"],as_index=False).first())
        metrics=pd.read_csv(folder/"walk_forward_metrics.csv")
        test=(metrics[metrics.family.isin(["gaussian_hmm","student_t_shared","student_t_state"])]
              .groupby("horizon_min").log_density.mean())
        for horizon,group in chosen.groupby("horizon_min"):
            rows.append({"policy":policy,"horizon_min":int(horizon),
                         "validation_log_density":float(group.validation_log_density.mean()),
                         "validation_gain_vs_student":float(group.validation_gain_vs_student.mean()),
                         "mean_selected_k":float(group.k.mean()),"folds":int(group.fold.nunique()),
                         "test_log_density_diagnostic":float(test.get(horizon,float("nan")))})
    table=pd.DataFrame(rows);args.output.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output/"policy_comparison.csv",index=False,encoding="utf-8-sig")
    aggregate=(table.groupby("policy",as_index=False).validation_gain_vs_student.mean()
               .sort_values("validation_gain_vs_student",ascending=False))
    winner=str(aggregate.iloc[0].policy);runner=str(aggregate.iloc[1].policy)
    decision={"selection_basis":"mean validation gain versus same-horizon Student-t across 1m/2m",
              "preferred_policy":winner,"runner_up_policy":runner,
              "validation_margin":float(aggregate.iloc[0].validation_gain_vs_student-aggregate.iloc[1].validation_gain_vs_student),
              "test_metrics_role":"diagnostic only; not used for policy selection",
              "locked_for_post_policy_plan":True}
    (args.output/"policy_decision.json").write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(decision,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
