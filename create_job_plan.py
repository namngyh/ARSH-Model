"""Create reproducible CPU/CUDA fold shards for the complete v0.5 protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import arsh_v05 as arsh


def fold_ids(data,train_years=3,expanding=False):
    raw,_=arsh.load_minutes(data)
    # Fold dates depend on coverage, not horizon; one observation per source day
    # is sufficient and avoids rebuilding every return series while planning.
    reference=pd.DataFrame(index=pd.DatetimeIndex(sorted(raw.index.normalize().unique())))
    return [f.fold for f in arsh.make_folds(reference,train_years,expanding=expanding)]


def experiments(policy, phase):
    # One very long continuous sequence is faster in the compiled CPU path;
    # daily/session policies expose many short equal-length sequences that the
    # CUDA backend can batch efficiently.
    backend=lambda value:"cpu" if value=="continuous_carry" else "cuda"
    common={"horizons":[1,2],"k_values":[2,3,4,5,6,7],"policy":policy,
            "backend":backend(policy)}
    policy_jobs=[
        {"id":"policy_continuous","horizons":[1,2],"k_values":[2,3,4,5,6,7],
         "policy":"continuous_carry","backend":"cpu"},
        {"id":"policy_daily","horizons":[1,2],"k_values":[2,3,4,5,6,7],
         "policy":"daily_sequence","backend":"cuda"},
        {"id":"policy_session","horizons":[1,2],"k_values":[2,3,4,5,6,7],
         "policy":"session_sequence","backend":"cuda"},
    ]
    post_policy_jobs=[
        {"id":"main_revised","horizons":list(arsh.HORIZONS),"k_values":[2,3,4,5,6,7],"policy":policy},
        {"id":"k_boundary","horizons":[1,2],"k_values":[2,3,4,5,6,7,8,9],"policy":policy},
        {"id":"intraday_adjusted",**common,"return_variant":"intraday_adjusted"},
        {"id":"overlapping",**common,"overlapping":True},
        {"id":"train1y",**common,"train_years":1},
        {"id":"train2y",**common,"train_years":2},
        {"id":"expanding",**common,"expanding_window":True},
        {"id":"exclude_zero",**common,"exclude_zero":True},
        {"id":"exclude_train_5sigma",**common,"outlier_policy":"exclude_train_5sigma"},
        {"id":"exclude_rollover_adjacent",**common,"exclude_rollover_adjacent":True},
        {"id":"allow_one_internal_missing",**common,"allow_one_internal_missing":True},
    ]
    return policy_jobs if phase=="policy" else post_policy_jobs if phase=="post_policy" else policy_jobs+post_policy_jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=arsh.DEFAULT_DATA)
    parser.add_argument("--output",type=Path,default=Path("v05_cuda_job_plan.json"))
    parser.add_argument("--runs-root",default="runs_v05_revised")
    parser.add_argument("--policy",choices=arsh.POLICIES,default="continuous_carry",
                        help="Provisional policy for main/robustness; regenerate after policy decision")
    parser.add_argument("--phase",choices=("policy","post_policy","all"),default="policy")
    parser.add_argument("--folds-per-shard",type=int,default=2)
    args=parser.parse_args()
    if args.folds_per_shard<1:parser.error("folds-per-shard must be positive")
    cache={};jobs=[];items=[]
    for experiment in experiments(args.policy,args.phase):
        years=int(experiment.get("train_years",3));expanding=bool(experiment.get("expanding_window",False))
        key=(years,expanding)
        if key not in cache:cache[key]=fold_ids(args.data,years,expanding)
        ids=cache[key]
        record={**experiment,"expected_folds":ids,"shards":[]}
        for offset in range(0,len(ids),args.folds_per_shard):
            selected=ids[offset:offset+args.folds_per_shard]
            job_id=f"{experiment['id']}__f{selected[0]:02d}_{selected[-1]:02d}"
            # POSIX separators remain valid on Windows and keep the same plan
            # usable by the CUDA workstation and Linux-based Colab coordinator.
            output=(Path(args.runs_root)/experiment["id"]/job_id).as_posix()
            job={"job_id":job_id,"experiment_id":experiment["id"],"output":output,
                 "folds":selected,"execution_target":experiment["backend"],
                 **{k:v for k,v in experiment.items() if k!="id"}}
            jobs.append(job);record["shards"].append(job_id)
        items.append(record)
    plan={"schema":1,"data_sha256":arsh.sha256(args.data),"code_sha256":arsh.sha256(Path(arsh.__file__)),
          "runs_root":args.runs_root,"experiments":items,"jobs":jobs}
    args.output.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding="utf8")
    cpu=sum(job["execution_target"]=="cpu" for job in jobs)
    cuda=len(jobs)-cpu
    print(f"Wrote {args.output.resolve()}: {len(items)} experiments, {cpu} CPU + {cuda} CUDA shards")


if __name__=="__main__":main()
