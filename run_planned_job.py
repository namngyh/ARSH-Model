"""Run one immutable CPU or CUDA shard from a generated v0.5 job plan."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import arsh_v05 as arsh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--job",required=True)
    parser.add_argument("--data",type=Path,default=arsh.DEFAULT_DATA)
    parser.add_argument("--time-budget-minutes",type=float)
    args=parser.parse_args();plan=json.loads(args.plan.read_text(encoding="utf8"))
    if arsh.sha256(args.data)!=plan["data_sha256"]:parser.error("data SHA-256 differs from plan")
    if arsh.sha256(Path(arsh.__file__))!=plan["code_sha256"]:parser.error("code SHA-256 differs from plan")
    matches=[item for item in plan["jobs"] if item["job_id"]==args.job]
    if len(matches)!=1:parser.error(f"unknown/duplicate job ID: {args.job}")
    job=matches[0]
    backend=job.get("backend",job.get("execution_target","cuda"))
    command=[sys.executable,str(Path(arsh.__file__)),"--data",str(args.data),"--output",job["output"],
             "--backend",backend,"--device","cuda:0","--policy",job["policy"],"--resume",
             "--horizons",*map(str,job["horizons"]),"--k-values",*map(str,job["k_values"]),
             "--folds",*map(str,job["folds"]),"--train-years",str(job.get("train_years",3)),
             "--return-variant",job.get("return_variant","raw"),
             "--outlier-policy",job.get("outlier_policy","keep")]
    for key,flag in (("overlapping","--overlapping"),("expanding_window","--expanding-window"),
                     ("exclude_zero","--exclude-zero"),
                     ("exclude_rollover_adjacent","--exclude-rollover-adjacent"),
                     ("allow_one_internal_missing","--allow-one-internal-missing")):
        if job.get(key):command.append(flag)
    if args.time_budget_minutes:command += ["--time-budget-minutes",str(args.time_budget_minutes)]
    print(f"Running immutable {backend.upper()} job {args.job}",flush=True);subprocess.run(command,check=True)


if __name__=="__main__":main()
