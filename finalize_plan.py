"""CPU coordinator: merge and finalize completed experiments from a CUDA plan."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--experiments",nargs="+")
    args=parser.parse_args();plan=json.loads(args.plan.read_text(encoding="utf8"))
    root=Path(plan["runs_root"]);wanted=set(args.experiments or [])
    here=Path(__file__).resolve().parent
    for experiment in plan["experiments"]:
        name=experiment["id"]
        if wanted and name not in wanted:continue
        merged=root/name/"merged"
        if (merged/"progress.json").is_file() and json.loads((merged/"progress.json").read_text())["complete"]:
            print(f"Skip completed merged experiment: {name}");continue
        shards=[root/name/job for job in experiment["shards"]]
        missing=[str(path) for path in shards if not (path/"progress.json").is_file()]
        if missing:raise SystemExit(f"{name}: missing shard outputs: {missing}")
        command=[sys.executable,str(here/"merge_shards.py"),*map(str,shards),"--output",str(merged),
                 "--expected-folds",*map(str,experiment["expected_folds"])]
        subprocess.run(command,check=True)
        subprocess.run([sys.executable,str(here/"finalize_experiment.py"),
                        "--output",str(merged),"--data",str(args.data)],check=True)


if __name__=="__main__":main()
