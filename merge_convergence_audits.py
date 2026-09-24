"""Merge independent 500-iteration convergence-audit fold shards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def common(identity):
    return {key:value for key,value in identity.items() if key!="selected_folds"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-folds",type=int,nargs="+",required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error("output must be empty")
    identities=[];models={};diagnostics=[]
    for shard in args.shards:
        identity_path=shard/"audit_identity.json";progress_path=shard/"progress.json"
        if not identity_path.is_file() or not progress_path.is_file():
            parser.error(f"missing identity/progress in {shard}")
        progress=json.loads(progress_path.read_text(encoding="utf8"))
        if not progress.get("complete"):parser.error(f"incomplete audit shard: {shard}")
        identity=json.loads(identity_path.read_text(encoding="utf8"));identities.append(identity)
        shard_models=joblib.load(shard/"convergence500_models.joblib")
        for fold,item in shard_models.items():
            fold=int(fold)
            if fold in models:parser.error(f"duplicate fold {fold}")
            models[fold]=item
        diagnostics.append(pd.read_csv(shard/"convergence500_diagnostics.csv"))
    reference=common(identities[0])
    if any(common(item)!=reference for item in identities[1:]):
        parser.error("audit identities differ")
    expected=set(args.expected_folds);found=set(models)
    if found!=expected:parser.error(f"fold mismatch: missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    args.output.mkdir(parents=True,exist_ok=True)
    table=pd.concat(diagnostics,ignore_index=True)
    summary=(table.dropna(subset=["train_ll"]).groupby("family",as_index=False)
             .agg(fits=("seed","size"),strict_rate=("strict_converged","mean"),
                  practical_rate=("converged","mean"),median_iterations=("iterations","median")))
    joblib.dump(models,args.output/"convergence500_models.joblib")
    table.to_csv(args.output/"convergence500_diagnostics.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(args.output/"convergence500_summary.csv",index=False,encoding="utf-8-sig")
    identity={**reference,"selected_folds":sorted(found)}
    (args.output/"audit_identity.json").write_text(json.dumps(identity,indent=2),encoding="utf8")
    (args.output/"progress.json").write_text(json.dumps({
        "complete":True,"completed_folds":len(found),"requested_folds":sorted(expected),
        "source_shards":[str(path.resolve()) for path in args.shards]},indent=2),encoding="utf8")
    print(summary.to_string(index=False))


if __name__=="__main__":main()
