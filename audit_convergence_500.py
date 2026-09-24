"""Targeted 500-iteration CUDA audit for selected HMMs from a revised main run."""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import arsh_v05 as arsh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data",type=Path,default=arsh.DEFAULT_DATA)
    parser.add_argument("--folds",type=int,nargs="+")
    parser.add_argument("--backend",choices=("auto",)+arsh.BACKENDS,default="auto")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--iterations",type=int,default=500)
    parser.add_argument("--restarts",type=int,default=5)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    audit=json.loads((args.reference_output/"data_audit.json").read_text(encoding="utf8"))
    if arsh.sha256(args.data)!=audit["source_sha256"]:parser.error("data hash differs from reference")
    config=json.loads((args.reference_output/"run_config.json").read_text(encoding="utf8"))
    if args.backend=="auto":
        args.backend="cpu" if config["policy"]=="continuous_carry" else "cuda"
    decision=json.loads((args.reference_output/"horizon_decision.json").read_text(encoding="utf8"))
    models=joblib.load(args.reference_output/"selected_fold_models.joblib")
    selected_folds=set(args.folds if args.folds is not None else models.keys())
    unknown=selected_folds-set(models)
    if unknown:parser.error(f"unknown folds: {sorted(unknown)}")
    horizon=int(decision["preferred_research_horizon"])
    identity={"schema":1,"source_sha256":audit["source_sha256"],
              "reference_models_sha256":arsh.sha256(args.reference_output/"selected_fold_models.joblib"),
              "reference_code_sha256":config.get("code_sha256"),
              "audit_code_sha256":arsh.sha256(Path(__file__)),"horizon":horizon,
              "policy":config["policy"],"iterations":args.iterations,"restarts":args.restarts,
              "backend":args.backend,"device":args.device,"selected_folds":sorted(selected_folds)}
    (args.output/"audit_identity.json").write_text(
        json.dumps(identity,indent=2,ensure_ascii=False),encoding="utf8")
    raw,_=arsh.load_minutes(args.data);frames=arsh.build_returns(
        raw,(horizon,),overlapping=not config["non_overlapping"],
        allow_one_internal_missing=config.get("allow_one_internal_missing",False))[0]
    rows=[];audited={}
    for fold in sorted(selected_folds):
        item=models[fold];bounds={field.name:pd.Timestamp(item["fold_bounds"][field.name])
                                 if field.name!="fold" else item["fold_bounds"][field.name]
                                 for field in fields(arsh.Fold)}
        train,val,test=arsh.slice_fold(frames[horizon],arsh.Fold(**bounds))
        (train,val,test),_=arsh.robust_filter_split(
            train,val,test,config.get("exclude_zero",False),config.get("outlier_policy","keep"),
            config.get("exclude_rollover_adjacent",False))
        if config["return_variant"]=="intraday_adjusted":
            train,val,test=arsh.intraday_adjust_split(train,val,test)
        development=pd.concat([train,val]).sort_index();scaler=StandardScaler().fit(development[["log_return"]])
        x=scaler.transform(development[["log_return"]]);lengths=arsh.sequence_lengths(development,config["policy"])
        selected=item["selection"][horizon];audited[fold]={}
        family_k={"gaussian_hmm":selected["gaussian_hmm_k"]}
        family_k.update({family:selected.get(f"{family}_k") for family in arsh.STUDENT_FAMILIES})
        for family,k in family_k.items():
            if k is None:continue
            model,diag=arsh.fit_hmm_restarts(
                x,family,int(k),args.iterations,args.restarts,542,horizon,"convergence500",
                args.output/"_checkpoints"/f"fold_{fold:02d}"/"fits",lengths,args.backend,args.device)
            audited[fold][family]=model
            for row in diag:row["fold"]=fold
            rows.extend(diag)
        (args.output/"progress.json").write_text(json.dumps({
            "completed_folds":len(audited),"requested_folds":sorted(selected_folds),
            "complete":set(audited)==selected_folds},indent=2),encoding="utf8")
        joblib.dump(audited,args.output/"convergence500_models.joblib")
        pd.DataFrame(rows).to_csv(args.output/"convergence500_diagnostics.csv",index=False,encoding="utf-8-sig")
    summary=(pd.DataFrame(rows).dropna(subset=["train_ll"]).groupby("family",as_index=False)
             .agg(fits=("seed","size"),strict_rate=("strict_converged","mean"),
                  practical_rate=("converged","mean"),median_iterations=("iterations","median")))
    summary.to_csv(args.output/"convergence500_summary.csv",index=False,encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__=="__main__":main()
