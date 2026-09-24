"""Out-of-sample Gaussian, Student-t and Jones-Faddy skew-t diagnostics for ARSH v0.5."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import arsh_v05 as arsh


def fit_score(train,score):
    scaler=StandardScaler().fit(train[["log_return"]])
    x=scaler.transform(train[["log_return"]]).ravel()
    y=scaler.transform(score[["log_return"]]).ravel()
    jac=float(np.log(scaler.scale_[0]));rows=[]
    candidates={
        "gaussian":lambda:(stats.norm.fit(x),stats.norm.logpdf),
        "student_t":lambda:(stats.t.fit(x),stats.t.logpdf),
        "skew_t_jones_faddy":lambda:(stats.jf_skew_t.fit(x),stats.jf_skew_t.logpdf),
    }
    for family,builder in candidates.items():
        try:
            parameters,logpdf=builder();value=float(np.mean(logpdf(y,*parameters))-jac)
            rows.append({"family":family,"log_density":value,"parameters":json.dumps([float(z) for z in parameters])})
        except Exception as exc:
            rows.append({"family":family,"log_density":np.nan,"error":f"{type(exc).__name__}: {exc}"})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=arsh.DEFAULT_DATA)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"distribution_outputs")
    parser.add_argument("--horizons",type=int,nargs="+",default=list(arsh.HORIZONS))
    parser.add_argument("--max-folds",type=int)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raw,audit=arsh.load_minutes(args.data);frames,_,_,_=arsh.build_returns(raw,args.horizons)
    reference=60 if 60 in frames else max(frames);folds=arsh.make_folds(frames[reference])
    if args.max_folds:folds=folds[:args.max_folds]
    rows=[]
    for fold in folds:
        for horizon in args.horizons:
            train,validation,test=arsh.slice_fold(frames[horizon],fold)
            for stage,left,right in (("validation",train,validation),
                                     ("test",pd.concat([train,validation]).sort_index(),test)):
                for result in fit_score(left,right):
                    rows.append({"fold":fold.fold,"horizon_min":horizon,"stage":stage,
                                 "train_observations":len(left),"score_observations":len(right),
                                 "train_skew":float(left.log_return.skew()),**result})
            print(f"fold={fold.fold} horizon={horizon}m done",flush=True)
    table=pd.DataFrame(rows);table["fit_success"]=table.log_density.notna()
    table.to_csv(args.output/"distribution_fold_scores.csv",index=False,encoding="utf-8-sig")
    summary=(table.groupby(["stage","horizon_min","family"],as_index=False)
             .agg(mean_log_density=("log_density","mean"),fits=("fit_success","size"),
                  successful_fits=("fit_success","sum"),success_rate=("fit_success","mean"),
                  folds=("fold","nunique")))
    summary.to_csv(args.output/"distribution_summary.csv",index=False,encoding="utf-8-sig")
    wide=table.pivot_table(index=["fold","horizon_min","stage"],columns="family",
                           values="log_density",aggfunc="first").reset_index()
    paired=[]
    if "student_t" in wide:
        for challenger in ("gaussian","skew_t_jones_faddy"):
            if challenger not in wide:continue
            valid=wide[[challenger,"student_t"]].notna().all(axis=1)
            for row in wide.loc[valid].itertuples(index=False):
                paired.append({"fold":row.fold,"horizon_min":row.horizon_min,"stage":row.stage,
                               "challenger":challenger,
                               "gain_vs_student":float(getattr(row,challenger)-row.student_t)})
    pairwise=pd.DataFrame(paired)
    pairwise.to_csv(args.output/"distribution_pairwise.csv",index=False,encoding="utf-8-sig")
    skew=table[table.family.eq("skew_t_jones_faddy")]
    skew_success=float(skew.fit_success.mean()) if len(skew) else 0.0
    skew_test=pairwise[(pairwise.stage=="test") &
                       (pairwise.challenger=="skew_t_jones_faddy")] if len(pairwise) else pairwise
    mean_gain=float(skew_test.gain_vs_student.mean()) if len(skew_test) else float("nan")
    positive_rate=float((skew_test.gain_vs_student>0).mean()) if len(skew_test) else 0.0
    retain=bool(skew_success>=.95 and len(skew_test)>0 and mean_gain>0 and positive_rate>=.70)
    decision={"diagnostic_complete":bool(len(table)>0),"skew_t_success_rate":skew_success,
              "skew_t_test_comparisons":int(len(skew_test)),
              "skew_t_mean_gain_vs_student":mean_gain,
              "skew_t_positive_comparison_rate":positive_rate,
              "retain_skew_t_challenger":retain,
              "rule":"Retain only with >=95% fit success, positive mean test gain, and positive gain in >=70% of fold-horizon comparisons."}
    (args.output/"distribution_decision.json").write_text(
        json.dumps(decision,indent=2,ensure_ascii=False),encoding="utf8")
    (args.output/"progress.json").write_text(json.dumps({
        "complete":True,"folds":len(folds),"horizons":list(args.horizons),
        "rows":len(table)},indent=2),encoding="utf8")
    (args.output/"data_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(decision,indent=2),flush=True)


if __name__=="__main__":main()
