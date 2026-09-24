"""Audit the 2025-05-05 timestamp regime change and its return-window impact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import arsh_v05 as arsh


CUTOVER=pd.Timestamp("2025-05-05")


def regime(values):
    dates=pd.to_datetime(values)
    dates=dates.dt.normalize() if isinstance(dates,pd.Series) else dates.normalize()
    return np.where(dates<CUTOVER,"before_2025_05_05","from_2025_05_05")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=arsh.DEFAULT_DATA)
    parser.add_argument("--output",type=Path,default=Path("timestamp_regime_outputs"))
    parser.add_argument("--main-output",type=Path,
                        help="Optional finalized main run for pre/post predictive diagnostics")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raw,audit=arsh.load_minutes(args.data)
    daily,structure=arsh.source_structure_audit(raw)
    frames,horizon_audit,rejected,_=arsh.build_returns(raw,arsh.HORIZONS)
    rows=[]
    for horizon,frame in frames.items():
        work=frame.copy();work["timestamp_regime"]=regime(work["date"])
        for name,group in work.groupby("timestamp_regime"):
            per_day=group.groupby("date").size()
            x=group.log_return
            rows.append({"timestamp_regime":name,"horizon_min":horizon,
                         "days":int(per_day.size),"accepted_returns":int(len(group)),
                         "mean_accepted_per_day":float(per_day.mean()),
                         "median_accepted_per_day":float(per_day.median()),
                         "zero_rate":float(x.eq(0).mean()),"mean_return":float(x.mean()),
                         "return_std":float(x.std(ddof=1)),"excess_kurtosis":float(x.kurt())})
    summary=pd.DataFrame(rows)
    rejected=rejected.copy()
    rejected["timestamp_regime"]=regime(rejected["date"])
    rejected_summary=(rejected.groupby(["timestamp_regime","horizon_min"],as_index=False)
                      .agg(rejected_windows=("reason","size")))
    summary=summary.merge(rejected_summary,on=["timestamp_regime","horizon_min"],how="left")
    summary["rejected_windows"]=summary.rejected_windows.fillna(0).astype(int)
    summary["acceptance_rate"]=summary.accepted_returns/(summary.accepted_returns+summary.rejected_windows)
    summary.to_csv(args.output/"timestamp_regime_return_windows.csv",index=False,encoding="utf-8-sig")
    daily.to_csv(args.output/"timestamp_structure_daily.csv",index=False,encoding="utf-8-sig")
    predictive_rows=0
    if args.main_output:
        prediction_path=args.main_output/"test_predictions_long.csv"
        if not prediction_path.is_file():parser.error(f"missing {prediction_path}")
        predictions=pd.read_csv(prediction_path,parse_dates=["datetime"])
        predictions["timestamp_regime"]=regime(predictions.datetime)
        predictive=(predictions.groupby(["timestamp_regime","horizon_min","family"],as_index=False)
                    .agg(observations=("log_density","size"),mean_log_density=("log_density","mean"),
                         mean_gain_vs_student=("gain_vs_student","mean")))
        predictive_rows=len(predictive)
        predictive.to_csv(args.output/"timestamp_regime_predictive_scores.csv",index=False,
                          encoding="utf-8-sig")
    decision={"complete":True,"cutover":"2025-05-05","source_sha256":audit["source_sha256"],
              "modal_rows_before":structure["modal_rows_per_day_before_2025_05_05"],
              "modal_rows_after":structure["modal_rows_per_day_from_2025_05_05"],
              "return_window_rows":int(len(summary)),"predictive_rows":int(predictive_rows),
              "interpretation":"The table measures association with the timestamp-format regime; it is not a causal KRX effect estimate and cannot prove the provider's bar convention."}
    (args.output/"timestamp_regime_decision.json").write_text(
        json.dumps(decision,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(decision,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
