"""Combine completed ARSH v0.5 experiments and evaluate completion gates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd


REQUIRED={"policy_continuous","policy_daily","policy_session","main_revised","k_boundary",
          "intraday_adjusted","overlapping","train1y","train2y","expanding","exclude_zero",
          "exclude_train_5sigma","exclude_rollover_adjacent","allow_one_internal_missing"}


def experiment_name(folder):
    return folder.parent.name if folder.name=="merged" else folder.name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--convergence-output",type=Path)
    parser.add_argument("--distribution-output",type=Path)
    parser.add_argument("--timestamp-output",type=Path)
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    tables=[];decisions=[];configs=[];seen=set();incomplete=[]
    for folder in args.folders:
        name=experiment_name(folder);seen.add(name)
        progress=folder/"progress.json"
        complete=progress.is_file() and json.loads(progress.read_text(encoding="utf8"))["complete"]
        if not complete:incomplete.append(name);continue
        metrics=folder/"walk_forward_metrics.csv"
        if not metrics.is_file():incomplete.append(name);continue
        table=pd.read_csv(metrics);table.insert(0,"experiment",name);tables.append(table)
        decision=folder/"horizon_decision.json"
        if decision.is_file():decisions.append({"experiment":name,**json.loads(decision.read_text(encoding="utf8"))})
        config=folder/"run_config.json"
        if config.is_file():configs.append({"experiment":name,**json.loads(config.read_text(encoding="utf8"))})
    missing=sorted(REQUIRED-seen)
    if args.require_complete and (missing or incomplete):
        raise SystemExit(f"incomplete v0.5 protocol: missing={missing}, incomplete={sorted(set(incomplete))}")
    if not tables:raise SystemExit("No completed experiment folders supplied")
    all_metrics=pd.concat(tables,ignore_index=True)
    summary=(all_metrics.groupby(["experiment","horizon_min","family"],as_index=False)
             .agg(mean_log_density=("log_density","mean"),mean_gain_vs_student=("gain_vs_student","mean"),
                  positive_fold_rate=("gain_vs_student",lambda x:float((x>0).mean())),folds=("fold","nunique")))
    decision_table=pd.DataFrame(decisions);config_table=pd.DataFrame(configs)
    summary.to_csv(args.output/"sensitivity_summary.csv",index=False,encoding="utf-8-sig")
    decision_table.to_csv(args.output/"sensitivity_decisions.csv",index=False,encoding="utf-8-sig")
    config_table.to_csv(args.output/"sensitivity_configs.csv",index=False,encoding="utf-8-sig")
    main=decision_table[decision_table.experiment.eq("main_revised")]
    gates={"required_experiments":sorted(REQUIRED),"missing_experiments":missing,
           "incomplete_experiments":sorted(set(incomplete)),"all_required_complete":not missing and not incomplete}
    convergence_ok=False;distribution_ok=False;timestamp_ok=False
    if args.convergence_output:
        progress=args.convergence_output/"progress.json";summary_path=args.convergence_output/"convergence500_summary.csv"
        if progress.is_file() and summary_path.is_file() and json.loads(progress.read_text(encoding="utf8")).get("complete"):
            convergence=pd.read_csv(summary_path)
            convergence_ok=bool(len(convergence) and convergence.practical_rate.min()>=.80)
            gates["convergence500_min_practical_rate"]=float(convergence.practical_rate.min())
    if args.distribution_output:
        path=args.distribution_output/"distribution_decision.json"
        if path.is_file():
            distribution=json.loads(path.read_text(encoding="utf8"))
            distribution_ok=bool(distribution.get("diagnostic_complete"))
            gates["retain_skew_t_challenger"]=bool(distribution.get("retain_skew_t_challenger"))
    if args.timestamp_output:
        path=args.timestamp_output/"timestamp_regime_decision.json"
        if path.is_file():timestamp_ok=bool(json.loads(path.read_text(encoding="utf8")).get("complete"))
    gates.update({"convergence500_gate":convergence_ok,"distribution_diagnostic_complete":distribution_ok,
                  "timestamp_regime_audit_complete":timestamp_ok})
    if args.require_complete and not (args.convergence_output and args.distribution_output and args.timestamp_output):
        raise SystemExit("--require-complete also requires convergence, distribution and timestamp output folders")
    if not main.empty:
        row=main.iloc[0];winner=int(row.preferred_research_horizon)
        excluded={"main_revised","k_boundary","policy_continuous","policy_daily","policy_session"}
        robust=decision_table[decision_table.experiment.isin(REQUIRED-excluded)]
        agreement=float((robust.preferred_research_horizon==winner).mean()) if len(robust) else 0.0
        boundary=decision_table[decision_table.experiment.eq("k_boundary")]
        boundary_rate=float(boundary.winner_k_upper_boundary_rate.iloc[0]) if len(boundary) else 1.0
        gates.update({"main_horizon":winner,"robustness_horizon_agreement_rate":agreement,
                      "k_2_9_upper_boundary_rate":boundary_rate,
                      "main_horizon_bootstrap_conclusive":bool(row.bootstrap_horizon_advantage_conclusive),
                      "main_practical_convergence_rate":float(row.winner_practical_convergence_rate)})
        gates["ready_for_v06"] = bool(gates["all_required_complete"] and convergence_ok
            and distribution_ok and timestamp_ok and agreement>=.75
            and boundary_rate<=.5 and gates["main_horizon_bootstrap_conclusive"]
            and gates["main_practical_convergence_rate"]>=.8)
    else:gates["ready_for_v06"]=False
    gates["interpretation"]=("V0.5 completion gates passed; proceed to formal regime stability work in v0.6."
        if gates["ready_for_v06"] else "V0.5 remains open; inspect failed gates without relabeling partial results.")
    (args.output/"v05_completion_gates.json").write_text(json.dumps(gates,indent=2,ensure_ascii=False),encoding="utf8")
    html=("<meta charset='utf-8'><h1>ARSH v0.5 — complete sensitivity protocol</h1>"
          f"<pre>{json.dumps(gates,indent=2,ensure_ascii=False)}</pre>"+summary.to_html(index=False))
    (args.output/"sensitivity_report.html").write_text(html,encoding="utf8")
    print(json.dumps(gates,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
