"""Isolated K/seed/convergence experiments; never replaces the production K=7 artifact."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import arsh_v05 as arsh
from runtime import RegimeRuntime
from label_confirmation import ConfirmationRule, confirm_labels, label_metrics
from stability_fit import atomic_joblib, fit_seed, match_states


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(value,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def write_csv(frame,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    frame.to_csv(tmp,index=False,encoding='utf-8-sig')
    os.replace(tmp,path)


def runtime_parameter_hash(runtime):
    """Canonical parameters avoid differences in pickle object-sharing on resume."""
    digest = hashlib.sha256(json.dumps({'version':runtime.model_version,'policy':runtime.policy,
        'stable_state_ids':runtime.stable_state_ids,'k':runtime.model.n_components,
        'seed':runtime.model.random_state,'class':type(runtime.model).__name__},sort_keys=True).encode())
    for name in ['means_','scale2_','df_','startprob_','transmat_']:
        array = np.ascontiguousarray(getattr(runtime.model,name),dtype=np.float64)
        digest.update(name.encode()+str(array.shape).encode()+array.tobytes())
    for name in ['mean_','var_','scale_']:
        digest.update(np.asarray(getattr(runtime.scaler,name),dtype=np.float64).tobytes())
    digest.update(np.asarray(runtime.model.monitor_.history,dtype=np.float64).tobytes())
    return digest.hexdigest()


def save_runtime(runtime,path):
    path = Path(path)
    lock = path.parent/'model_manifest.json'
    if lock.exists():
        manifest = json.loads(lock.read_text(encoding='utf-8'))
        if not path.exists() or sha(path) != manifest['artifact_sha256']:
            raise ValueError('Existing locked final artifact changed; refusing to overwrite')
        # Numerical parameters must match; pickle aliasing may differ after reload.
        if runtime_parameter_hash(runtime) != manifest['parameter_sha256']:
            raise ValueError('Resumed fit differs from locked final parameters; use a new output directory')
        return
    atomic_joblib(runtime,path)


def code_hashes():
    paths = [HERE/'run_experiment.py',HERE/'stability_fit.py',HERE/'label_confirmation.py']
    paths += sorted((HERE/'reference').glob('*.py'))
    return {str(p.relative_to(HERE)):sha(p) for p in paths}


def fit_config(cfg):
    keys = ['max_updates','min_updates','absolute_ll_tolerance','ll_tolerance_per_observation',
            'consecutive_small_improvements','negative_ll_tolerance','checkpoint_every']
    return {key:cfg[key] for key in keys}


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if cfg['family'] != 'student_t_shared' or cfg['policy'] != 'daily_sequence' or cfg['horizon_min'] != 1:
        raise ValueError('This controlled experiment fixes student_t_shared, daily_sequence, horizon=1')
    ks = cfg['k_values']
    if not ks or any(not isinstance(k,int) or k < 2 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError('K values must be unique integers >=2')
    for key in ['selection_seeds','final_seeds']:
        seeds = cfg[key]
        if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s,int) or s<0 for s in seeds):
            raise ValueError(f'Invalid seeds: {key}')
    if not (cfg['selection_train_start'] < cfg['selection_validation_start'] < cfg['data_before']):
        raise ValueError('Selection dates are not chronological')
    if not cfg['selection_train_start'] <= cfg['final_train_start'] < cfg['data_before']:
        raise ValueError('Final train window must be covered by selection source window')
    if cfg['required_last_training_day'] >= cfg['data_before']:
        raise ValueError('Required training day reaches the cutoff')
    f = fit_config(cfg)
    if not 1 <= f['min_updates'] <= f['max_updates'] or f['checkpoint_every'] < 1:
        raise ValueError('Invalid iteration settings')
    if f['consecutive_small_improvements'] < 1 or min(f['absolute_ll_tolerance'],f['ll_tolerance_per_observation']) <= 0:
        raise ValueError('Invalid convergence thresholds')
    if not 0 < cfg['min_soft_share'] < 1/max(ks) or cfg['validation_tie_margin'] < 0:
        raise ValueError('Invalid selection thresholds')
    names = [r['name'] for r in cfg['label_rules']]
    if len(set(names)) != len(names) or 'raw' not in names:
        raise ValueError('Label rules require unique names and a raw control')
    for rule in cfg['label_rules']:
        ConfirmationRule(**rule)
    raw = next(r for r in cfg['label_rules'] if r['name']=='raw')
    if (raw['min_probability'],raw['min_margin'],raw['confirm_observations']) != (0,0,1):
        raise ValueError('Raw control must exactly reproduce argmax')
    return cfg


def verify_reference():
    folder = HERE/'reference_k7/model'
    manifest = json.loads((folder/'artifact_manifest.json').read_text(encoding='utf-8'))
    artifact = folder/'runtime_research_candidate.joblib'
    if sha(artifact) != manifest[artifact.name]:
        raise ValueError('Frozen K=7 artifact hash mismatch')
    for name in ['arsh_v05.py','student_t_hmm.py','runtime.py']:
        if sha(HERE/'reference'/name) != manifest[name]:
            raise ValueError(f'Frozen source hash mismatch: {name}')
    replay_manifest = json.loads((HERE/'reference_k7/replay_manifest.json').read_text(encoding='utf-8'))
    if sha(HERE/'reference_k7/replay_seen.csv') != replay_manifest['replay_seen_sha256']:
        raise ValueError('Seen replay hash mismatch')
    return manifest


def load_seen():
    verify_reference()
    df = pd.read_csv(HERE/'reference_k7/replay_seen.csv')
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df = df.set_index('timestamp')
    starts = df.sequence_reset.to_numpy(dtype=bool)
    return df,starts


def score(runtime,frame,starts):
    x = runtime.scaler.transform(frame[['log_return']])
    posterior,logscore = arsh.causal_filter(runtime.model,x,starts=starts)
    return posterior,logscore-float(np.log(runtime.scaler.scale_[0]))


def make_runtime(model,scaler,cfg,k,phase):
    std = np.sqrt(model.scale2_*model.df_/(model.df_-2))
    mapping = {int(internal):f'k{k}::S{rank}' for rank,internal in enumerate(np.argsort(std))}
    return RegimeRuntime(model,scaler,cfg['policy'],f"{cfg['experiment']}-{phase}-k{k}",mapping)


def audit_labels(runtime,frame,starts,cfg,out,period):
    posterior,density = score(runtime,frame,starts)
    raw = posterior.argmax(1)
    detail = pd.DataFrame({'timestamp':frame.index.astype(str),'sequence_reset':starts,
        'log_return':frame.log_return.to_numpy(),'raw_state_id':[runtime.stable_state_ids[int(i)] for i in raw],
        'raw_confidence':posterior.max(1),'predictive_log_density':density})
    # Column labels use stable IDs explicitly; no implicit internal-ID mapping.
    for i in range(runtime.model.n_components):
        detail['p_'+runtime.stable_state_ids[i]] = posterior[:,i]
    metrics = []
    for rule_cfg in cfg['label_rules']:
        rule = ConfirmationRule(**rule_cfg)
        result = confirm_labels(posterior,starts,rule)
        metrics_for_rule = label_metrics(result,starts)
        accepted = np.flatnonzero(result['changed'])
        wall_wait = np.zeros(len(frame),dtype=float)
        if len(accepted):
            first = result['confirmation_start_index'][accepted]
            wall_wait[accepted] = (frame.index[accepted]-frame.index[first]).total_seconds()/60
        metrics_for_rule['mean_wait_wall_minutes_after_first_qualifying_observation'] = float(wall_wait[accepted].mean()) if len(accepted) else None
        metrics_for_rule['max_wait_wall_minutes_after_first_qualifying_observation'] = float(wall_wait[accepted].max()) if len(accepted) else None
        metrics.append({'rule':rule.name,'period':period,'observations':len(frame),
                        'model_version':runtime.model_version,'metrics':metrics_for_rule})
        prefix = rule.name+'_'
        detail[prefix+'state_id'] = [runtime.stable_state_ids[int(i)] if i>=0 else 'UNCERTAIN' for i in result['display_label']]
        detail[prefix+'confirmed_memory_state_id'] = [runtime.stable_state_ids[int(i)] for i in result['confirmed_label']]
        detail[prefix+'confirmed_memory_probability'] = result['display_probability']
        detail[prefix+'uncertain'] = result['uncertain']
        detail[prefix+'wait_observations'] = result['confirmation_wait_observations']
        detail[prefix+'wait_wall_minutes'] = wall_wait
    write_json({'period':period,'label_rules':cfg['label_rules'],'results':metrics,
        'note':'Display confirmation changes labels only, never posteriors or predictive density. No true-regime accuracy is measured.'},out/'label_metrics.json')
    write_csv(detail,out/'confirmed_states.csv')
    shares = posterior.mean(0)
    means,scales,stds,dfs = arsh.distribution(runtime.model,cfg['family'],runtime.scaler)
    rows = []
    for i in sorted(runtime.stable_state_ids,key=runtime.stable_state_ids.get):
        rows.append({'state_id':runtime.stable_state_ids[i],'internal_state':i,
            'mean_return_pct':float(means[i]),'std_return_pct':float(stds[i]),'df':float(dfs[i]),
            'soft_share':float(shares[i]),'hard_share':float((raw==i).mean()),
            'self_transition':float(runtime.model.transmat_[i,i])})
    write_csv(pd.DataFrame(rows),out/'state_profiles.csv')
    daily = pd.DataFrame({'date':frame.index.normalize(),'log_density':density})
    for i in range(runtime.model.n_components):
        daily[runtime.stable_state_ids[i]] = posterior[:,i]
    daily = daily.groupby('date').mean().reset_index()
    write_csv(daily,out/'daily_state_shares.csv')
    return posterior,density


def export_db(cfg,out):
    import v06_fetch_db as fetch
    fetch.DAYS_DIR = out/'source/days'
    start = pd.Timestamp(cfg['selection_train_start']).date()
    end = (pd.Timestamp(cfg['data_before'])-pd.Timedelta(days=1)).date()
    metas = fetch.sync(start,end)
    stored = [m for m in metas if m['stored']]
    if not stored:
        raise ValueError('No complete training days from DB')
    destination = out/'source/training_complete_days.csv'
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp = destination.with_suffix('.csv.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as stream:
        for i,meta in enumerate(stored):
            p = fetch.DAYS_DIR/meta['csv']
            if sha(p) != meta['sha256'] or not meta['complete']:
                raise ValueError(f'Invalid immutable source day: {p.name}')
            lines = p.read_text(encoding='utf-8').splitlines(keepends=True)
            stream.writelines(lines if i==0 else lines[1:])
    os.replace(tmp,destination)
    write_json({'days':metas,'sha256':sha(destination),'end_exclusive':cfg['data_before']},out/'source/source_manifest.json')
    return destination


def prepare_frames(data,cfg):
    raw,audit = arsh.load_minutes(Path(data))
    raw = raw[(raw.index>=pd.Timestamp(cfg['selection_train_start'])) & (raw.index<pd.Timestamp(cfg['data_before']))]
    if raw.empty:
        raise ValueError('No data before the training cutoff')
    days = raw.groupby(raw.index.normalize())
    complete_days = [d for d,g in days if (g.index.strftime('%H:%M')=='14:45').any()]
    incomplete = [str(d.date()) for d,g in days if d not in complete_days]
    raw = raw[raw.index.normalize().isin(complete_days)]
    if raw.empty or raw.index.max().date().isoformat() < cfg['required_last_training_day']:
        raise ValueError(f"Source ends before required complete day {cfg['required_last_training_day']}; do not silently shorten this comparison")
    if raw.index.min().normalize() > pd.Timestamp(cfg['selection_train_start']):
        raise ValueError('Source does not cover selection_train_start')
    frames,_,rejected,_ = arsh.build_returns(raw,horizons=(1,),overlapping=False,allow_one_internal_missing=False)
    frame = frames[1]
    train = frame[frame.index<pd.Timestamp(cfg['selection_validation_start'])]
    valid = frame[frame.index>=pd.Timestamp(cfg['selection_validation_start'])]
    final = frame[frame.index>=pd.Timestamp(cfg['final_train_start'])]
    if min(len(train),len(valid),len(final)) < 100:
        raise ValueError('Insufficient train/validation/final observations')
    manifest = verify_reference()
    old_source_hash = next(v for k,v in manifest.items() if k.startswith('training_data_'))
    audit.update({'data_sha256':sha(data),'matches_original_training_export_bytes':sha(data)==old_source_hash,
        'completeness':'ATC checked; CSV mode assumes bars were finalized at export. DB mode additionally enforces is_final.',
        'incomplete_days_skipped':incomplete,'first_day':str(raw.index.min().date()),
        'last_day':str(raw.index.max().date()),'train_observations':len(train),
        'validation_observations':len(valid),'final_observations':len(final),'rejected_windows':len(rejected)})
    return train,valid,final,audit


def verify_cuda(out,cfg,device):
    import cuda_hmm
    if not cuda_hmm.cuda_available(device):
        raise RuntimeError('CUDA unavailable. Run full training on the CUDA machine; use --backend cpu only deliberately.')
    path = out/'cuda_verification.json'
    identity = {'code_hashes':code_hashes(),'device':device}
    torch = cuda_hmm._torch()
    identity.update(torch_version=str(torch.__version__),cuda_version=str(torch.version.cuda),
                    gpu=torch.cuda.get_device_name(torch.device(device)))
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous.get('identity')==identity and previous.get('passed'):
            return
    rng = np.random.default_rng(217)
    values = np.r_[rng.normal(-.7,.35,160),rng.normal(.1,.8,160),rng.normal(1,.4,160)]
    rng.shuffle(values)
    f = fit_config(cfg)
    f.update(max_updates=25,min_updates=25)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in [min(cfg['k_values']),max(cfg['k_values'])]:
            cpu,dc = fit_seed(values,[80]*6,k,62,f,Path(tmp)/f'cpu{k}.joblib')
            gpu,dg = fit_seed(values,[80]*6,k,62,f,Path(tmp)/f'cuda{k}.joblib','cuda',device)
            pc = arsh.causal_filter(cpu,values[:,None])[0]
            pg = arsh.causal_filter(gpu,values[:,None])[0]
            difference = float(abs(pc-pg).max())
            ll_difference = abs(dc['train_ll']-dg['train_ll'])
            passed = bool(difference<1e-6 and ll_difference<1e-5
                          and dc['reason']!='likelihood_decreased' and dg['reason']!='likelihood_decreased')
            results.append({'k':k,'max_posterior_difference':difference,'train_ll_difference':ll_difference,'passed':passed})
    passed = all(r['passed'] for r in results)
    write_json({'identity':identity,'passed':passed,'results':results},path)
    if not passed:
        raise RuntimeError('CPU/CUDA verification failed; full training stopped')


def establish_lock(out,cfg,data_audit,backend,device):
    identity = {'config':cfg,'code_hashes':code_hashes(),'source_data_sha256':data_audit['data_sha256'],
        'reference_model_sha256':sha(HERE/'reference_k7/model/runtime_research_candidate.joblib'),
        'backend':backend,'device':device}
    path = out/'experiment_lock.json'
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing['identity'] != identity:
            raise ValueError('Output directory belongs to different data/config/code/backend. Use a new --out directory.')
        return existing
    versions = {n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','hmmlearn']}
    lock = {'identity':identity,'created_at':datetime.now().astimezone().isoformat(),
        'software_versions':versions,'data_audit':data_audit,
        'seen_period_usage':'August/September replay is descriptive only. No selection uses it.'}
    write_json(lock,path)
    return lock


def fit_many(phase,frame,valid,cfg,out,backend,device):
    scaler = StandardScaler().fit(frame[['log_return']])
    x = scaler.transform(frame[['log_return']])
    lengths = arsh.sequence_lengths(frame,cfg['policy'])
    vstarts = arsh.sequence_starts(valid,cfg['policy'])
    xv = scaler.transform(valid[['log_return']])
    jac = float(np.log(scaler.scale_[0]))
    _,student = arsh.fit_iid(x)
    baseline_density = arsh.iid_scores(student,xv,'student_t')-jac
    seeds = cfg['selection_seeds'] if phase=='selection' else cfg['final_seeds']
    all_diagnostics,per_k,best_models = [],[],{}
    for k in cfg['k_values']:
        fits = []
        for seed in seeds:
            print(f'{phase}: K={k}, seed={seed}',flush=True)
            model,diag = fit_seed(x,lengths,k,seed,fit_config(cfg),out/f'fits/{phase}/k{k}/seed{seed}.joblib',backend,device)
            diag = dict(diag,phase=phase)
            if diag['strict_converged']:
                posterior,logscore = arsh.causal_filter(model,xv,starts=vstarts)
                density = logscore-jac
                diag.update(validation_log_density=float(density.mean()),
                    validation_gain_vs_student=float((density-baseline_density).mean()),
                    validation_min_soft_share=float(posterior.mean(0).min()))
                fits.append((model,diag,posterior))
            all_diagnostics.append(diag)
            exported = pd.DataFrame(all_diagnostics)
            if phase=='final':
                exported = exported.rename(columns={c:c.replace('validation_','training_audit_') for c in exported if c.startswith('validation_')})
            write_csv(exported,out/f'{phase}_seed_metrics.csv')
            print(f"  {diag['reason']}, updates={diag['updates']}, elapsed={diag['elapsed_seconds']:.1f}s",flush=True)
        if not fits:
            per_k.append({'k':k,'strict_converged_seeds':0,'total_seeds':len(seeds),
                          'eligible':False,'status':'NO_CONVERGED_SEED'})
            continue
        # Seed selection uses training likelihood only, never the best validation score.
        model,diag,posterior = max(fits,key=lambda item:item[1]['train_ll'])
        best_models[k] = model
        eligible = diag['validation_min_soft_share']>=cfg['min_soft_share']
        per_k.append({'k':k,'chosen_seed':diag['seed'],'strict_converged_seeds':len(fits),
            'total_seeds':len(seeds),'eligible':bool(eligible),'status':'OK' if eligible else 'SOFT_SHARE_BELOW_THRESHOLD',
            'validation_log_density':diag['validation_log_density'],
            'validation_gain_vs_student':diag['validation_gain_vs_student'],
            'validation_min_soft_share':diag['validation_min_soft_share']})
        runtime = make_runtime(model,scaler,cfg,k,phase)
        save_runtime(runtime,out/f'{phase}_models/k{k}/runtime_research_candidate.joblib')
        audit_labels(runtime,valid,vstarts,cfg,out/f'{phase}_audit/k{k}',
                     'PRE_AUGUST_VALIDATION' if phase=='selection' else 'IN_SAMPLE_FINAL_TRAIN_AUDIT')
        alignment = []
        reference_labels = posterior.argmax(1)
        for other,other_diag,other_posterior in fits:
            permutation,cost = match_states(model,other)
            aligned = other_posterior[:,permutation]
            alignment.append({'k':k,'reference_seed':diag['seed'],'other_seed':other_diag['seed'],
                'mean_parameter_match_cost_standardized':cost,
                'posterior_mean_absolute_difference':float(abs(posterior-aligned).mean()),
                'argmax_disagreement_after_alignment':float((reference_labels!=aligned.argmax(1)).mean()),
                'other_min_soft_share':float(aligned.mean(0).min()),
                'reference_internal_to_other_internal':json.dumps(permutation.tolist())})
        write_csv(pd.DataFrame(alignment),out/f'{phase}_audit/k{k}/seed_alignment.csv')
    table = pd.DataFrame(per_k)
    if phase=='final':
        table = table.rename(columns={c:c.replace('validation_','training_audit_') for c in table if c.startswith('validation_')})
    write_csv(table,out/f'{phase}_results.csv')
    return table,scaler,student,best_models


def select_k(table,cfg):
    eligible = table[table.eligible]
    if eligible.empty:
        return {'status':'NO_ELIGIBLE_K','chosen_k':None,'reason':'No K has a strictly converged selected seed and >=1% validation soft occupancy.'}
    best = float(eligible.validation_log_density.max())
    tied = eligible[eligible.validation_log_density>=best-cfg['validation_tie_margin']]
    return {'status':'RESEARCH_CANDIDATE','chosen_k':int(tied.k.min()),
        'best_validation_log_density':best,'tied_k':tied.k.astype(int).tolist(),
        'rule':'Strict convergence, >=1% soft occupancy; within 0.002 log-density of best, prefer smaller K.',
        'note':'No production model is replaced. This is selection on pre-August validation only.'}


def run(args,cfg):
    if args.backend=='cpu':
        args.device='cpu'
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.backend=='cuda':
        verify_cuda(out,cfg,args.device)
    if args.fetch_db:
        if not os.environ.get('PG_DSN'):
            raise ValueError('PG_DSN is not set. Supply --data with the complete original training export or run on the DB-connected machine.')
        # A resume uses the frozen export instead of silently refreshing DB source versions.
        frozen = out/'source/training_complete_days.csv'
        data = frozen if (out/'experiment_lock.json').exists() and frozen.exists() else export_db(cfg,out)
    else:
        data = args.data
    train,valid,final,data_audit = prepare_frames(data,cfg)
    lock = establish_lock(out,cfg,data_audit,args.backend,args.device)
    write_json(data_audit,out/'data_audit.json')
    decision_path = out/'selection_decision.json'
    if args.phase in ['selection','all']:
        table,_,_,_ = fit_many('selection',train,valid,cfg,out,args.backend,args.device)
        write_json(select_k(table,cfg),decision_path)
    if args.phase in ['final','all']:
        if not decision_path.exists():
            raise ValueError('Run selection first; final models must not select K on seen replay')
        # The third argument is a training audit here; never interpret it as validation.
        table,scaler,student,models = fit_many('final',final,final,cfg,out,args.backend,args.device)
        seen,seen_starts = load_seen()
        for k,model in models.items():
            runtime_path = out/f'final_models/k{k}/runtime_research_candidate.joblib'
            row = table[table.k==k].iloc[0]
            decision = json.loads(decision_path.read_text(encoding='utf-8'))
            selection = pd.read_csv(out/'selection_results.csv')
            selected_row = selection[selection.k==k].iloc[0]
            runtime = RegimeRuntime.load(runtime_path)
            baseline_path = runtime_path.parent/'student_baseline.json'
            if baseline_path.exists() and (runtime_path.parent/'model_manifest.json').exists():
                original_manifest = json.loads((runtime_path.parent/'model_manifest.json').read_text(encoding='utf-8'))
                if sha(baseline_path) != original_manifest['student_baseline_sha256']:
                    raise ValueError('Locked Student-t baseline changed')
            else:
                write_json({'parameters':student},baseline_path)
            model_manifest = {'artifact_sha256':sha(runtime_path),'parameter_sha256':runtime_parameter_hash(runtime),
                'student_baseline_sha256':sha(baseline_path),'config':cfg,'experiment_identity':lock['identity'],
                'locked_at':datetime.now().astimezone().isoformat(),'chosen_seed':int(row.chosen_seed),
                'strict_converged':True,'training_end_exclusive':cfg['data_before'],
                'future_holdout_not_before':cfg['new_holdout_start'],'label_rules_are_experimental':True,
                'selection_eligible':bool(selected_row.eligible),
                'chosen_k_by_pre_august_validation':decision['chosen_k'],
                'source_hash_matches_original':data_audit['matches_original_training_export_bytes']}
            manifest_path = runtime_path.parent/'model_manifest.json'
            if manifest_path.exists():
                existing = json.loads(manifest_path.read_text(encoding='utf-8'))
                if existing['artifact_sha256'] != sha(runtime_path):
                    raise ValueError('Previously locked final artifact changed')
            else:
                write_json(model_manifest,manifest_path)
            audit_labels(runtime,seen,seen_starts,cfg,out/f'seen_replay/k{k}','ALREADY_SEEN_DESCRIPTIVE_ONLY')
        audit_reference(cfg,out/'frozen_k7_reference')
        write_json({'status':'RESEARCH_RUN_FINISHED','selection':decision,
                    'exported_final_k':sorted(models),'production_model_replaced':False},out/'experiment_status.json')
    print(f'Results: {out}',flush=True)


def audit_reference(cfg,out):
    seen,starts = load_seen()
    runtime = RegimeRuntime.load(HERE/'reference_k7/model/runtime_research_candidate.joblib')
    posterior,_ = audit_labels(runtime,seen,starts,cfg,out,'ALREADY_SEEN_DESCRIPTIVE_ONLY')
    saved = seen[[f'p_state_{i}' for i in range(7)]].to_numpy()
    difference = float(abs(posterior-saved).max())
    if difference>1e-9:
        raise ValueError(f'Frozen K=7 replay mismatch: {difference}')
    write_json({'max_posterior_difference':difference,'observations':len(seen),
        'status':'VERIFIED_DESCRIPTIVE_REPLAY','no_model_selection':True},out/'verification.json')
    print(f'Frozen K=7 descriptive label audit: {out}',flush=True)


def evaluate_future(args,cfg):
    folder = args.model.resolve()
    manifest = json.loads((folder/'model_manifest.json').read_text(encoding='utf-8'))
    if manifest['config'] != cfg:
        raise ValueError('Use the exact config that was locked with this model')
    if sha(folder/'runtime_research_candidate.joblib') != manifest['artifact_sha256']:
        raise ValueError('Model hash differs from its lock')
    if sha(folder/'student_baseline.json') != manifest['student_baseline_sha256']:
        raise ValueError('Student-t baseline differs from its lock')
    if manifest['experiment_identity']['code_hashes'] != code_hashes():
        raise ValueError('Code changed after model lock; restore locked code before scoring')
    lock_day = pd.Timestamp(manifest['locked_at']).date()
    first = max(pd.Timestamp(cfg['new_holdout_start']).date(),lock_day+timedelta(days=1))
    raw,_ = arsh.load_minutes(args.data)
    raw = raw[raw.index.date>=first]
    complete = [d for d,g in raw.groupby(raw.index.normalize()) if (g.index.strftime('%H:%M')=='14:45').any()]
    raw = raw[raw.index.normalize().isin(complete)]
    if raw.empty:
        raise ValueError(f'No complete unseen days on or after {first}; historical August/September is not a fresh holdout')
    frames,_,rejected,_ = arsh.build_returns(raw,horizons=(1,),overlapping=False,allow_one_internal_missing=False)
    frame = frames[1]
    starts = arsh.sequence_starts(frame,cfg['policy'])
    runtime = RegimeRuntime.load(folder/'runtime_research_candidate.joblib')
    posterior,density = audit_labels(runtime,frame,starts,cfg,args.out,'PROSPECTIVE_AFTER_MODEL_LOCK')
    student = json.loads((folder/'student_baseline.json').read_text(encoding='utf-8'))['parameters']
    x = runtime.scaler.transform(frame[['log_return']])
    baseline = arsh.iid_scores(student,x,'student_t')-np.log(runtime.scaler.scale_[0])
    old = RegimeRuntime.load(HERE/'reference_k7/model/runtime_research_candidate.joblib')
    _,old_density = score(old,frame,starts)
    write_json({'first_day':str(frame.index.min().date()),'last_day':str(frame.index.max().date()),
        'observations':len(frame),'rejected_windows':len(rejected),'source_sha256':sha(args.data),
        'gain_vs_student':float((density-baseline).mean()),'gain_vs_frozen_k7':float((density-old_density).mean()),
        'min_soft_share':float(posterior.mean(0).min()),'model_sha256':manifest['artifact_sha256'],
        'note':'Prospective evaluation, no fitting, selection or production deployment.'},args.out/'future_evaluation.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'experiment_config.json')
    subs = parser.add_subparsers(dest='command',required=True)
    subs.add_parser('plan')
    r = subs.add_parser('run')
    source = r.add_mutually_exclusive_group(required=True)
    source.add_argument('--data',type=Path)
    source.add_argument('--fetch-db',action='store_true')
    r.add_argument('--out',type=Path,default=HERE/'outputs/full')
    r.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    r.add_argument('--device',default='cuda:0')
    r.add_argument('--phase',choices=['selection','final','all'],default='all')
    a = subs.add_parser('audit-reference')
    a.add_argument('--out',type=Path,default=HERE/'outputs/reference_label_audit')
    v = subs.add_parser('verify-cuda')
    v.add_argument('--out',type=Path,default=HERE/'outputs/full')
    v.add_argument('--device',default='cuda:0')
    e = subs.add_parser('evaluate-future')
    e.add_argument('--model',type=Path,required=True)
    e.add_argument('--data',type=Path,required=True)
    e.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    verify_reference()
    if args.command=='plan':
        import cuda_hmm
        count = len(cfg['k_values'])*(len(cfg['selection_seeds'])+len(cfg['final_seeds']))
        print(json.dumps({'k_values':cfg['k_values'],'family':cfg['family'],'total_seed_fits':count,
            'max_updates_per_fit':cfg['max_updates'],'strict_convergence':fit_config(cfg),
            'cuda_available':cuda_hmm.cuda_available(),'PG_DSN_available':bool(os.environ.get('PG_DSN')),
            'cutoff_exclusive':cfg['data_before'],'new_holdout_start':cfg['new_holdout_start'],
            'original_k7_kept':True,'same_budget_k7_refit':True},indent=2))
    elif args.command=='run':
        run(args,cfg)
    elif args.command=='audit-reference':
        audit_reference(cfg,args.out)
    elif args.command=='verify-cuda':
        verify_cuda(args.out,cfg,args.device)
    else:
        evaluate_future(args,cfg)


if __name__=='__main__':
    try:
        main()
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'STOPPED: {exc}',file=sys.stderr)
        raise SystemExit(2)
