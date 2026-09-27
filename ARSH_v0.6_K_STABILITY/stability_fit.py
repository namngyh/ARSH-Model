"""Resumable EM with explicit convergence criteria, using unchanged v0.5 emissions."""
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.special import digamma
from threadpoolctl import threadpool_limits

from student_t_hmm import Monitor, StudentTHMM, _normalize


def atomic_joblib(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    joblib.dump(value, tmp)
    os.replace(tmp, path)


def convergence(history, observations, cfg):
    patience = cfg['consecutive_small_improvements']
    if len(history) - 1 < cfg['min_updates'] or len(history) <= patience:
        return False
    deltas = np.diff(history[-patience-1:])
    return bool(np.isfinite(deltas).all()
                and (deltas >= -cfg['negative_ll_tolerance']).all()
                and (deltas <= cfg['absolute_ll_tolerance']).all()
                and (deltas / observations <= cfg['ll_tolerance_per_observation']).all())


def fit_seed(x, lengths, k, seed, cfg, checkpoint, backend='cpu', device='cuda:0',
             max_evaluations_this_call=None):
    """Score current parameters, check stopping, then update. Checkpoints preserve seed progress."""
    values = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
    lengths = np.asarray(lengths, dtype=np.int64)
    if len(values) < max(20, 5*k) or not np.isfinite(values).all():
        raise ValueError('Invalid or insufficient observations')
    if (lengths <= 0).any() or lengths.sum() != len(values):
        raise ValueError('Invalid independent sequence lengths')
    if backend=='cpu':
        device='cpu'
    signature = hashlib.sha256(values.tobytes() + lengths.tobytes()
        + json.dumps({'k':k, 'seed':seed, 'cfg':cfg, 'backend':backend, 'device':device},
                     sort_keys=True).encode()).hexdigest()
    path = Path(checkpoint)
    if path.exists():
        state = joblib.load(path)
        if state['signature'] != signature:
            raise ValueError(f'Checkpoint inputs differ: {path}')
        if state['finished']:
            return state['model'], state['diagnostic']
        model = state['model']
    else:
        model = StudentTHMM(k, df_mode='shared', random_state=seed,
                            n_iter=cfg['max_updates'],tol=cfg['absolute_ll_tolerance'])
        with threadpool_limits(limits=1):
            model._initialize(values, lengths)
        state = {'signature':signature, 'model':model, 'history':[], 'updates':0,
                 'finished':False, 'elapsed_seconds':0.0}

    if backend == 'cuda':
        import cuda_hmm
        if not cuda_hmm.cuda_available(device):
            raise RuntimeError(f'CUDA unavailable: {device}')
        torch = cuda_hmm._torch()
        dev = torch.device(device)
        tv = torch.as_tensor(values, dtype=torch.float64, device=dev)
    elif backend != 'cpu':
        raise ValueError(f'Unknown backend: {backend}')

    started = time.monotonic()
    evaluations = 0
    reason = None
    with threadpool_limits(limits=1):
        while True:
            if backend == 'cpu':
                ll, gamma, sc, tc = model._e_step(values, lengths)
            else:
                means = torch.as_tensor(model.means_, dtype=torch.float64, device=dev)
                scale2 = torch.as_tensor(model.scale2_, dtype=torch.float64, device=dev)
                df = torch.as_tensor(model.df_, dtype=torch.float64, device=dev)
                ll, gamma, sc, tc = cuda_hmm._e_step(model,tv,means,scale2,df,lengths,torch)
                ll = float(ll.item())
            ll = float(ll)
            if not np.isfinite(ll):
                reason = 'nonfinite_likelihood'
                break
            state['history'].append(ll)
            evaluations += 1
            if len(state['history']) > 1 and ll - state['history'][-2] < -cfg['negative_ll_tolerance']:
                reason = 'likelihood_decreased'
                break
            if convergence(state['history'], len(values), cfg):
                reason = 'strict_converged'
                break
            if state['updates'] >= cfg['max_updates']:
                reason = 'iteration_budget_exhausted'
                break
            if backend == 'cpu':
                delta2 = (values[:,None]-model.means_[None,:])**2 / model.scale2_[None,:]
                u = (model.df_[None,:]+1)/(model.df_[None,:]+delta2)
                logu = digamma((model.df_[None,:]+1)/2)-np.log((model.df_[None,:]+delta2)/2)
                weights = gamma*u
                means = (weights*values[:,None]).sum(0)/np.maximum(weights.sum(0),1e-12)
                scale2 = (weights*(values[:,None]-means[None,:])**2).sum(0)/np.maximum(gamma.sum(0),1e-12)
                dfs = model._update_df(gamma,u,logu)
                model.startprob_ = _normalize(sc)
                model.transmat_ = _normalize(tc+1e-12,axis=1)
            else:
                delta2 = (tv[:,None]-means[None,:]).square()/scale2[None,:]
                u = (df[None,:]+1)/(df[None,:]+delta2)
                logu = torch.digamma((df[None,:]+1)/2)-torch.log((df[None,:]+delta2)/2)
                weights = gamma*u
                means = (weights*tv[:,None]).sum(0)/torch.clamp(weights.sum(0),min=1e-12)
                scale2 = (weights*(tv[:,None]-means[None,:]).square()).sum(0)/torch.clamp(gamma.sum(0),min=1e-12)
                dfs = model._update_df(gamma.detach().cpu().numpy(), u.detach().cpu().numpy(),logu.detach().cpu().numpy())
                model.startprob_ = _normalize(sc.detach().cpu().numpy())
                model.transmat_ = _normalize(tc.detach().cpu().numpy()+1e-12,axis=1)
                means, scale2 = means.detach().cpu().numpy(),scale2.detach().cpu().numpy()
            model.means_ = np.asarray(means)
            model.scale2_ = np.maximum(scale2,model.min_scale2)
            model.df_ = np.clip(dfs,model.min_df,model.max_df)
            state['updates'] += 1
            if state['updates'] % cfg['checkpoint_every'] == 0:
                state['elapsed_seconds'] += time.monotonic()-started
                started = time.monotonic()
                atomic_joblib(state,path)
                print(f'    checkpoint K={k} seed={seed}: updates={state["updates"]}, ll={ll:.6f}',flush=True)
            if max_evaluations_this_call is not None and evaluations >= max_evaluations_this_call:
                state['elapsed_seconds'] += time.monotonic()-started
                atomic_joblib(state,path)
                return model, {'finished':False,'reason':'checkpointed_for_test','updates':state['updates']}
    state['elapsed_seconds'] += time.monotonic()-started
    history = state['history']
    delta = history[-1]-history[-2] if len(history)>1 else None
    strict = reason == 'strict_converged'
    model.monitor_ = Monitor(history,state['updates'],strict)
    model.n_features = 1
    model.lengths_ = lengths.tolist()
    model.strict_converged_ = strict
    model.practical_converged_ = strict
    diagnostic = {'k':int(k),'seed':int(seed),'backend':backend,'device':device,
        'finished':True,'strict_converged':strict,'reason':reason,'updates':state['updates'],
        'likelihood_evaluations':len(history),'train_ll':history[-1] if history else None,
        'last_delta':delta,'delta_per_observation':delta/len(values) if delta is not None else None,
        'observations':len(values),'sequences':len(lengths),'elapsed_seconds':state['elapsed_seconds']}
    state.update(finished=True,diagnostic=diagnostic)
    atomic_joblib(state,path)
    return model,diagnostic


def match_states(reference_model, other_model):
    """Match within the same K/scaler using means and theoretical standard deviations."""
    from scipy.optimize import linear_sum_assignment
    if reference_model.n_components != other_model.n_components:
        raise ValueError('State correspondence requires the same K')
    def features(m):
        sd = np.sqrt(m.scale2_*m.df_/(m.df_-2))
        return np.c_[np.asarray(m.means_).reshape(-1),sd]
    a,b = features(reference_model),features(other_model)
    cost = np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    rows,cols = linear_sum_assignment(cost)
    return cols[np.argsort(rows)],float(cost[rows,cols].mean())
