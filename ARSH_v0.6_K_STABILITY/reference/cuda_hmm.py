"""CUDA EM backend for ARSH's univariate Gaussian and Student-t HMMs.

Data preparation, model selection, bootstrap and reporting remain on CPU.  This
module only accelerates the repeated emission/forward-backward/M-step work and
returns the ordinary CPU model object so checkpoints are portable.
"""
from __future__ import annotations

import numpy as np

from student_t_hmm import GaussianSequenceHMM, Monitor, _slices


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on CUDA host
        raise ImportError("install a CUDA-enabled PyTorch build") from exc
    return torch


def cuda_available(device="cuda:0") -> bool:
    try:
        torch = _torch()
        selected=torch.device(device);index=0 if selected.index is None else selected.index
        return bool(torch.cuda.is_available() and selected.type == "cuda" and
                    torch.cuda.device_count() > index)
    except (ImportError, RuntimeError, TypeError):
        return False


def _groups(n, lengths):
    groups = {}
    for segment in _slices(n, lengths):
        groups.setdefault(segment.stop-segment.start, []).append(segment)
    return groups


def _emission(model, values, means, scale2, df, torch):
    delta2 = (values[:, None]-means[None, :]).square()/scale2[None, :]
    if isinstance(model, GaussianSequenceHMM):
        return -.5*(torch.log(2*torch.pi*scale2)[None, :]+delta2)
    nu=df[None, :]
    return (torch.lgamma((nu+1)/2)-torch.lgamma(nu/2)
            -.5*(torch.log(nu*torch.pi)+torch.log(scale2)[None, :])
            -(nu+1)/2*torch.log1p(delta2/nu))


def _e_step(model, values, means, scale2, df, lengths, torch):
    logb=_emission(model, values, means, scale2, df, torch)
    k=model.n_components;device=values.device;dtype=values.dtype
    gamma=torch.zeros((len(values),k),device=device,dtype=dtype)
    start_counts=torch.zeros(k,device=device,dtype=dtype)
    trans_counts=torch.zeros((k,k),device=device,dtype=dtype)
    total_ll=torch.zeros((),device=device,dtype=dtype)
    log_start=torch.log(torch.clamp(torch.as_tensor(model.startprob_,device=device,dtype=dtype),min=1e-300))
    log_trans=torch.log(torch.clamp(torch.as_tensor(model.transmat_,device=device,dtype=dtype),min=1e-300))
    for length,segments in _groups(len(values),lengths).items():
        indices=torch.as_tensor(np.asarray([np.arange(s.start,s.stop) for s in segments]),device=device)
        b=logb[indices];count=len(segments)
        alpha=torch.empty((count,length,k),device=device,dtype=dtype)
        alpha[:,0]=log_start+b[:,0]
        for t in range(1,length):
            alpha[:,t]=torch.logsumexp(alpha[:,t-1,:,None]+log_trans[None,:,:],dim=1)+b[:,t]
        ll=torch.logsumexp(alpha[:,-1],dim=1)
        beta=torch.zeros_like(alpha)
        for t in range(length-2,-1,-1):
            beta[:,t]=torch.logsumexp(log_trans[None,:,:]+b[:,t+1,None,:]
                                      +beta[:,t+1,None,:],dim=2)
        g=torch.exp(alpha+beta-ll[:,None,None])
        g/=g.sum(dim=2,keepdim=True)
        gamma[indices.reshape(-1)]=g.reshape(-1,k)
        start_counts+=g[:,0].sum(dim=0)
        for t in range(length-1):
            log_xi=(alpha[:,t,:,None]+log_trans[None,:,:]+b[:,t+1,None,:]
                    +beta[:,t+1,None,:]-ll[:,None,None])
            trans_counts+=torch.exp(log_xi).sum(dim=0)
        total_ll+=ll.sum()
    return total_ll,gamma,start_counts,trans_counts


def fit_cuda_hmm(model, x, lengths=None, device="cuda:0"):
    """Fit on CUDA and return a CPU-compatible ARSH model instance."""
    torch=_torch()
    if not cuda_available(device):
        raise RuntimeError(f"CUDA device unavailable: {device}")
    values_np=np.asarray(x,dtype=float).reshape(-1)
    if len(values_np)<max(20,5*model.n_components) or not np.isfinite(values_np).all():
        raise ValueError("invalid or insufficient observations")
    model._initialize(values_np,lengths)
    dtype=torch.float64;dev=torch.device(device)
    values=torch.as_tensor(values_np,device=dev,dtype=dtype)
    means=torch.as_tensor(np.asarray(model.means_).reshape(-1),device=dev,dtype=dtype)
    scale2=torch.as_tensor(model.scale2_,device=dev,dtype=dtype)
    df=torch.as_tensor(model.df_,device=dev,dtype=dtype)
    history=[]
    for iteration in range(1,model.n_iter+1):
        ll,gamma,start_counts,trans_counts=_e_step(
            model,values,means,scale2,df,lengths,torch)
        history.append(float(ll.item()))
        nk=torch.clamp(gamma.sum(dim=0),min=1e-12)
        if isinstance(model,GaussianSequenceHMM):
            means=(gamma*values[:,None]).sum(dim=0)/nk
            scale2=(gamma*(values[:,None]-means[None,:]).square()).sum(dim=0)/nk
        else:
            delta=(values[:,None]-means[None,:]).square()/scale2[None,:]
            expected_u=(df[None,:]+1)/(df[None,:]+delta)
            expected_log_u=torch.digamma((df[None,:]+1)/2)-torch.log((df[None,:]+delta)/2)
            weights=gamma*expected_u;denom=torch.clamp(weights.sum(dim=0),min=1e-12)
            means=(weights*values[:,None]).sum(dim=0)/denom
            scale2=(weights*(values[:,None]-means[None,:]).square()).sum(dim=0)/nk
            new_df=model._update_df(gamma.detach().cpu().numpy(),
                                    expected_u.detach().cpu().numpy(),
                                    expected_log_u.detach().cpu().numpy())
            df=torch.as_tensor(np.clip(new_df,model.min_df,model.max_df),device=dev,dtype=dtype)
        scale2=torch.clamp(scale2,min=model.min_scale2)
        model.startprob_=(start_counts/start_counts.sum()).detach().cpu().numpy()
        model.transmat_=((trans_counts+1e-12)/(trans_counts+1e-12).sum(dim=1,keepdim=True)).detach().cpu().numpy()
        if iteration>1:
            improvement=history[-1]-history[-2]
            if improvement>=-1e-6 and improvement<model.tol:
                model.monitor_=Monitor(history,iteration,True);break
    else:
        model.monitor_=Monitor(history,model.n_iter,False)
    model.means_=means.detach().cpu().numpy()
    model.scale2_=scale2.detach().cpu().numpy()
    model.df_=df.detach().cpu().numpy()
    model.n_features=1
    model.lengths_=[s.stop-s.start for s in _slices(len(values_np),lengths)]
    model.fit_backend_="cuda"
    model.fit_device_=str(device)
    torch.cuda.synchronize(dev)
    return model
