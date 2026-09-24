"""Required CPU/CUDA numerical parity gate before accepting CUDA artifacts."""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import arsh_v05 as arsh
from cuda_hmm import cuda_available


def align(cpu,cuda):
    a=np.asarray(cpu.means_).reshape(-1);b=np.asarray(cuda.means_).reshape(-1)
    rows,cols=linear_sum_assignment(abs(a[:,None]-b[None,:]))
    return rows,cols


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--observations",type=int,default=12000)
    args=parser.parse_args()
    if not cuda_available(args.device):raise SystemExit(f"CUDA unavailable: {args.device}")
    rng=np.random.default_rng(504);x=np.r_[rng.standard_t(5,args.observations//2)*.5,
                                          rng.standard_t(8,args.observations-args.observations//2)*1.4]
    lengths=np.repeat(120,args.observations//120)
    if lengths.sum()!=len(x):lengths=np.r_[lengths,len(x)-lengths.sum()]
    for family in arsh.ALL_HMM_FAMILIES:
        fitted={};timing={}
        for backend in ("cpu","cuda"):
            start=time.perf_counter()
            fitted[backend]=arsh.fit_one_hmm(x,family,3,20,77,lengths,backend,args.device)[0]
            timing[backend]=time.perf_counter()-start
        cpu,gpu=fitted["cpu"],fitted["cuda"];rows,cols=align(cpu,gpu)
        cpu_ll=cpu.score(x,lengths);gpu_ll=gpu.score(x,lengths)
        ll_per_observation=abs(cpu_ll-gpu_ll)/len(x)
        mean_error=float(np.max(abs(np.asarray(cpu.means_).reshape(-1)[rows]
                                    -np.asarray(gpu.means_).reshape(-1)[cols])))
        scale_error=float(np.max(abs(cpu.scale2_[rows]-gpu.scale2_[cols])))
        print({"family":family,"cpu_seconds":timing["cpu"],"cuda_seconds":timing["cuda"],
               "ll_error_per_observation":ll_per_observation,
               "max_mean_error":mean_error,"max_scale2_error":scale_error})
        if ll_per_observation>1e-7 or mean_error>1e-5 or scale_error>1e-5:
            raise SystemExit(f"CPU/CUDA parity failed for {family}")
    print("CPU/CUDA parity passed")


if __name__=="__main__":main()
