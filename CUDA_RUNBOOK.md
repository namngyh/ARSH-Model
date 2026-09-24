# ARSH v0.5 CUDA worker

The CUDA bundle includes the exact research CSV.  The runner verifies the data
SHA-256 through run identity, and merged shards are rejected when code, data or
configuration hashes differ.

1. Extract the bundle on the NVIDIA Windows machine.
2. Run `nvidia-smi` and update the NVIDIA driver if it is unavailable.
3. Run `./setup_cuda.ps1`.  When the default PyTorch wheel does not match the
   driver, pass the official PyTorch CUDA wheel index URL as `-TorchIndexUrl`.
4. Do not run two workers into the same output folder.
5. The ZIP already contains the immutable policy-stage plan and exact data.
   Verify both before starting:

```powershell
python verify_cuda_bundle.py
```

6. List job IDs and run one immutable shard at a time:

```powershell
python -c "import json; print(*[x['job_id'] for x in json.load(open('v05_policy_job_plan.json'))['jobs'] if x['execution_target']=='cuda'], sep='\n')"
python run_planned_job.py --plan v05_policy_job_plan.json --job policy_daily__f00_01
```

7. A direct ad-hoc two-fold shard remains available:

```powershell
./run_cuda_shard.ps1 -Output './runs/main/folds_00_01' -Folds 0,1
```

The setup is accepted only after `verify_cuda.py` passes CPU/CUDA numerical
parity for Gaussian and both Student-t HMM variants.  GPU speed is not assumed;
the parity command prints CPU and CUDA timings so an unsuitable GPU can be
rejected before the full run.

Run the three policy experiments first.  The CPU coordinator selects the policy
from validation results.  Only then generate `--phase post_policy` with that
policy; this prevents an unvalidated carry/reset choice from contaminating the
main and robustness runs.

The plan routes `continuous_carry` to CPU because one long chain is sequential;
it routes daily/session sequences to CUDA because equal-length sequences can be
batched. Only run jobs whose `execution_target` matches the current machine.

After the revised main run is finalized, copy its `merged` folder back to the
CUDA machine. Run the targeted 500-iteration audit in independent fold folders,
then combine them with `merge_convergence_audits.py`. Never let two workers
write to the same output directory.
