# ARSH v0.5 CPU coordinator

Colab CPU is responsible for integrity checks, merging completed CUDA shards,
bootstrap, state alignment, tables and the final HTML report.  It must not refit
models during finalization.

It also runs planned jobs with `execution_target=cpu`. In particular,
`continuous_carry` stays on CPU because its single long forward-backward chain
does not expose the short-sequence batching that benefits CUDA.

1. Copy every completed CUDA shard folder to Drive.
2. Run `merge_shards.py`, explicitly listing the expected fold IDs.
3. Run `finalize_experiment.py --output MERGED --data data/ohlc_export.csv`.
4. `--finalize-only` aborts if any fold checkpoint is missing, so a CPU fallback
   fit cannot silently contaminate a CUDA experiment.

Example:

```bash
python merge_shards.py runs/main/folds_* --output runs/main/merged \
  --expected-folds 0 1 2 3 4 5 6 7 8 9
python finalize_experiment.py --output runs/main/merged --data data/ohlc_export.csv
```

Never combine shards unless `merge_shards.py` accepts their data hash, code hash,
backend, policy and experiment configuration.

After finalizing the three policy experiments, select the policy on validation:

```bash
python select_policy.py runs_v05_revised/policy_*/merged \
  --output runs_v05_revised/policy_decision
```

Regenerate the post-policy plan using `preferred_policy`, then send that JSON to
the CUDA machine.  After all post-policy experiments are finalized, run
`compare_sensitivities.py --require-complete`; it refuses to declare v0.5 ready
when a required experiment, the 500-iteration convergence audit, distribution
diagnostics, or the timestamp-regime audit is missing.
