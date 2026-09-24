"""Finalize merged CUDA fold checkpoints on a CPU-only Colab or workstation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    args=parser.parse_args()
    manifest=args.output/"merge_manifest.json"
    if not manifest.is_file():parser.error(f"missing {manifest}")
    merge=json.loads(manifest.read_text(encoding="utf8"));identity=merge["identity"]
    command=[sys.executable,str(Path(__file__).with_name("arsh_v05.py")),
             "--data",str(args.data),"--output",str(args.output),"--resume","--finalize-only",
             "--backend",identity["backend"],"--device",identity["device"],
             "--policy",identity["policy"],"--return-variant",identity["return_variant"],
             "--train-years",str(identity["train_years"]),
             "--iterations",str(identity["iterations"]),"--restarts",str(identity["restarts"]),
             "--selection-iterations",str(identity["selection_iterations"]),
             "--selection-restarts",str(identity["selection_restarts"]),
             "--top-horizons",str(identity["top_horizons"]),"--outlier-policy",identity["outlier_policy"],
             "--horizons",*map(str,identity["horizons"]),"--k-values",*map(str,identity["k_values"])]
    command += ["--folds",*map(str,merge["expected_folds"])]
    for key,flag in (("overlapping","--overlapping"),("expanding_window","--expanding-window"),
                     ("exclude_zero","--exclude-zero"),
                     ("exclude_rollover_adjacent","--exclude-rollover-adjacent"),
                     ("allow_one_internal_missing","--allow-one-internal-missing")):
        if identity.get(key):command.append(flag)
    print("Finalizing on CPU; no HMM fit will run.",flush=True)
    subprocess.run(command,check=True)


if __name__=="__main__":main()
