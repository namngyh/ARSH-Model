"""Validate and merge independent ARSH fold shards without refitting."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib


IGNORED_IDENTITY_KEYS={"selected_fold_ids","folds"}


def common_identity(identity):
    return {k:v for k,v in identity.items() if k not in IGNORED_IDENTITY_KEYS}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-folds",type=int,nargs="+",required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("merge output must be empty")
    identities=[];folds={}
    for shard in args.shards:
        identity_path=shard/"_checkpoints"/"run_identity.joblib"
        if not identity_path.is_file():parser.error(f"missing identity: {identity_path}")
        identity=joblib.load(identity_path);identities.append(identity)
        progress_path=shard/"progress.json"
        if not progress_path.is_file() or not json.loads(progress_path.read_text(encoding="utf8"))["complete"]:
            parser.error(f"incomplete shard: {shard}")
        for checkpoint in (shard/"_checkpoints").glob("fold_[0-9][0-9].joblib"):
            fold=int(checkpoint.stem.split("_")[1])
            if fold in folds:parser.error(f"duplicate fold {fold}: {checkpoint} and {folds[fold]}")
            folds[fold]=checkpoint
    reference=common_identity(identities[0])
    if any(common_identity(item)!=reference for item in identities[1:]):
        parser.error("shard configuration/data/code identities differ")
    expected=set(args.expected_folds);found=set(folds)
    if found!=expected:parser.error(f"fold mismatch; missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    checkpoint_out=args.output/"_checkpoints";checkpoint_out.mkdir(parents=True)
    for fold,path in sorted(folds.items()):
        shutil.copy2(path,checkpoint_out/f"fold_{fold:02d}.joblib")
    (args.output/"merge_manifest.json").write_text(json.dumps({
        "source_shards":[str(p.resolve()) for p in args.shards],
        "folds":sorted(folds),"expected_folds":sorted(expected),"identity":reference,
        "next_step":"Run arsh_v05.py with the same configuration, --resume and --finalize-only."
    },indent=2,ensure_ascii=False,default=str),encoding="utf8")
    print(f"Merged {len(folds)} fold checkpoints into {args.output.resolve()}")
    print("Now run the identical full command with --resume --finalize-only.")


if __name__=="__main__":main()
