"""Verify every CUDA worker file and the canonical research data hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_DATA_SHA256="bf84b23d6fa48b9fd90c477ef6788cc0aca19e86a0480d91600fa77363e71096"


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):value.update(block)
    return value.hexdigest()


def main():
    root=Path(__file__).resolve().parent
    manifest_path=root/"CUDA_BUNDLE_SHA256.json"
    if not manifest_path.is_file():raise SystemExit("missing CUDA_BUNDLE_SHA256.json")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"));bad=[]
    for name,expected in manifest.items():
        path=root/name
        if not path.is_file() or digest(path)!=expected:bad.append(name)
    data_hash=digest(root/"data"/"ohlc_export.csv")
    if data_hash!=EXPECTED_DATA_SHA256:bad.append("data/ohlc_export.csv (canonical hash mismatch)")
    if bad:raise SystemExit(f"CUDA bundle verification failed: {bad}")
    print(f"CUDA bundle verified: {len(manifest)} files; data SHA-256={data_hash}")


if __name__=="__main__":main()
