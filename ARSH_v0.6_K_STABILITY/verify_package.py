"""Verify the handoff package using only Python's standard library."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest_path = root/'PACKAGE_MANIFEST_SHA256.json'
if not manifest_path.exists():
    raise SystemExit('PACKAGE_MANIFEST_SHA256.json is missing. Extract the complete handoff ZIP.')
manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
for relative,expected in manifest['files'].items():
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise SystemExit(f'Missing or invalid package path: {relative}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected['sha256'] or path.stat().st_size != expected['bytes']:
        raise SystemExit(f'Package integrity mismatch: {relative}')
print(f"Package verified: {len(manifest['files'])} files.")
