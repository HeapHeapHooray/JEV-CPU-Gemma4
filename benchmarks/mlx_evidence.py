"""Read plain or losslessly gzipped MLX evidence without changing its schema."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def read_bytes(path: str | Path) -> bytes:
    """Resolve an original filename or an explicit .gz path; reject ambiguity."""
    path = Path(path)
    if path.suffix != ".gz":
        compressed = path.with_name(path.name + ".gz")
        if path.exists() and compressed.exists():
            raise ValueError(f"Both plain and compressed evidence exist: {path}")
        if compressed.exists():
            path = compressed
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return path.read_bytes()


def read_json(path: str | Path):
    return json.loads(read_bytes(path))


def verify_checksums(manifest: Path) -> int:
    """Check decompressed payload bytes against their original recorded hashes."""
    count = 0
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256(read_bytes(manifest.parent / name)).hexdigest()
        if actual != expected:
            raise ValueError(f"Original evidence checksum mismatch: {name}")
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps({"verified_original_payloads": verify_checksums(args.manifest), "status": "ok"}))
