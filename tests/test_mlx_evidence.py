"""Evidence storage must preserve original payload bytes and reject corruption."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "mlx_evidence", Path(__file__).resolve().parents[1] / "benchmarks/mlx_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)

benchmarks = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(benchmarks))
verify_spec = importlib.util.spec_from_file_location("verify_mlx", benchmarks / "verify_mlx.py")
verify = importlib.util.module_from_spec(verify_spec)
verify_spec.loader.exec_module(verify)


@pytest.mark.parametrize("compressed", [False, True])
def test_original_checksums_accept_plain_or_gzip(tmp_path, compressed):
    payload = b'{"probabilities": [0.125, 0.875]}\n'
    path = tmp_path / "predictions.json"
    if compressed:
        path = path.with_suffix(".json.gz")
    path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
    manifest = tmp_path / "UNCOMPRESSED_SHA256SUMS"
    manifest.write_text(hashlib.sha256(payload).hexdigest() + "  predictions.json\n")
    assert evidence.verify_checksums(manifest) == 1
    assert evidence.read_bytes(path) == payload
    assert evidence.read_json(tmp_path / "predictions.json")["probabilities"] == [0.125, 0.875]
    path.write_bytes(gzip.compress(b"changed", mtime=0) if compressed else b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evidence.verify_checksums(manifest)


def test_rejects_ambiguous_or_corrupt_compressed_evidence(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"{}\n")
    compressed = path.with_suffix(".jsonl.gz")
    compressed.write_bytes(gzip.compress(b"{}\n", mtime=0))
    with pytest.raises(ValueError, match="Both plain and compressed"):
        evidence.read_bytes(path)
    path.unlink()
    compressed.write_bytes(b"not gzip")
    with pytest.raises(gzip.BadGzipFile):
        evidence.read_bytes(path)


def test_verifier_tolerates_only_float_roundoff():
    verify.assert_json_close(
        {"score": 0.1 + 0.2, "choices": ["yes", "no"]},
        {"score": 0.3, "choices": ["yes", "no"]},
    )
    with pytest.raises(AssertionError):
        verify.assert_json_close({"score": 0.31}, {"score": 0.3})
    with pytest.raises(AssertionError):
        verify.assert_json_close({"choices": ["no"]}, {"choices": ["yes"]})
