"""Verify a completed MLX run against its row-level evidence and frozen fixtures."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import evaluate
from mlx_evidence import read_bytes, read_json
from mlx_benchmark import choice, compare, read


def load(path):
    return read_json(path)


def assert_json_close(actual, expected, path="$"):
    """Require identical JSON structure while tolerating float roundoff."""
    assert type(actual) is type(expected), f"{path}: type mismatch"
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), f"{path}: key mismatch"
        for key in actual:
            assert_json_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        assert len(actual) == len(expected), f"{path}: length mismatch"
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_json_close(left, right, f"{path}[{index}]")
    elif isinstance(actual, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (
            f"{path}: {actual!r} != {expected!r}")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def verify(root):
    manifest, summary = load(root / "manifest.json"), load(root / "summary.json")
    checks = 0
    reference_run = manifest.get("reference_run")
    if reference_run:
        for name, expected in reference_run["sha256"].items():
            assert hashlib.sha256(read_bytes(Path(reference_run["path"]) / name)).hexdigest() == expected
            checks += 1
    for name, expected in manifest["fixture_sha256"].items():
        assert hashlib.sha256((Path("benchmarks/data") / name).read_bytes()).hexdigest() == expected
        checks += 1
    for suite in summary:
        assert suite in {"diagnostic", "quality", "shape", "generation"}
    if manifest["suite"] == "all":
        assert set(summary) == {"diagnostic", "quality", "shape", "generation"}
    elif manifest["suite"] == "quantization":
        assert set(summary) == {"diagnostic", "quality", "generation"}
    else:
        assert set(summary) == {manifest["suite"]}

    def validate(rows, gold):
        nonlocal checks
        by_id = evaluate.indexed(rows, "predictions")
        assert by_id.keys() == {row["id"] for row in gold}
        for row in gold:
            prediction = by_id[row["id"]]
            ids = [option["id"] for option in row["options"]]
            assert prediction["option_ids"] == ids
            evaluate.vector(prediction["probabilities"], ids)
            assert prediction["model"]["backend"] == "mlx"
            for key, value in manifest["model"].items():
                assert prediction["model"][key] == value
            checks += 1

    if "diagnostic" in summary:
        report = load(root / "diagnostic.json")
        for item, scalar in zip(report, summary["diagnostic"]):
            for mode in ("serial", "shared"):
                key = f"{mode}_vs_fresh"
                assert item[key] == compare(item["fresh"], item[mode])
                assert scalar[key] == item[key]
                assert not item[key]["prompt_mismatches"]
                checks += 3

    if "quality" in summary:
        report = load(root / "quality.json")
        for name in ("authored144", "perturbations108"):
            gold, rows = read(f"benchmarks/data/{name}.jsonl"), read(root / f"{name}.jsonl")
            validate(rows, gold)
            original = read(f"results/raw/predictions/direct-{name}.jsonl")
            assert_json_close(report[name]["evaluation"], evaluate.evaluate(gold, rows, comparison=original))
            assert report[name]["vs_published_torch"] == compare(original, rows)
            assert not report[name]["vs_published_torch"]["prompt_mismatches"]
            assert summary["quality"][name] == report[name]["evaluation"]["mean_family_balanced_accuracy"]
            if reference_run:
                reference = read(Path(reference_run["path"]) / f"{name}.jsonl")
                assert report[name]["vs_mlx_reference"] == compare(reference, rows)
                assert not report[name]["vs_mlx_reference"]["prompt_mismatches"]
                checks += 2
            checks += 3
    if "shape" in summary:
        report = load(root / "shape777.json")
        gold = read("benchmarks/data/shape777.jsonl")
        assert summary["shape"] == report["runs"]
        assert [run["mode"] for run in report["runs"]] == ["direct", "serial", "shared"]
        rows_by_mode = {}
        for run in report["runs"]:
            rows = read(root / f"shape777-{run['mode']}.jsonl")
            validate(rows, gold)
            rows_by_mode[run["mode"]] = rows
            assert len(run["state_seconds"]) == 37
            assert sum(run["state_seconds"]) == run["wall_seconds"]
            assert run["decisions_per_second"] == len(rows) / run["wall_seconds"]
            checks += 3
        for mode in ("serial", "shared"):
            assert report["vs_fresh"][mode] == compare(rows_by_mode["direct"], rows_by_mode[mode])
            assert not report["vs_fresh"][mode]["prompt_mismatches"]
            checks += 2
    if "generation" in summary:
        report = load(root / "generation.json")
        gold = read("benchmarks/data/shape777.jsonl")[:21]
        assert len(report["direct_runs"]) == len(report["generation_runs"]) == 3
        for direct, generated in zip(report["direct_runs"], report["generation_runs"]):
            validate(direct["outputs"], gold)
            assert generated["output_tokens"] == len(generated["timeline"])
            candidate = None
            try:
                candidate = json.loads(generated["output_text"])
            except json.JSONDecodeError:
                pass
            valid = isinstance(candidate, list) and len(candidate) == 21 and all(
                isinstance(value, str) and value in {"yes", "no"} for value in candidate
            )
            assert valid == generated["valid_complete_array"]
            assert generated["choices"] == (candidate if valid else None)
            assert generated["agreement_with_direct"] == (
                sum(choice(row) == value for row, value in zip(direct["outputs"], candidate)) / len(gold) if valid else None)
            checks += 4
        for mode, key in (("direct", "direct_median_seconds"), ("generation", "generation_median_seconds")):
            assert report[key] == statistics.median(run["total_seconds"] for run in report[f"{mode}_runs"])
        assert summary["generation"] == {key: value for key, value in report.items() if not key.endswith("runs")}
        assert report["all_generated_arrays_valid"] == all(run["valid_complete_array"] for run in report["generation_runs"])
        assert report["generation_over_direct_ratio"] == report["generation_median_seconds"] / report["direct_median_seconds"]
    print(json.dumps({"run": str(root), "verified_checks": checks, "status": "ok"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
