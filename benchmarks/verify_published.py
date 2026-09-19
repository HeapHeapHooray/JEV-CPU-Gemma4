"""Verify that the machine-readable summary is backed by committed raw evidence."""
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with (ROOT / path).open() as stream:
        return json.load(stream)


def close(left, right, tolerance=5e-10):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if abs(left - right) > tolerance:
            raise AssertionError(f"{left!r} != {right!r}")
    elif left != right:
        raise AssertionError(f"{left!r} != {right!r}")


def top_choice(row):
    return row["option_ids"][max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)]


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line.strip()]


def main():
    summary = load("results/phase1-summary.json")
    quality = load("results/raw/quality-comparison.json")
    perturb = load("results/raw/perturbation-comparison.json")
    direct_shape = load("results/raw/shape777-direct.json")
    reranker_shape = load("results/raw/shape777-reranker.json")
    compact = load("results/raw/decision-vs-compact-array.json")
    checks = 0

    semantic = summary["semantic_quality"]
    claims = {
        "authored_144_mean_family_balanced_accuracy": (
            quality["hard_label"]["authored"]["direct_logits"]["mean_family_balanced_accuracy"],
            quality["hard_label"]["authored"]["reranker"]["mean_family_balanced_accuracy"],
        ),
        "wanli_256_balanced_accuracy": (
            quality["hard_label"]["wanli"]["direct_logits"]["mean_family_balanced_accuracy"],
            quality["hard_label"]["wanli"]["reranker"]["mean_family_balanced_accuracy"],
        ),
        "every_judge_grid_36_accuracy": (
            quality["every"]["direct_logits"]["judge-grid"]["accuracy"],
            quality["every"]["reranker"]["judge-grid"]["accuracy"],
        ),
        "every_action_firewall_10_accuracy": (
            quality["every"]["direct_logits"]["action-firewall"]["accuracy"],
            quality["every"]["reranker"]["action-firewall"]["accuracy"],
        ),
        "every_code_rag_recall_at_1": (
            quality["every"]["direct_logits"]["code-rag"]["recall_at_1"],
            quality["every"]["reranker"]["code-rag"]["recall_at_1"],
        ),
        "every_company_brain_recall_at_1": (
            quality["every"]["direct_logits"]["company-brain"]["recall_at_1"],
            quality["every"]["reranker"]["company-brain"]["recall_at_1"],
        ),
    }
    for key, (direct, reranker) in claims.items():
        close(semantic[key]["direct_logits"], direct)
        close(semantic[key]["reranker"], reranker)
        checks += 2
    typesafe = quality["typesafe"]["systems"]
    for suffix, field in (("modal_agreement", "agreement"), ("tv_distance", "tv")):
        claim = semantic[f"typesafe_public_102_equal_case_{suffix}" if suffix == "modal_agreement" else "typesafe_public_102_tv_distance"]
        close(claim["direct_logits"], typesafe["direct_logits"][field])
        close(claim["reranker"], typesafe["reranker"][field])
        close(claim["published_jev"], typesafe["typesafe"][field])
        checks += 3

    for system in ("direct_logits", "reranker"):
        source = perturb["systems"][system]
        claim = summary["perturbations_36"][system]
        close(claim["base_balanced_accuracy"], source["base_original"]["mean_family_balanced_accuracy"])
        for variant in ("option_reversal", "criterion_wrapper", "irrelevant_context"):
            close(claim[variant]["balanced_accuracy"], source["variants"][variant]["evaluation"]["mean_family_balanced_accuracy"])
            close(claim[variant]["argmax_flips"], source["variants"][variant]["argmax_flips"])
            checks += 2
        close(claim["missing_evidence_confident_non_insufficient_at_0_8"], source["missing_evidence"]["confident_non_insufficient_at_0_8"])
        checks += 2

    direct_comparisons = {"fresh_batch1": 0, "serial_prefix": 5, "parallel_suffix": 6}
    for claim, raw in zip(summary["shape777"]["direct"], direct_shape["results"]):
        for summary_key, raw_key in (("wall_seconds", "wall_seconds"), ("judgments_per_second", "judgments_per_second"),
                                     ("state_p50_seconds", "state_latency_p50_seconds"), ("peak_cuda_bytes", "peak_cuda_bytes")):
            close(claim[summary_key], raw[raw_key])
            checks += 1
        close(claim["argmax_flips_vs_fresh"], direct_comparisons[claim["mode"]])
        checks += 1

    reranker_predictions = defaultdict(dict)
    for row in rows("results/raw/shape777-reranker.predictions.jsonl"):
        reranker_predictions[row["pair_batch_size"]][row["id"]] = top_choice(row)
    reference = reranker_predictions[1]
    for claim, raw in zip(summary["shape777"]["reranker"], reranker_shape["results"]):
        for summary_key, raw_key in (("wall_seconds", "wall_seconds"), ("judgments_per_second", "judgments_per_second"),
                                     ("state_p50_seconds", "state_latency_p50_seconds"), ("peak_cuda_bytes", "peak_cuda_bytes")):
            close(claim[summary_key], raw[raw_key])
            checks += 1
        flips = sum(choice != reference[row_id] for row_id, choice in reranker_predictions[claim["pair_batch_size"]].items())
        close(claim["argmax_flips_vs_batch1"], flips)
        checks += 1

    generation = summary["decision_vs_compact_generation21"]
    close(generation["direct_parallel"]["median_seconds"], compact["direct_parallel"]["median_total_seconds"])
    close(generation["compact_generation"]["median_seconds"], compact["compact_generation"]["median_total_seconds"])
    close(generation["compact_generation"]["median_output_tokens"], compact["compact_generation"]["median_output_tokens"])
    close(generation["compact_generation"]["agreement_with_direct_argmax"], compact["compact_generation"]["agreement_with_direct_argmax_first_run"])
    close(generation["wall_time_ratio_generation_over_direct"], compact["median_wall_ratio"])
    checks += 5
    print(json.dumps({"verified_summary_claims": checks, "status": "ok"}))


if __name__ == "__main__":
    main()
