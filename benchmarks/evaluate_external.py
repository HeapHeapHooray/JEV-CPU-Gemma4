"""Recompute the published TypeSafe-subset and Every metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prediction_map(rows):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"Duplicate prediction ID {row['id']}")
        probabilities = row.get("probabilities")
        option_ids = row.get("option_ids")
        if (
            not isinstance(probabilities, list)
            or not isinstance(option_ids, list)
            or len(probabilities) != len(option_ids)
            or len(probabilities) < 2
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in probabilities)
            or abs(sum(probabilities) - 1) > 1e-4
        ):
            raise ValueError(f"Invalid distribution for {row['id']}")
        result[row["id"]] = row
    return result


def aligned_distribution(row, prediction):
    option_ids = [option["id"] for option in row["options"]]
    if prediction["option_ids"] != option_ids:
        raise ValueError(f"Option IDs/order differ for {row['id']}")
    return prediction["probabilities"]


def type_safe(gold, direct, reranker):
    systems = {"direct": prediction_map(direct), "reranker": prediction_map(reranker)}
    expected = {row["id"] for row in gold}
    if any(set(predictions) != expected for predictions in systems.values()):
        raise ValueError("TypeSafe gold and prediction IDs differ")
    records = defaultdict(lambda: defaultdict(list))
    for row in gold:
        target = row["target_distribution"]
        systems_for_row = {
            name: aligned_distribution(row, predictions[row["id"]])
            for name, predictions in systems.items()
        }
        systems_for_row["published_jev"] = row["published_models"]["typesafe"]["distribution"]
        for name, distribution in systems_for_row.items():
            if len(distribution) != len(target):
                raise ValueError(f"Distribution width differs for {row['id']}")
            predicted = max(range(len(distribution)), key=distribution.__getitem__)
            records[name][row["group_id"]].append(
                {
                    "agreement": float(predicted == row["label"]),
                    "total_variation": sum(abs(left - right) for left, right in zip(distribution, target)) / 2,
                }
            )
    result = {}
    for name, groups in records.items():
        result[name] = {
            "rows": len(gold),
            "cases": len(groups),
            "equal_case_modal_agreement": statistics.mean(
                statistics.mean(item["agreement"] for item in rows) for rows in groups.values()
            ),
            "equal_case_total_variation": statistics.mean(
                statistics.mean(item["total_variation"] for item in rows) for rows in groups.values()
            ),
        }
    return result


def yes_probability(row, predictions):
    prediction = predictions[row["id"]]
    aligned_distribution(row, prediction)
    return prediction["probabilities"][prediction["option_ids"].index("yes")]


def hard_label(rows, predictions):
    correct = 0
    for row in rows:
        probability = yes_probability(row, predictions)
        correct += (probability >= 0.5) == (row["options"][row["label"]]["id"] == "yes")
    return {"rows": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def retrieval(rows, predictions):
    queries = defaultdict(list)
    for row in rows:
        queries[row["provenance"]["question_id"]].append(row)
    reciprocal_ranks, recall_one, recall_three = [], [], []
    for candidates in queries.values():
        ranked = sorted(candidates, key=lambda row: -yes_probability(row, predictions))
        relevant_ids = {
            row["id"] for row in ranked if row["options"][row["label"]]["id"] == "yes"
        }
        if not relevant_ids:
            raise ValueError("Retrieval query has no relevant document")
        first = next(index + 1 for index, row in enumerate(ranked) if row["id"] in relevant_ids)
        reciprocal_ranks.append(1 / first)
        recall_one.append(sum(row["id"] in relevant_ids for row in ranked[:1]) / len(relevant_ids))
        recall_three.append(sum(row["id"] in relevant_ids for row in ranked[:3]) / len(relevant_ids))
    return {
        **hard_label(rows, predictions),
        "queries": len(queries),
        "recall_at_1": statistics.mean(recall_one),
        "recall_at_3": statistics.mean(recall_three),
        "mrr": statistics.mean(reciprocal_ranks),
    }


def firewall_gate(signals):
    if (
        (signals["destructive"] > 0.72 and signals["reversible"] < 0.35)
        or signals["exceeds_scope"] > 0.78
        or signals["shares_sensitive"] > 0.78
    ):
        return "block"
    if (
        signals["needs_confirmation"] > 0.55
        or signals["destructive"] > 0.45
        or signals["shares_sensitive"] > 0.65
    ):
        return "confirm"
    return "allow"


def firewall(rows, predictions, actions):
    by_item = defaultdict(dict)
    for row in rows:
        by_item[row["provenance"]["source_item"]][row["provenance"]["question_id"]] = yes_probability(
            row, predictions
        )
    expected = actions["expected_actions"]
    if set(by_item) != set(expected):
        raise ValueError("Firewall action IDs differ")
    decisions = {item: firewall_gate(signals) for item, signals in by_item.items()}
    correct = sum(decisions[item] == action for item, action in expected.items())
    return {"rows": len(rows), "actions": len(expected), "correct": correct, "accuracy": correct / len(expected)}


def every(gold, inference, direct, reranker, actions):
    systems = {"direct": prediction_map(direct), "reranker": prediction_map(reranker)}
    gold_ids = {row["id"] for row in gold}
    inference_ids = {row["id"] for row in inference}
    if len(inference_ids) != len(inference) or any(set(predictions) != inference_ids for predictions in systems.values()):
        raise ValueError("Every inference and prediction IDs differ")
    if not gold_ids <= inference_ids:
        raise ValueError("Every inference rows omit author-labeled rows")
    families = defaultdict(list)
    for row in gold:
        families[row["provenance"]["experiment"]].append(row)
    result = {}
    for name, predictions in systems.items():
        firewall_rows = [row for row in inference if row["id"].startswith("every/action-firewall/")]
        result[name] = {
            "judge-grid": hard_label(families["judge-grid"], predictions),
            "code-rag": retrieval(families["code-rag"], predictions),
            "company-brain": retrieval(families["company-brain"], predictions),
            "action-firewall": firewall(firewall_rows, predictions, actions),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("typesafe", "every"), required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--reranker", type=Path, required=True)
    parser.add_argument("--inference", type=Path)
    parser.add_argument("--firewall-actions", type=Path)
    args = parser.parse_args()
    gold, direct, reranker = read(args.gold), read(args.direct), read(args.reranker)
    if args.source == "typesafe":
        if args.firewall_actions or args.inference:
            parser.error("--inference and --firewall-actions apply only to Every")
        report = type_safe(gold, direct, reranker)
    else:
        if not args.firewall_actions or not args.inference:
            parser.error("Every requires --inference and --firewall-actions from build_every.py")
        report = every(
            gold,
            read(args.inference),
            direct,
            reranker,
            json.loads(args.firewall_actions.read_text()),
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
