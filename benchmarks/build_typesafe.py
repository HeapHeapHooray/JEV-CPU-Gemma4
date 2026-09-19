"""Rebuild the frozen 102-row TypeSafe subset from verified public case payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def parse_payload(path: Path):
    text = path.read_text().strip()
    prefix = "__VIEWER_DATA__("
    if not text.startswith(prefix):
        raise ValueError(f"Unexpected wrapper in {path}")
    payload, end = json.JSONDecoder().raw_decode(text[len(prefix) :])
    if text[len(prefix) + end :].strip() not in {")", ");"}:
        raise ValueError(f"Unexpected trailing content in {path}")
    return payload


def vector(raw, keys):
    values = [float(raw.get(key, 0)) for key in keys]
    if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("Invalid probability vector")
    total = sum(values)
    return [value / total for value in values]


def reference_vector(answer, keys):
    if answer.get("probabilities"):
        return vector(answer["probabilities"], keys)
    value = answer.get("value")
    key = str(value).lower() if isinstance(value, bool) else str(value)
    if key not in keys:
        raise ValueError("Reference value is outside the declared options")
    return [float(candidate == key) for candidate in keys]


def published_vector(answer, primitive, keys):
    if primitive == "noul":
        probability = float(answer["noul"])
        return [probability if key == "true" else 1 - probability for key in keys]
    return vector(answer["probabilities"], keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    manifest = [json.loads(line) for line in args.selection.read_text().splitlines() if line.strip()]
    selected = [row for row in manifest if row["source"] == "typesafe"]
    payloads = {}
    for workflow in {row["upstream"]["workflow"] for row in selected}:
        path = args.source_dir / f"typesafe-{workflow}-cases.js"
        payloads[workflow] = parse_payload(path)
    rows = []
    for item in selected:
        upstream = item["upstream"]
        workflow = upstream["workflow"]
        payload = payloads[workflow]
        # The frozen manifest hashes the human-readable parsed snapshot.
        parsed_hash = hashlib.sha256(json.dumps(payload, indent=2).encode()).hexdigest()
        if parsed_hash != upstream["snapshot_sha256"]:
            raise ValueError(f"Parsed {workflow} snapshot hash changed")
        evaluation = payload["eval"]
        case = evaluation["cases"][upstream["case_id"]]
        question = evaluation["questions"][upstream["question_index"]]
        document = evaluation["documents"][upstream["document_index"]]
        primitive = upstream["primitive"]
        criteria = question.get("criteria")
        if primitive == "noul":
            options = [
                {"id": key, "description": (criteria or {}).get(key, f"The proposition is {key}.")}
                for key in ("true", "false")
            ]
        elif primitive == "choice":
            options = [{"id": key, "description": value} for key, value in criteria.items()]
        else:
            raise ValueError("The frozen comparison excludes Score primitives")
        for option in options:
            option["description"] = option["id"] + ": " + option["description"]
        keys = [option["id"] for option in options]
        if keys != item["option_ids"]:
            raise ValueError(f"Option order changed for {item['id']}")
        references = case["reference_answers"][upstream["node"]][upstream["question_id"]]["sets"]
        distributions = [reference_vector(answer, keys) for answer in references]
        target = [sum(row[index] for row in distributions) / len(distributions) for index in range(len(keys))]
        maxima = [index for index, value in enumerate(target) if abs(value - max(target)) < 1e-12]
        if len(maxima) != 1:
            raise ValueError("Frozen TypeSafe rows require a unique reference argmax")
        published = {}
        for model_key, model_record in case["models"].items():
            for node in model_record["nodes"]:
                if (
                    node["node"] == upstream["node"]
                    and node.get("doc") == upstream["document_index"]
                    and node.get("questions", {}).get(upstream["question_id"]) == upstream["question_index"]
                    and upstream["question_id"] in node.get("answers", {})
                ):
                    answer = node["answers"][upstream["question_id"]]
                    published[model_key] = {
                        "model": model_record["model"],
                        "distribution": published_vector(answer, primitive, keys),
                    }
        if "typesafe" not in published:
            raise ValueError("Selected row lacks a published TypeSafe answer")
        rows.append(
            {
                "id": item["id"],
                "group_id": item["group_id"],
                "family": item["family"],
                "split": "external_typesafe_selected",
                "state": json.dumps(document, ensure_ascii=False, indent=2),
                "question": question["instructions"],
                "options": options,
                "label": maxima[0],
                "target_distribution": target,
                "primitive": primitive,
                "published_models": published,
                "provenance": upstream,
            }
        )
    if len(rows) != 102 or len({row["id"] for row in rows}) != 102:
        raise ValueError("Selection must rebuild exactly 102 unique TypeSafe rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print("wrote 102 rows")


if __name__ == "__main__":
    main()
