"""Rebuild Every's 204 inference rows and 154 author-labeled rows without executing its archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--experiments", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must be new")
    args.output_dir.mkdir(parents=True)
    data = json.loads(args.experiments.read_text())
    wanted = {
        row["id"]
        for line in args.selection.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row["source"] == "every"
    }
    original = []
    with zipfile.ZipFile(args.archive) as archive:
        script = archive.read("scripts/run-experiments.mjs").decode()
        policy = json.loads(re.search(r'const SUPPORT_POLICY = ("[^\n]+");', script).group(1))
        for experiment in data["experiments"]:
            name = experiment["id"]
            if name == "judge-grid":
                cells = {(cell["output"], cell["question"]): cell for cell in experiment["typesafe"]["cells"]}
                for document in experiment["documents"]:
                    for question_id, question in experiment["questions"].items():
                        original.append(
                            {
                                "id": f"{name}/{document['id']}/{question_id}",
                                "experiment": name,
                                "source_item": document["id"],
                                "question_id": question_id,
                                "document": {"policy": policy, "candidate_response": document["response"]},
                                "question": question,
                                "expected": bool(document["expected"][question_id]),
                                "recorded_typesafe": cells[document["id"], question_id],
                            }
                        )
            if name in {"code-rag", "company-brain"}:
                folder = "code-repo" if name == "code-rag" else "company-brain"
                for query in experiment["grid"]:
                    for cell in query["cells"]:
                        original.append(
                            {
                                "id": f"{name}/{cell['document']}/{query['id']}",
                                "experiment": name,
                                "source_item": cell["document"],
                                "question_id": query["id"],
                                "document": archive.read(f"fixtures/{folder}/{cell['document']}").decode(),
                                "question": query["question"],
                                "expected": cell["relevant"],
                                "recorded_typesafe": cell,
                            }
                        )
    options = [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}]
    inference, gold = [], []
    for row in original:
        item = {
            "id": "every/" + row["id"],
            "group_id": f"every/{row['experiment']}/{row['source_item']}",
            "family": "every_" + row["experiment"],
            "split": "external_test",
            "state": row["document"],
            "question": row["question"],
            "options": options,
            "provenance": {
                "source": "Every Parallel Judgment Lab",
                "experiment": row["experiment"],
                "question_id": row["question_id"],
                "source_item": row["source_item"],
                "label_source": "Original author label",
                "adaptation": "Native Noul replaced with binary Yes/No Choice; state/question unchanged",
            },
        }
        inference.append(item)
        gold.append({**item, "label": 0 if row["expected"] else 1})
    firewall = next(experiment for experiment in data["experiments"] if experiment["id"] == "action-firewall")
    for item in firewall["rows"]:
        for question_id, question in firewall["questions"].items():
            inference.append(
                {
                    "id": f"every/action-firewall/{item['id']}/{question_id}",
                    "group_id": f"every/action-firewall/{item['id']}",
                    "family": "every_action-firewall",
                    "split": "external_test",
                    "state": item["text"],
                    "question": question,
                    "options": options,
                    "provenance": {
                        "source": "Every Parallel Judgment Lab",
                        "experiment": "action-firewall",
                        "question_id": question_id,
                        "source_item": item["id"],
                        "label_source": "No per-signal labels; evaluate composed action only",
                        "adaptation": "Native Noul replaced with binary Yes/No Choice; state/question unchanged",
                    },
                }
            )
    if len(inference) != 204 or len(gold) != 154 or {row["id"] for row in inference} != wanted:
        raise ValueError("Rebuilt Every rows differ from the frozen 204-row selection")
    (args.output_dir / "inference204.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inference)
    )
    (args.output_dir / "gold154.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gold)
    )
    (args.output_dir / "firewall-actions.json").write_text(
        json.dumps(
            {
                "expected_actions": {row["id"]: row["expected"] for row in firewall["rows"]},
                "label_source": "Original author intended actions; no per-signal truth labels",
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote 204 inference rows, 154 labeled rows, and firewall actions")


if __name__ == "__main__":
    main()
