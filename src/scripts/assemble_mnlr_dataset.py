"""Build the final MNLR dataset from validated questions and judge verdicts.

Reads <data>/validated.jsonl and the verdict files in <data>/judged/ (lines with `row`,
`verdict` and `reason`), keeps the accepted pairs, and writes train.jsonl,
validation.jsonl and test.jsonl to <data>/dataset/. The split is by node.

Example Usage:
    python -m src.scripts.assemble_mnlr_dataset
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.mnlr.assembly import apply_verdicts
from src.mnlr.assembly import split_by_node


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the MNLR dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/mnlr"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lines = (args.data / "validated.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]

    verdicts: dict[int, str] = {}
    reasons: Counter = Counter()
    for path in sorted((args.data / "judged").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                verdict = json.loads(line)
                verdicts[int(verdict["row"])] = verdict["verdict"]
            except (ValueError, KeyError, TypeError):
                continue
            if verdict["verdict"] != "accept":
                reasons[verdict.get("reason") or "no reason"] += 1

    accepted = apply_verdicts(rows, verdicts)
    judged = len(set(verdicts) & set(range(len(rows))))
    print(f"{len(rows):,} validated pairs: {len(accepted):,} accepted, ", end="")
    print(f"{judged - len(accepted):,} rejected, {len(rows) - judged:,} not judged.")
    for reason, count in reasons.most_common():
        print(f"  rejected, {reason}: {count:,}")

    output = args.data / "dataset"
    output.mkdir(exist_ok=True)
    for split, split_rows in split_by_node(accepted, seed=args.seed).items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows)
        types = Counter(row["question_type"] for row in split_rows)
        nodes = len({row["node_id"] for row in split_rows})
        print(f"{split}: {len(split_rows):,} pairs from {nodes:,} nodes, {dict(types)}")


if __name__ == "__main__":
    main()
