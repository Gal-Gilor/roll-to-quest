"""Build the final MNLR dataset from the validated questions.

Reads <data>/validated.jsonl and writes train.jsonl, validation.jsonl and test.jsonl to
<data>/dataset/. The split is by node, so all pairs of a node land in the same split.

Example Usage:
    python -m src.scripts.assemble_mnlr_dataset
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.mnlr.assembly import split_by_node


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the MNLR dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/mnlr"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lines = (args.data / "validated.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]

    output = args.data / "dataset"
    output.mkdir(exist_ok=True)
    for split, split_rows in split_by_node(rows, seed=args.seed).items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows)
        types = Counter(row["question_type"] for row in split_rows)
        nodes = len({row["node_id"] for row in split_rows})
        print(f"{split}: {len(split_rows):,} pairs from {nodes:,} nodes, {dict(types)}")


if __name__ == "__main__":
    main()
