"""Build the final MNLR dataset from the validated questions and the judge's verdicts.

Reads <data>/validated.jsonl and the verdict files in <data>/judged/ (lines with `row`
and `verdict`). Questions BM25 ranks well are kept as they are. Hard questions are kept
only with an "accept" verdict. Files in <data>/judged/ are read in name order and a
later verdict for a row replaces an earlier one, so a file like zz_adjudicated.jsonl
settles the rows the judge marked "unsure".

Writes train.jsonl, validation.jsonl and test.jsonl to <data>/dataset/, with the dataset
card as README.md. The split is by node, so all pairs of a node land in the same split.

To publish the folder:
    huggingface-cli upload <user>/<name> data/mnlr/dataset . --repo-type dataset

Example Usage:
    python -m src.scripts.assemble_mnlr_dataset
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from src.mnlr.assembly import apply_verdicts
from src.mnlr.assembly import split_by_node
from src.mnlr.validation import HARD_RANK

DATASET_CARD = Path(__file__).parents[1] / "mnlr" / "DATASET_CARD.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the MNLR dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/mnlr"))
    parser.add_argument("--hard-rank", type=int, default=HARD_RANK)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lines = (args.data / "validated.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]

    verdicts: dict[int, str] = {}
    for path in sorted((args.data / "judged").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                verdict = json.loads(line)
                verdicts[int(verdict["row"])] = str(verdict["verdict"])
            except (ValueError, KeyError, TypeError):
                continue

    hard = [index for index, row in enumerate(rows) if row["bm25_rank"] > args.hard_rank]
    outcomes = Counter(verdicts.get(index, "not judged") for index in hard)
    accepted = apply_verdicts(rows, verdicts, args.hard_rank)
    print(f"{len(rows):,} validated pairs, {len(hard):,} hard: {dict(outcomes)}")
    print(f"{len(accepted):,} pairs kept.")

    output = args.data / "dataset"
    output.mkdir(exist_ok=True)
    shutil.copy(DATASET_CARD, output / "README.md")
    for split, split_rows in split_by_node(accepted, seed=args.seed).items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows)
        types = Counter(row["question_type"] for row in split_rows)
        nodes = len({row["node_id"] for row in split_rows})
        hard_kept = sum(row["bm25_rank"] > args.hard_rank for row in split_rows)
        print(
            f"{split}: {len(split_rows):,} pairs from {nodes:,} nodes, "
            f"{hard_kept:,} hard, {dict(types)}"
        )


if __name__ == "__main__":
    main()
