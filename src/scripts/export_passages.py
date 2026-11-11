"""Export the SRD graph's passages as batch files for question generation.

Passages are shuffled with a fixed seed so every batch mixes entity types, then written
as JSONL files of --batch-size passages to <output>/batches/batch_000.jsonl and so on.
Each line has `passage_id`, `label`, `entity` and `positive`.

With --entities-only, plain sections are left out and the files go to
<output>/hard_batches/. That set is for a second pass of harder questions (see
src/templates/generate_mnlr_hard_questions.md), which works best on named entities.

Example Usage:
    python -m src.scripts.export_passages
    python -m src.scripts.export_passages --batch-size 25 --output data/mnlr
    python -m src.scripts.export_passages --entities-only --batch-size 40
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.graphrag import DEFAULT_MARKDOWN
from src.graphrag.graph import SECTION
from src.graphrag.graph import build_graph
from src.mnlr.passages import extract_passages


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SRD passages in batch files.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--output", type=Path, default=Path("data/mnlr"))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--entities-only", action="store_true")
    args = parser.parse_args()

    passages = extract_passages(build_graph(args.markdown))
    if args.entities_only:
        passages = [passage for passage in passages if passage.label != SECTION]
    random.Random(args.seed).shuffle(passages)

    batches = args.output / ("hard_batches" if args.entities_only else "batches")
    batches.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(passages), args.batch_size):
        path = batches / f"batch_{start // args.batch_size:03d}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for passage in passages[start : start + args.batch_size]:
                row = {
                    "passage_id": passage.passage_id,
                    "label": passage.label,
                    "entity": passage.entity,
                    "positive": passage.positive,
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    labels = Counter(passage.label for passage in passages)
    print(f"Wrote {len(passages):,} passages to {len(list(batches.glob('*.jsonl')))} files.")
    print(", ".join(f"{label}: {count:,}" for label, count in labels.most_common()))


if __name__ == "__main__":
    main()
