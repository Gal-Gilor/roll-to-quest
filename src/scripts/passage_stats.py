"""Print the passage-size numbers stated in the MNLR dataset card.

Reads every distinct `positive` of the published dataset and counts its tokens with a
model's tokenizer, so the card's claims (median tokens, passages over the model's input
limit) can be checked or updated after the passages change.

Example Usage:
    uv run python -m src.scripts.passage_stats
    uv run python -m src.scripts.passage_stats --tokenizer BAAI/bge-base-en-v1.5 --limit 512
"""

import argparse
import json
import statistics
from pathlib import Path

SPLITS = ("train", "validation", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count tokens in the dataset's passages.")
    parser.add_argument("--dataset", type=Path, default=Path("data/mnlr/dataset"))
    parser.add_argument("--tokenizer", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--limit", type=int, default=512, help="The model's input limit.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    passages = {
        json.loads(line)["positive"]
        for split in SPLITS
        for line in (args.dataset / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = sorted(len(tokenizer(passage)["input_ids"]) for passage in passages)
    over = sum(count > args.limit for count in tokens)

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Passages: {len(tokens):,}")
    print(f"Longest passage: {max(len(p) for p in passages):,} characters")
    print(f"Tokens: median {statistics.median(tokens):g}, max {tokens[-1]:,}")
    print(f"Over {args.limit} tokens: {over:,}")


if __name__ == "__main__":
    main()
