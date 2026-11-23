"""Compare embedders fine-tuned on the original and the new SRD pair datasets.

Fine-tunes the same base model twice with the same settings, once on each dataset's
training pairs, and scores section retrieval for:

    - bm25: keyword search, no model.
    - base: the base model without fine-tuning.
    - original: fine-tuned on the original dataset, after leakage removal.
    - new: fine-tuned on the new dataset's train split.

Each is scored on up to three question sets, against the same corpus (every passage of
the new dataset):

    - new_test: the new dataset's test split.
    - original_style: original-dataset questions about the same test sections. They were
      removed from the original training set, and test whether the new model only wins
      on questions written in its own style.
    - manual: hand-written questions from --manual, one JSON object per line with
      `anchor` and `breadcrumb` (the gold section's header path, as in the dataset).

Leakage: original pairs whose passage overlaps a validation or test section of the new
dataset are dropped from its training set (see src.embedding_eval.data).

Writes results.json and results.md to --output, and the fine-tuned models to
<output>/models/<name>-seed<seed>.

Example Usage:
    uv run python -m src.scripts.compare_embedders
    uv run python -m src.scripts.compare_embedders \\
        --manual data/mnlr/manual_questions.jsonl \\
        --base-model BAAI/bge-base-en-v1.5 --seeds 0 1 2
    # Smoke test on a CPU
    uv run python -m src.scripts.compare_embedders --max-train 200 --epochs 1
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from src.embedding_eval.data import corpus_passages
from src.embedding_eval.data import load_jsonl
from src.embedding_eval.data import load_new_dataset
from src.embedding_eval.data import load_original_dataset
from src.embedding_eval.data import split_original
from src.embedding_eval.metrics import bm25_scores
from src.embedding_eval.metrics import embedding_scores
from src.embedding_eval.metrics import rank_sections
from src.embedding_eval.metrics import score_rankings
from src.embedding_eval.training import fine_tune


def _evaluate(scores_for, eval_sets: dict, passages: list[str], sections: list[str]) -> dict:
    """Score one retriever on every question set."""
    results = {}
    for name, questions in eval_sets.items():
        scores = scores_for([q["anchor"] for q in questions], passages)
        rankings = rank_sections(scores, sections)
        results[name] = score_rankings(rankings, [q["breadcrumb"] for q in questions])

    return results


def _markdown(results: dict, sizes: dict) -> str:
    """Render one table per question set, with the mean over seeds for fine-tuned models."""
    lines = []
    for set_name, size in sizes.items():
        lines += [f"## {set_name} ({size:,} questions)", ""]
        metrics = list(next(iter(results.values()))[set_name])
        lines += [
            "| model | " + " | ".join(metrics) + " |",
            "|---" * (len(metrics) + 1) + "|",
        ]
        groups: dict[str, list[dict]] = {}
        for model, by_set in results.items():
            groups.setdefault(model.split("-seed")[0], []).append(by_set[set_name])
        for model, runs in groups.items():
            cells = []
            for metric in metrics:
                values = [run[metric] for run in runs]
                spread = f" ± {statistics.stdev(values):.3f}" if len(values) > 1 else ""
                cells.append(f"{statistics.mean(values):.3f}{spread}")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare embedders fine-tuned on each dataset."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/mnlr/dataset"))
    parser.add_argument("--manual", type=Path, help="Hand-written questions (JSONL).")
    parser.add_argument("--output", type=Path, default=Path("data/mnlr/comparison"))
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=16,
        help="Texts the model encodes at once. Lower it if memory runs out.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--match-size",
        action="store_true",
        help="Subsample the larger training set to the size of the smaller one.",
    )
    parser.add_argument("--max-train", type=int, help="Cap both training sets (smoke tests).")
    args = parser.parse_args()

    new_splits = load_new_dataset(args.dataset)
    original = split_original(load_original_dataset(), new_splits)
    corpus = corpus_passages(new_splits)
    passages = [text for text, _ in corpus]
    sections = [section for _, section in corpus]
    print(f"Corpus: {len(passages):,} passages from {len(set(sections)):,} sections.")
    print(
        f"Original dataset: {len(original.train) + original.removed:,} pairs, "
        f"{original.removed:,} removed for overlapping held-out sections, "
        f"{len(original.train):,} kept for training, "
        f"{original.unmatched:,} unmatched to any section."
    )

    eval_sets = {
        "new_test": [
            {"anchor": r["anchor"], "breadcrumb": r["breadcrumb"]} for r in new_splits["test"]
        ],
        "original_style": original.test,
    }
    if args.manual:
        manual = load_jsonl(args.manual)
        unknown = sorted({q["breadcrumb"] for q in manual} - set(sections))
        if unknown:
            raise SystemExit(
                f"Manual questions name sections not in the corpus: {unknown[:5]}"
            )
        eval_sets["manual"] = manual
    sizes = {name: len(questions) for name, questions in eval_sets.items() if questions}
    eval_sets = {name: questions for name, questions in eval_sets.items() if questions}
    print("Question sets: " + ", ".join(f"{name} {size:,}" for name, size in sizes.items()))

    results = {"bm25": _evaluate(bm25_scores, eval_sets, passages, sections)}

    from sentence_transformers import SentenceTransformer

    base = SentenceTransformer(args.base_model)
    results["base"] = _evaluate(
        lambda q, p: embedding_scores(base, q, p, args.batch_size),
        eval_sets,
        passages,
        sections,
    )

    for seed in args.seeds:
        training_sets = {
            "original": list(original.train),
            "new": [
                {"anchor": r["anchor"], "positive": r["positive"]}
                for r in new_splits["train"]
            ],
        }
        size = min(len(rows) for rows in training_sets.values()) if args.match_size else None
        if args.max_train:
            size = min(size or args.max_train, args.max_train)
        for name, rows in training_sets.items():
            if size is not None and len(rows) > size:
                rows = random.Random(seed).sample(rows, size)
            print(f"Fine-tuning '{name}' on {len(rows):,} pairs, seed {seed}.")
            model = fine_tune(
                args.base_model,
                rows,
                args.output / "models" / f"{name}-seed{seed}",
                args.epochs,
                args.batch_size,
                args.learning_rate,
                seed,
                args.mini_batch_size,
            )
            results[f"{name}-seed{seed}"] = _evaluate(
                lambda q, p: embedding_scores(model, q, p, args.batch_size),
                eval_sets,
                passages,
                sections,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "results.json").write_text(
        json.dumps({"settings": settings, "sizes": sizes, "results": results}, indent=2)
    )
    report = _markdown(results, sizes)
    (args.output / "results.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
