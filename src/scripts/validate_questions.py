"""Run the scripted checks on generated questions and write the ones that pass.

Reads every JSONL file in <data>/generated/ (lines with `passage_id`, `question_type`
and `anchor`), checks each question against its passage, and writes the survivors with
their positives to <data>/validated.jsonl. It also writes them in files of --judge-size
rows to <data>/to_judge/ for the model judge, where `row` is the line number in
validated.jsonl.

Example Usage:
    python -m src.scripts.validate_questions
    python -m src.scripts.validate_questions --data data/mnlr
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from src.graphrag import DEFAULT_MARKDOWN
from src.graphrag.graph import build_graph
from src.mnlr.passages import extract_passages
from src.mnlr.validation import Question
from src.mnlr.validation import filter_questions
from src.mnlr.validation import overlap


def main() -> None:
    parser = argparse.ArgumentParser(description="Check generated questions.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--data", type=Path, default=Path("data/mnlr"))
    parser.add_argument("--judge-size", type=int, default=50)
    args = parser.parse_args()

    passages = {p.passage_id: p for p in extract_passages(build_graph(args.markdown))}

    questions, malformed = [], 0
    for path in sorted((args.data / "generated").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                questions.append(Question.model_validate_json(line))
            except ValidationError:
                malformed += 1

    kept, rejected = filter_questions(questions, passages)

    rows = []
    for question in kept:
        passage = passages[question.passage_id]
        rows.append(
            {
                "anchor": question.anchor.strip(),
                "positive": passage.positive,
                "node_id": passage.node_id,
                "passage_id": passage.passage_id,
                "label": passage.label,
                "breadcrumb": passage.breadcrumb,
                "question_type": question.question_type,
            }
        )

    output = args.data / "validated.jsonl"
    with output.open("w", encoding="utf-8") as file:
        file.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    to_judge = args.data / "to_judge"
    to_judge.mkdir(exist_ok=True)
    for stale in to_judge.glob("*.jsonl"):
        stale.unlink()
    for start in range(0, len(rows), args.judge_size):
        path = to_judge / f"batch_{start // args.judge_size:03d}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for index in range(start, min(start + args.judge_size, len(rows))):
                row = {
                    "row": index,
                    "question_type": rows[index]["question_type"],
                    "anchor": rows[index]["anchor"],
                    "positive": rows[index]["positive"],
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Read {len(questions):,} questions ({malformed} malformed lines skipped).")
    print(f"Kept {len(kept):,}, wrote {output}.")
    for problem, count in rejected.most_common():
        print(f"  rejected, {problem}: {count:,}")

    by_type = Counter(question.question_type for question in kept)
    for question_type, count in by_type.most_common():
        scores = [
            overlap(q.anchor, passages[q.passage_id])
            for q in kept
            if q.question_type == question_type
        ]
        print(f"  {question_type}: {count:,} kept, mean overlap {sum(scores) / count:.2f}")


if __name__ == "__main__":
    main()
