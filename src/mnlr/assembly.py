"""Apply the judge's verdicts and split the pairs into train, validation and test."""

import random
from collections import defaultdict

from src.mnlr.validation import HARD_RANK


def apply_verdicts(
    rows: list[dict], verdicts: dict[int, str], hard_rank: int = HARD_RANK
) -> list[dict]:
    """Keep the rows BM25 ranks well, and the hard rows a judge accepted.

    A hard row without an "accept" verdict is dropped: it may be ambiguous.
    """
    return [
        row
        for index, row in enumerate(rows)
        if row["bm25_rank"] <= hard_rank or verdicts.get(index) == "accept"
    ]


def split_by_section(
    rows: list[dict], validation_share: float = 0.05, test_share: float = 0.1, seed: int = 0
) -> dict[str, list[dict]]:
    """Split rows so that all pairs of an SRD section land in the same split.

    Rows are grouped by `breadcrumb`, the path of headers above the passage. A model is
    then tested on sections it never saw a question about.
    """
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_section[row["breadcrumb"]].append(row)

    sections = sorted(by_section)
    random.Random(seed).shuffle(sections)
    num_test = round(len(sections) * test_share)
    num_validation = round(len(sections) * validation_share)
    section_splits = {
        "test": sections[:num_test],
        "validation": sections[num_test : num_test + num_validation],
        "train": sections[num_test + num_validation :],
    }

    return {
        split: [row for section in split_sections for row in by_section[section]]
        for split, split_sections in section_splits.items()
    }
