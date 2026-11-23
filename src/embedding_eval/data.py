"""Load both pair datasets and keep the comparison free of leakage.

The new dataset is split by SRD section. The original dataset has a single train split
and was cut from a different chunking, so its passages don't line up with the new
sections. `split_original` matches each original passage to the new sections by word
5-gram overlap. Original pairs whose passage overlaps a held-out (validation or test)
section are removed from training, so neither model trains on text it is tested on.
Original pairs that clearly match a test section become a second test set, in the
original dataset's question style.
"""

import json
import re
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

ORIGINAL_DATASET = "galgilor/dnd-5e-srd-anchor-positive-pairs"
ORIGINAL_REVISION = "27dfa295e70a42728376955ed792aac756dc58a8"
SPLITS = ("train", "validation", "test")
SHINGLE_SIZE = 5
LEAK_OVERLAP = 0.3  # Share of a passage's 5-grams in a held-out section that drops it
MATCH_OVERLAP = 0.6  # Share needed to use an original pair as a test question


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def load_new_dataset(directory: Path) -> dict[str, list[dict]]:
    """Load train, validation and test from a folder written by assemble_mnlr_dataset."""
    return {split: load_jsonl(directory / f"{split}.jsonl") for split in SPLITS}


def load_original_dataset() -> list[dict]:
    """Download the original pairs at the revision the comparison was designed on."""
    from datasets import load_dataset

    dataset = load_dataset(ORIGINAL_DATASET, revision=ORIGINAL_REVISION, split="train")

    return [{"anchor": row["anchor"], "positive": row["positive"]} for row in dataset]


def _shingles(text: str) -> set[tuple[str, ...]]:
    """Word 5-grams of lowercased text without markup, so formatting doesn't count."""
    words = re.findall(r"[a-z0-9]+", text.lower())

    return {tuple(words[i : i + SHINGLE_SIZE]) for i in range(len(words) - SHINGLE_SIZE + 1)}


class SectionIndex:
    """Find which SRD sections contain a piece of text.

    Args:
        passages: Pairs of (passage text, section) from the new dataset.
    """

    def __init__(self, passages: list[tuple[str, str]]):
        self._sections_by_shingle: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for text, section in passages:
            for shingle in _shingles(text):
                self._sections_by_shingle[shingle].add(section)

    def overlaps(self, text: str) -> dict[str, float]:
        """Return each section's share of the text's 5-grams. Empty for text under 5 words."""
        shingles = _shingles(text)
        if not shingles:
            return {}
        counts = Counter(
            section
            for shingle in shingles
            for section in self._sections_by_shingle.get(shingle, ())
        )

        return {section: count / len(shingles) for section, count in counts.items()}


@dataclass
class OriginalSplit:
    """The original dataset after leakage removal."""

    train: list[dict]
    test: list[dict] = field(default_factory=list)  # anchor, breadcrumb
    removed: int = 0
    unmatched: int = 0


def split_original(original: list[dict], new_splits: dict[str, list[dict]]) -> OriginalSplit:
    """Drop original pairs overlapping held-out sections; keep clear test matches as tests."""
    index = SectionIndex(
        [(row["positive"], row["breadcrumb"]) for rows in new_splits.values() for row in rows]
    )
    held_out = {
        row["breadcrumb"] for split in ("validation", "test") for row in new_splits[split]
    }
    test_sections = {row["breadcrumb"] for row in new_splits["test"]}

    result = OriginalSplit(train=[])
    for row in original:
        overlaps = index.overlaps(row["positive"])
        if not overlaps:
            result.unmatched += 1
        leak = max(
            (share for section, share in overlaps.items() if section in held_out), default=0
        )
        if leak < LEAK_OVERLAP:
            result.train.append(row)
            continue
        result.removed += 1
        # Sorted first so a tie between sections resolves the same way on every run.
        best, share = max(sorted(overlaps.items()), key=lambda item: item[1])
        if best in test_sections and share >= MATCH_OVERLAP:
            result.test.append({"anchor": row["anchor"], "breadcrumb": best})

    return result


def corpus_passages(new_splits: dict[str, list[dict]]) -> list[tuple[str, str]]:
    """Return each distinct passage of the new dataset with its section, in stable order."""
    seen: dict[str, str] = {}
    for split in SPLITS:
        for row in new_splits[split]:
            seen.setdefault(row["positive"], row["breadcrumb"])

    return sorted(seen.items(), key=lambda item: (item[1], item[0]))
