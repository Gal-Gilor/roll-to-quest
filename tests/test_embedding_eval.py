import numpy as np

from src.embedding_eval.data import SectionIndex
from src.embedding_eval.data import corpus_passages
from src.embedding_eval.data import split_original
from src.embedding_eval.metrics import bm25_scores
from src.embedding_eval.metrics import rank_sections
from src.embedding_eval.metrics import score_rankings

FIREBALL = (
    "Spells > Fireball\n\nA bright streak flashes from you to a point you choose within "
    "range and then blossoms with a low roar into a fiery explosion."
)
GHOUL = (
    "Monsters > Ghoul\n\nThe ghoul makes two claw attacks. A creature hit by the claws "
    "must succeed on a Constitution saving throw or have the Paralyzed condition."
)
TORCH = (
    "Equipment > Torch\n\nA torch burns for 1 hour, casting Bright Light in a 20-foot "
    "radius and Dim Light for an additional 20 feet."
)


def _row(anchor: str, positive: str) -> dict:
    return {"anchor": anchor, "positive": positive, "breadcrumb": positive.split("\n")[0]}


NEW_SPLITS = {
    "train": [_row("How long does a torch burn?", TORCH)],
    "validation": [_row("What happens when a ghoul claws me?", GHOUL)],
    "test": [
        _row("How big is the Fireball explosion?", FIREBALL),
        _row("Which spell makes a fiery blast?", FIREBALL),
    ],
}


def test_section_index_ignores_markup_and_reports_shares():
    index = SectionIndex([(FIREBALL, "Spells > Fireball"), (GHOUL, "Monsters > Ghoul")])

    overlaps = index.overlaps("**A bright streak** flashes from you to a point you choose")

    assert overlaps == {"Spells > Fireball": 1.0}
    assert index.overlaps("too short") == {}


def test_split_original_drops_held_out_overlap_and_keeps_test_matches():
    original = [
        # Reformatted copy of the test passage: dropped, and kept as a test question.
        {
            "anchor": "What does the spell do?",
            "positive": "A bright streak flashes from you to a point you choose within range "
            "and then blossoms with a low roar into a fiery explosion.",
        },
        # Validation passage: dropped, but not a test question.
        {
            "anchor": "Ghoul claws?",
            "positive": "The ghoul makes two claw attacks. A creature hit by the claws must "
            "succeed on a Constitution saving throw",
        },
        # Train passage: kept for training.
        {
            "anchor": "Torch light?",
            "positive": "A torch burns for 1 hour, casting Bright Light",
        },
        # No match anywhere: kept for training and counted.
        {
            "anchor": "Hirelings?",
            "positive": "Skilled hirelings include anyone hired to do work",
        },
    ]

    split = split_original(original, NEW_SPLITS)

    assert [row["anchor"] for row in split.train] == ["Torch light?", "Hirelings?"]
    assert split.test == [
        {"anchor": "What does the spell do?", "breadcrumb": "Spells > Fireball"}
    ]
    assert split.removed == 2
    assert split.unmatched == 1


def test_corpus_passages_are_distinct_with_their_sections():
    corpus = corpus_passages(NEW_SPLITS)

    assert len(corpus) == 3
    assert ("Spells > Fireball" in dict(corpus).values()) and corpus == sorted(
        corpus, key=lambda item: (item[1], item[0])
    )


def test_rank_sections_dedupes_passages_of_one_section():
    scores = np.array([[0.9, 0.8, 0.1]])

    assert rank_sections(scores, ["A", "A", "B"]) == [["A", "B"]]


def test_score_rankings_reports_recall_and_mrr():
    result = score_rankings([["A", "B"], ["B", "A"], ["C"]], ["A", "A", "A"])

    assert result["recall@1"] == 1 / 3
    assert result["recall@5"] == 2 / 3
    assert result["mrr@10"] == (1 + 1 / 2) / 3


def test_bm25_scores_rank_the_matching_passage_first():
    scores = bm25_scores(["ghoul claws paralyzed", "torch light"], [FIREBALL, GHOUL, TORCH])

    assert scores[0].argmax() == 1
    assert scores[1].argmax() == 2
