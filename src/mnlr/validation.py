"""Scripted checks for generated questions, before any model judges them."""

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from src.graphrag.retrieval import STOPWORDS
from src.mnlr.passages import Passage

QuestionType = Literal["lookup", "paraphrase", "descriptive", "table"]

MAX_OVERLAP = 0.75  # Share of a question's content words that may come from the passage
MIN_QUESTION_WORDS = 4
MAX_RANK = 50  # Ranks beyond this are reported as MAX_RANK + 1
HARD_RANK = 4  # A question whose node BM25 ranks below this needs a judge's verdict
EXACT_TYPES = ("lookup", "table")  # Questions about one fact may reuse the passage's terms
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)

# A question must make sense without the passage in front of the reader.
_CONTEXT_DEPENDENT = re.compile(
    r"\b(?:this|that|these|those|the (?:above|following|described|given|mentioned))\s+"
    r"(?:spell|creature|monster|item|feature|feat|class|subclass|condition|action|rule|"
    r"section|passage|text|table|ability|trait|background|species|tool|weapon|armor)s?\b"
    r"|\b(?:the|this) (?:passage|text|section|excerpt)\b|\baccording to\b",
    re.IGNORECASE,
)


class Question(BaseModel):
    """A generated question for one passage."""

    passage_id: str
    question_type: QuestionType
    anchor: str


def _content_words(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def overlap(question: str, passage: Passage) -> float:
    """Return the share of the question's content words that appear in the passage.

    Words of the entity's name don't count: a question has to name what it asks about.
    """
    name_words = set(_content_words(passage.entity))
    words = [word for word in _content_words(question) if word not in name_words]
    if not words:
        return 0.0
    passage_words = set(_content_words(passage.text))

    return sum(word in passage_words for word in words) / len(words)


def _names_entity(question: str, passage: Passage) -> bool:
    return passage.entity.lower() in question.lower()


def problems(question: Question, passage: Passage) -> list[str]:
    """Return why a question fails the scripted checks. Empty when it passes."""
    found = []
    anchor = question.anchor.strip()

    if len(anchor.split()) < MIN_QUESTION_WORDS:
        found.append("too short")
    if _CONTEXT_DEPENDENT.search(anchor):
        found.append("refers to the passage instead of naming its subject")

    # Plain sections have headers like "Rhythm of Play" that a player wouldn't say.
    if passage.label != "Section":
        if question.question_type == "descriptive" and _names_entity(anchor, passage):
            found.append("descriptive question names the entity")
        if question.question_type != "descriptive" and not _names_entity(anchor, passage):
            found.append("question does not name the entity")

    if question.question_type == "table" and len(_TABLE_ROW.findall(passage.text)) < 3:
        found.append("table question for a passage without a table")
    if question.question_type not in EXACT_TYPES and overlap(anchor, passage) > MAX_OVERLAP:
        found.append("repeats too many words of the passage")

    return found


class PassageRanker:
    """Rank the graph nodes for a question with BM25 over their passages.

    A question BM25 ranks its own node low for is either a good hard question, worded
    unlike its passage, or an ambiguous one that fits another passage as well. The rank
    alone can't tell them apart, so those questions go to a judge with the competing
    passage.
    """

    def __init__(self, passages: dict[str, Passage]):
        self._passages = list(passages.values())
        self._index = {passage.passage_id: i for i, passage in enumerate(self._passages)}
        self._bm25 = BM25Okapi([_content_words(p.positive) for p in self._passages])

    def rank(self, question: Question, max_rank: int = MAX_RANK) -> tuple[int, Passage]:
        """Return the rank of the question's node and the best passage of another node.

        The rank is 1 when no other node scores higher than the question's own node.
        It is capped at `max_rank + 1`, which means "not among the first `max_rank`".
        """
        scores = self._bm25.get_scores(_content_words(question.anchor))
        own_node = self._passages[self._index[question.passage_id]].node_id

        best_by_node: dict[str, tuple[float, Passage]] = {}
        for score, passage in zip(scores, self._passages):
            if score > best_by_node.get(passage.node_id, (-1.0, None))[0]:
                best_by_node[passage.node_id] = (score, passage)

        own_score = best_by_node.pop(own_node)[0]
        competitor = max(best_by_node.values(), key=lambda best: best[0])[1]
        better = sum(1 for score, _ in best_by_node.values() if score > own_score)

        return min(better + 1, max_rank + 1), competitor


def filter_questions(
    questions: list[Question], passages: dict[str, Passage]
) -> tuple[list[Question], Counter]:
    """Split questions into the ones that pass and a count of each problem found.

    A question is also rejected when its passage is unknown or its anchor was already
    seen: the same anchor with two positives would make one of them a false negative.
    """
    kept, rejected, seen = [], Counter(), set()
    for question in questions:
        passage = passages.get(question.passage_id)
        key = " ".join(question.anchor.lower().split())
        if passage is None:
            found = ["unknown passage"]
        elif key in seen:
            found = ["duplicate question"]
        else:
            found = problems(question, passage)

        if found:
            rejected.update(found)
        else:
            seen.add(key)
            kept.append(question)

    return kept, rejected
