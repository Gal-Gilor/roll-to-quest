"""Scripted checks for generated questions, before any model judges them."""

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from src.graphrag.retrieval import STOPWORDS
from src.mnlr.passages import Passage

QuestionType = Literal["lookup", "paraphrase", "descriptive"]

MAX_OVERLAP = 0.6  # Share of a question's content words that may come from the passage
MIN_QUESTION_WORDS = 4

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

    if question.question_type != "lookup" and overlap(anchor, passage) > MAX_OVERLAP:
        found.append("repeats too many words of the passage")

    return found


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
