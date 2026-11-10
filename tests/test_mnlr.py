import networkx as nx

from src.mnlr.assembly import apply_verdicts
from src.mnlr.assembly import split_by_node
from src.mnlr.passages import MAX_PASSAGE_CHARS
from src.mnlr.passages import Passage
from src.mnlr.passages import drop_near_duplicates
from src.mnlr.passages import entity_name
from src.mnlr.passages import extract_passages
from src.mnlr.passages import split_text
from src.mnlr.validation import Question
from src.mnlr.validation import filter_questions
from src.mnlr.validation import overlap
from src.mnlr.validation import problems

FIREBALL = Passage(
    passage_id="7-0",
    node_id="7",
    label="Spell",
    entity="Fireball",
    breadcrumb="Spells > Spell Descriptions > Fireball",
    text=(
        "A bright streak flashes to a point within range and blossoms into an explosion "
        "of flame. Each creature in a 20-foot-radius Sphere makes a Dexterity saving "
        "throw, taking 8d6 Fire damage on a failed save."
    ),
)


def _passage(passage_id: str, text: str) -> Passage:
    return FIREBALL.model_copy(update={"passage_id": passage_id, "text": text})


def _question(question_type: str, anchor: str, passage_id: str = "7-0") -> Question:
    return Question(passage_id=passage_id, question_type=question_type, anchor=anchor)


def test_entity_name_strips_level_prefix_and_tags():
    assert entity_name("Level 14: Overchannel") == "Overchannel"
    assert entity_name("Paralyzed [Condition]") == "Paralyzed"
    assert entity_name("Cook's Utensils (1 GP)") == "Cook's Utensils"
    assert entity_name("Fireball") == "Fireball"


def test_split_text_keeps_short_text_whole():
    assert split_text("One paragraph.\n\nAnother one.") == ["One paragraph.\n\nAnother one."]


def test_split_text_splits_on_paragraphs_then_lines():
    paragraph = "word " * 60  # 300 characters
    rows = "\n".join(f"| row {i} | {'x' * 90} |" for i in range(30))
    table = f"| Name | Value |\n|------|-------|\n{rows}"  # One long paragraph

    chunks = split_text(f"{paragraph}\n\n{paragraph}\n\n{table}", limit=700)

    assert all(len(chunk) <= 700 for chunk in chunks)
    assert chunks[0].count("word") == 120  # Both short paragraphs packed together
    assert all(chunk.startswith("| Name | Value |\n|------|") for chunk in chunks[1:])
    assert "".join(chunks).count("| row") == 30


def test_drop_near_duplicates_keeps_the_first_of_a_group():
    trait = "If the dragon fails a saving throw, it can choose to succeed instead of failing."
    passages = [
        _passage("1-0", f"Legendary Resistance (3/Day). {trait}"),
        _passage("2-0", f"Legendary Resistance (4/Day). {trait}"),
        _passage("3-0", "The dragon exhales fire in a 60-foot Cone and everything burns."),
    ]

    assert [p.passage_id for p in drop_near_duplicates(passages)] == ["1-0", "3-0"]


def test_extract_passages_splits_long_nodes_and_skips_empty_ones():
    graph = nx.DiGraph()
    long_text = "\n\n".join(f"Paragraph {i} " + "lorem ipsum dolor " * 30 for i in range(6))
    graph.add_node(
        "1", label="Monster", header="Lich", breadcrumb="Monsters > Lich", content=long_text
    )
    graph.add_node("2", label="Section", header="Combat", breadcrumb="Combat", content="")
    graph.add_node(
        "3", label="Section", header="Intro", breadcrumb="Intro", content="Too short."
    )

    passages = extract_passages(graph)

    assert len(passages) > 1
    assert {p.node_id for p in passages} == {"1"}
    assert all(len(p.text) <= MAX_PASSAGE_CHARS for p in passages)
    assert passages[0].passage_id == "1-0"
    assert passages[0].positive.startswith("Monsters > Lich\n\nParagraph 0")


def test_overlap_ignores_the_entity_name_and_stopwords():
    assert overlap("What does Fireball do?", FIREBALL) == 0.0
    assert overlap("How much Fire damage does Fireball deal?", FIREBALL) == 2 / 4
    assert overlap("Which blast scorches a crowd?", FIREBALL) == 0.0


def test_problems_accepts_good_questions_of_each_type():
    for question in [
        _question("lookup", "How much Fire damage does Fireball deal on a failed save?"),
        _question("paraphrase", "Can I dodge Fireball to avoid getting burned?"),
        _question(
            "descriptive", "Which magic blast scorches everyone standing in a wide circle?"
        ),
    ]:
        assert problems(question, FIREBALL) == []


def test_problems_flags_context_dependent_questions():
    question = _question("lookup", "What saving throw does this spell require?")

    assert "refers to the passage instead of naming its subject" in problems(
        question, FIREBALL
    )


def test_problems_checks_entity_name_by_question_type():
    named = _question("descriptive", "Which spell is like Fireball but bigger and louder?")
    unnamed = _question("lookup", "How wide is the explosion of flame in feet?")

    assert problems(named, FIREBALL) == ["descriptive question names the entity"]
    assert problems(unnamed, FIREBALL) == ["question does not name the entity"]


def test_problems_does_not_require_section_titles_in_questions():
    section = FIREBALL.model_copy(update={"label": "Section", "entity": "Rhythm of Play"})

    assert (
        problems(_question("lookup", "How wide is the blast of a fiery burst?"), section)
        == []
    )


def test_problems_flags_paraphrases_that_copy_the_passage():
    copied = _question(
        "paraphrase",
        "Does each creature in a Sphere make a Dexterity saving throw for Fireball?",
    )

    assert problems(copied, FIREBALL) == ["repeats too many words of the passage"]


def test_filter_questions_rejects_duplicates_and_unknown_passages():
    questions = [
        _question("lookup", "How much Fire damage does Fireball deal?"),
        _question("lookup", "how much fire damage does  Fireball deal?"),
        _question("lookup", "How far does Fireball reach?", passage_id="missing"),
    ]

    kept, rejected = filter_questions(questions, {"7-0": FIREBALL})

    assert [q.anchor for q in kept] == ["How much Fire damage does Fireball deal?"]
    assert rejected == {"duplicate question": 1, "unknown passage": 1}


def test_apply_verdicts_keeps_only_accepted_rows():
    rows = [{"anchor": "a"}, {"anchor": "b"}, {"anchor": "c"}]

    assert apply_verdicts(rows, {0: "accept", 1: "reject"}) == [{"anchor": "a"}]


def test_split_by_node_keeps_a_node_in_one_split():
    rows = [
        {"node_id": str(node), "anchor": f"{node}-{i}"}
        for node in range(40)
        for i in range(3)
    ]

    splits = split_by_node(rows, validation_share=0.1, test_share=0.2)

    nodes = {name: {row["node_id"] for row in split} for name, split in splits.items()}
    assert [len(nodes[name]) for name in ("train", "validation", "test")] == [28, 4, 8]
    assert not nodes["train"] & nodes["test"] and not nodes["train"] & nodes["validation"]
    assert sum(len(split) for split in splits.values()) == len(rows)
    assert splits == split_by_node(rows, validation_share=0.1, test_share=0.2)
