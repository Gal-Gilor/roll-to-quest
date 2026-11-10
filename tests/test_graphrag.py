import json
import os
from pathlib import Path
from textwrap import dedent

import networkx as nx
import pytest

from src.graphrag.evaluation import GoldIndex
from src.graphrag.evaluation import score
from src.graphrag.graph import CONTAINS
from src.graphrag.graph import REFERENCES
from src.graphrag.graph import build_graph
from src.graphrag.retrieval import GraphRetriever

REPO_ROOT = Path(__file__).parents[1]

MARKDOWN = dedent(
    """\
    # Classes
    ## Wizard
    | Attribute       | Description  |
    |-----------------|--------------|
    | Primary Ability | Intelligence |
    ### Wizard Subclass: Evoker
    #### Level 3: Potent Cantrip
    Your cantrips hit harder.
    # Spells
    ## Spell Descriptions
    ### Fireball
    Level 3 Evocation (Wizard)

    Each creature in a Sphere makes a Dexterity saving throw, taking 8d6 Fire damage.
    ### Hold Person
    Level 2 Enchantment (Wizard)

    The target has the Paralyzed condition.
    ### Darkvision
    Level 2 Transmutation (Wizard)

    A willing creature can see in the dark.
    # Rules Glossary
    ## Rules Definitions
    ### Sphere [Area of Effect]
    A Sphere extends from a point of origin.
    ### Paralyzed [Condition]
    You can't move or speak.
    ### Poisoned [Condition]
    You have Disadvantage on attack rolls.
    # Monsters
    ## Monsters A-Z
    ### Mage
    #### Mage
    *Medium or Small Humanoid (Wizard), Neutral Evil*

    **Skills** Arcana +6
    **Resistances** Cold
    **Immunities** Poison; Poisoned
    **Senses** Darkvision 60 ft.
    ##### Actions
    **Spellcasting.** The mage casts one of the following spells:

    - 2/Day: Fireball
    # Magic Items
    ## Magic Items A-Z
    ### Wand of Fireballs
    *Wand, Very Rare (Requires Attunement by a Spellcaster)*

    You can cast Fireball from the wand.
    """
)


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    path = tmp_path_factory.mktemp("srd") / "doc.md"
    path.write_text(MARKDOWN, encoding="utf-8")

    return build_graph(path)


def _node(graph, label, header):
    return next(
        node
        for node, data in graph.nodes(data=True)
        if data["label"] == label and data["header"] == header
    )


def _edges(graph, label, header):
    """Return the outgoing edges of a node as {(type, target header)}."""
    node = _node(graph, label, header)

    return {
        (data["type"], graph.nodes[target]["header"])
        for _, target, data in graph.out_edges(node, data=True)
    }


def test_sections_are_labeled_by_their_place_in_the_document(graph):
    for label, header in [
        ("Class", "Wizard"),
        ("Subclass", "Wizard Subclass: Evoker"),
        ("Feature", "Level 3: Potent Cantrip"),
        ("Spell", "Fireball"),
        ("Condition", "Paralyzed [Condition]"),
        ("Monster", "Mage"),
        ("MagicItem", "Wand of Fireballs"),
        ("Section", "Sphere [Area of Effect]"),
        ("DamageType", "Fire"),
    ]:
        assert _node(graph, label, header) is not None


def test_entity_absorbs_its_subsections(graph):
    mage = graph.nodes[_node(graph, "Monster", "Mage")]

    assert "##### Actions" in mage["content"]
    assert not any(data["header"] == "Actions" for _, data in graph.nodes(data=True))


def test_class_structure_edges(graph):
    assert ("HAS_SUBCLASS", "Wizard Subclass: Evoker") in _edges(graph, "Class", "Wizard")
    assert ("USES_ABILITY", "Intelligence") in _edges(graph, "Class", "Wizard")
    assert _edges(graph, "Subclass", "Wizard Subclass: Evoker") == {
        ("HAS_FEATURE", "Level 3: Potent Cantrip")
    }
    assert (CONTAINS, "Fireball") in _edges(graph, "Section", "Spell Descriptions")


def test_spell_edges(graph):
    assert _edges(graph, "Spell", "Fireball") == {
        ("BELONGS_TO_SCHOOL", "Evocation"),
        ("CAST_BY", "Wizard"),
        ("DEALS_DAMAGE", "Fire"),
        ("REQUIRES_SAVE", "Dexterity"),
        (REFERENCES, "Sphere [Area of Effect]"),
    }
    assert ("INFLICTS_CONDITION", "Paralyzed [Condition]") in _edges(
        graph, "Spell", "Hold Person"
    )


def test_monster_edges(graph):
    assert _edges(graph, "Monster", "Mage") == {
        ("HAS_SIZE", "Medium"),
        ("HAS_SIZE", "Small"),
        ("HAS_TYPE", "Humanoid"),
        ("HAS_ALIGNMENT", "Neutral Evil"),
        ("PROFICIENT_IN", "Arcana"),
        ("RESISTANT_TO", "Cold"),
        ("IMMUNE_TO_DAMAGE", "Poison"),
        ("IMMUNE_TO_CONDITION", "Poisoned [Condition]"),
        ("CAN_CAST", "Fireball"),
        (REFERENCES, "Wizard"),
    }


def test_spell_name_outside_a_casting_line_is_not_a_cast_spell(graph):
    # "Senses Darkvision 60 ft." is the sense, not the Darkvision spell.
    assert "Darkvision" not in {header for _, header in _edges(graph, "Monster", "Mage")}


def test_magic_item_edges(graph):
    assert _edges(graph, "MagicItem", "Wand of Fireballs") == {
        ("HAS_RARITY", "Very Rare"),
        ("GRANTS_SPELL", "Fireball"),
    }


def test_srd_edge_types_are_defined_in_relationships_json():
    definitions = json.loads(
        (REPO_ROOT / "src/extraction/relationships.json").read_text(encoding="utf-8")
    )["relationships"]
    defined = {
        (source, relationship["type"], relationship["target"])
        for source, relationships in definitions.items()
        for relationship in relationships
    }
    graph = build_graph(REPO_ROOT / "srd_extraction.md")

    used = {
        (graph.nodes[source]["label"], data["type"], graph.nodes[target]["label"])
        for source, target, data in graph.edges(data=True)
        if data["type"] not in (CONTAINS, REFERENCES)
    }

    assert used - defined == set()


def test_retrieve_returns_seed_then_graph_neighbors(graph):
    retriever = GraphRetriever(graph)

    node_ids = retriever.retrieve("What does Fireball do?", num_seeds=1)
    headers = [graph.nodes[node_id]["header"] for node_id in node_ids]

    assert headers[0] == "Fireball"
    # Neither is named in the question; both come from the graph.
    assert "Mage" in headers
    assert "Sphere [Area of Effect]" in headers
    assert "Hold Person" not in headers


def test_retrieve_returns_nothing_for_unknown_terms(graph):
    assert GraphRetriever(graph).retrieve("xyzzy") == []


def test_format_context_lists_relationships_in_both_directions(graph):
    context = GraphRetriever(graph).format_context([_node(graph, "Spell", "Fireball")])

    assert context.startswith("## [Spell] Spells > Spell Descriptions > Fireball\n")
    assert "- DEALS_DAMAGE: Fire" in context
    assert "- <- CAN_CAST: Mage" in context
    assert REFERENCES not in context


def test_format_context_renders_concepts_from_edges_and_skips_empty_nodes(graph):
    retriever = GraphRetriever(graph)

    context = retriever.format_context(["DamageType:Fire", "DamageType:Acid"])

    assert context == (
        "## [DamageType] DamageType > Fire\nRelationships:\n- <- DEALS_DAMAGE: Fireball"
    )


# Hybrid seeds -----------------------------------------------------------------------

# Toy embedding space: axis 0 is "paralysis", axis 1 is "fire".
NODE_EMBEDDINGS = {
    ("Spell", "Hold Person"): [1.0, 0.0],
    ("Condition", "Paralyzed [Condition]"): [0.9, 0.1],
    ("Spell", "Fireball"): [0.0, 1.0],
}


@pytest.fixture(scope="module")
def embedded_graph(graph):
    embedded = graph.copy()
    for (label, header), embedding in NODE_EMBEDDINGS.items():
        embedded.nodes[_node(embedded, label, header)]["embedding"] = embedding

    return embedded


def _embed_query(query: str) -> list[float]:
    return [1.0, 0.0] if "freeze" in query else [0.0, 1.0]


def _headers(graph, node_ids):
    return [graph.nodes[node_id]["header"] for node_id in node_ids]


def test_default_mode_is_hybrid_only_with_embeddings_and_embedder(graph, embedded_graph):
    assert GraphRetriever(graph).default_mode == "bm25"
    assert GraphRetriever(embedded_graph).default_mode == "bm25"
    assert GraphRetriever(embedded_graph, _embed_query).default_mode == "hybrid"


def test_dense_ranking_finds_a_paraphrase_bm25_misses(embedded_graph):
    retriever = GraphRetriever(embedded_graph, _embed_query)
    query = "What can freeze someone in place?"

    assert retriever.rank(query, "bm25") == {}
    assert _headers(embedded_graph, retriever.rank(query, "dense"))[0] == "Hold Person"
    assert (
        _headers(embedded_graph, retriever.retrieve(query, num_seeds=1))[0] == "Hold Person"
    )


def test_hybrid_ranks_a_node_found_by_both_rankings_first(embedded_graph):
    retriever = GraphRetriever(embedded_graph, _embed_query)

    scores = retriever.rank("What does the Fireball spell do?", "hybrid")

    # Fireball is first for BM25 and for the "fire" query embedding.
    assert _headers(embedded_graph, scores)[0] == "Fireball"
    assert next(iter(scores.values())) == 1.0
    assert list(scores.values()) == sorted(scores.values(), reverse=True)


def test_dense_mode_without_embeddings_raises(graph):
    with pytest.raises(ValueError, match="node embeddings"):
        GraphRetriever(graph).rank("Fireball", "dense")


# Evaluation -------------------------------------------------------------------------


def test_gold_index_maps_a_reformatted_passage_to_its_node(graph):
    # The dataset's passages come from another extraction, with different markup.
    passage = "Each creature in a sphere makes a **Dexterity** saving throw,\ntaking 8d6 Fire"

    assert GoldIndex(graph).find(passage) == {_node(graph, "Spell", "Fireball")}


def test_gold_index_returns_nothing_for_unknown_or_tiny_passages(graph):
    index = GoldIndex(graph)

    assert index.find("A passage about starships and warp drives that is not there") == set()
    assert index.find("Fire damage") == set()


def test_score_reports_recall_mrr_and_context_recall():
    result = score(ranked=["a", "b", "c"], context=["a", "x"], gold={"c"}, cutoffs=[1, 4])

    assert result == {"recall@1": 0.0, "recall@4": 1.0, "mrr": 1 / 3, "context_recall": 0.0}
    assert score(["a"], ["a", "c"], {"c"}, [1])["context_recall"] == 1.0


# Neo4j ------------------------------------------------------------------------------


def test_identifier_rejects_text_that_is_not_a_label():
    from src.graphrag.store import _identifier

    assert _identifier("IMMUNE_TO_DAMAGE") == "IMMUNE_TO_DAMAGE"
    with pytest.raises(ValueError):
        _identifier("Spell) DETACH DELETE (n")


@pytest.mark.skipif(not os.getenv("NEO4J_PASSWORD"), reason="Needs a running Neo4j")
def test_graph_round_trips_through_neo4j(embedded_graph):
    from src.graphrag.store import connect
    from src.graphrag.store import load_graph
    from src.graphrag.store import save_graph

    # A label of its own keeps the test away from the real SRD graph.
    with connect() as driver:
        save_graph(embedded_graph, driver, node_label="TestSrdNode")
        loaded = load_graph(driver, node_label="TestSrdNode")
        save_graph(nx.DiGraph(), driver, node_label="TestSrdNode")

    assert dict(loaded.nodes(data=True)) == dict(embedded_graph.nodes(data=True))
    assert {(s, t, d["type"]) for s, t, d in loaded.edges(data=True)} == {
        (s, t, d["type"]) for s, t, d in embedded_graph.edges(data=True)
    }
