from textwrap import dedent

import pytest

from src.graphrag.graph import CONTAINS
from src.graphrag.graph import MENTIONS
from src.graphrag.graph import build_graph
from src.graphrag.retrieval import GraphRetriever

MARKDOWN = dedent(
    """\
    # Spells
    ## Fireball
    A bright streak explodes in a Sphere. Creatures may be knocked Prone.
    ## Shield
    An invisible barrier protects you.
    # Glossary
    ## Sphere [Area of Effect]
    A Sphere extends from a point of origin.
    ## Prone [Condition]
    Your only movement option is to crawl.
    # Monsters
    ## Mage
    ### Actions
    The mage casts Fireball.
    ## Ghoul
    ### Actions
    The ghoul attacks with its claws.
    """
)


@pytest.fixture
def graph(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(MARKDOWN, encoding="utf-8")

    return build_graph(path)


def _node(graph, breadcrumb):
    return next(n for n, data in graph.nodes(data=True) if data["breadcrumb"] == breadcrumb)


def _edge_type(graph, source, target):
    return graph.edges[_node(graph, source), _node(graph, target)]["type"]


def test_graph_links_sections_to_subsections(graph):
    assert graph.number_of_nodes() == 11
    assert _edge_type(graph, "Spells", "Spells > Fireball") == CONTAINS
    assert graph.nodes[_node(graph, "Spells > Fireball")]["category"] == "Spells"


def test_graph_links_mentions_and_strips_header_tags(graph):
    fireball = "Spells > Fireball"

    assert _edge_type(graph, fireball, "Glossary > Sphere [Area of Effect]") == MENTIONS
    assert _edge_type(graph, fireball, "Glossary > Prone [Condition]") == MENTIONS
    assert _edge_type(graph, "Monsters > Mage > Actions", fireball) == MENTIONS


def test_graph_skips_ambiguous_headers_and_self_mentions(graph):
    sphere = _node(graph, "Glossary > Sphere [Area of Effect]")

    assert not graph.has_edge(sphere, sphere)
    # "Actions" names two sections, so no section can be linked to it by name.
    assert all(
        data["type"] == CONTAINS
        for _, target, data in graph.edges(data=True)
        if graph.nodes[target]["header"] == "Actions"
    )


def test_retrieve_returns_seed_then_graph_neighbors(graph):
    retriever = GraphRetriever(graph)

    node_ids = retriever.retrieve("What does Fireball do?", num_seeds=1)
    breadcrumbs = [graph.nodes[node_id]["breadcrumb"] for node_id in node_ids]

    assert breadcrumbs[0] == "Spells > Fireball"
    # Neither is named in the question; both come from the graph.
    assert "Glossary > Prone [Condition]" in breadcrumbs
    assert "Monsters > Mage > Actions" in breadcrumbs
    assert "Monsters > Ghoul > Actions" not in breadcrumbs


def test_retrieve_returns_nothing_for_unknown_terms(graph):
    assert GraphRetriever(graph).retrieve("xyzzy") == []


def test_format_context_skips_empty_sections(graph):
    retriever = GraphRetriever(graph)

    context = retriever.format_context(
        [_node(graph, "Spells"), _node(graph, "Spells > Shield")]
    )

    assert context == "## Spells > Shield\nAn invisible barrier protects you."
