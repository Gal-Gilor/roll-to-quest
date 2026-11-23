"""Rank SRD sections for questions and score the rankings.

A question counts as answered at k when its gold section is among the first k distinct
sections. Relevance is by section, not passage, because the two training sets were cut
from different chunkings and a long section can hold several passages.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

CUTOFFS = (1, 5, 10)


def rank_sections(
    scores: np.ndarray, sections: list[str], depth: int = max(CUTOFFS)
) -> list[list[str]]:
    """Turn a query-by-passage score matrix into ranked lists of distinct sections."""
    rankings = []
    for row in scores:
        ranked: list[str] = []
        for index in np.argsort(-row, kind="stable"):
            if sections[index] not in ranked:
                ranked.append(sections[index])
                if len(ranked) == depth:
                    break
        rankings.append(ranked)

    return rankings


def score_rankings(rankings: list[list[str]], gold: list[str]) -> dict[str, float]:
    """Return recall@k for each cutoff and MRR@10, averaged over questions."""
    if not gold:
        return {}
    hits = [
        ranking.index(answer) + 1 if answer in ranking else None
        for ranking, answer in zip(rankings, gold)
    ]
    result = {
        f"recall@{k}": sum(rank is not None and rank <= k for rank in hits) / len(hits)
        for k in CUTOFFS
    }
    result["mrr@10"] = sum(
        1 / rank for rank in hits if rank is not None and rank <= 10
    ) / len(hits)

    return result


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(queries: list[str], passages: list[str]) -> np.ndarray:
    bm25 = BM25Okapi([_tokens(passage) for passage in passages])

    return np.array([bm25.get_scores(_tokens(query)) for query in queries])


def embedding_scores(
    model, queries: list[str], passages: list[str], batch_size: int = 64
) -> np.ndarray:
    """Cosine similarity between every query and every passage with a SentenceTransformer."""
    encode = dict(batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    passage_vectors = model.encode(passages, **encode)
    query_vectors = model.encode(queries, **encode)

    return query_vectors @ passage_vectors.T
