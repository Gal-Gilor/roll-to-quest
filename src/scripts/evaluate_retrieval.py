"""Measure retrieval on the D&D SRD anchor-positive pairs dataset.

Each pair is a question and the SRD passage that answers it. The passage is mapped to
the graph nodes that contain it (see `src.graphrag.evaluation`), and a question counts
as answered when one of those nodes is retrieved. Pairs whose passage maps to no node
are skipped and counted.

Reported per seed mode:
    - recall@k: the answer node is among the top k seed candidates.
    - mrr: mean reciprocal rank of the answer node among the top candidates.
    - context_recall: the answer node is among the nodes the chatbot sends to the
      model, which are the seeds plus their graph neighbors.

By default the pairs come from the published dataset, which was generated from a
different chunking. With --pairs they come from a local JSONL file whose rows carry a
`node_id`, such as data/mnlr/dataset/test.jsonl, and no passage matching is needed.

Without --neo4j the graph is built in memory and only BM25 is measured. With --neo4j
the graph and node embeddings come from Neo4j (run `build_graph_db` first) and dense
and hybrid seeds are measured too. Each question is then embedded with one Vertex AI
call, so use --sample to bound the cost.

Example Usage:
    python -m src.scripts.evaluate_retrieval
    python -m src.scripts.evaluate_retrieval --neo4j --sample 1000
    python -m src.scripts.evaluate_retrieval --sample 0   # All pairs
    python -m src.scripts.evaluate_retrieval --pairs data/mnlr/dataset/test.jsonl --sample 0
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.graphrag import DEFAULT_MARKDOWN
from src.graphrag import create_retriever
from src.graphrag.evaluation import GoldIndex
from src.graphrag.evaluation import score
from src.graphrag.vertex import create_client

DATASET = "galgilor/dnd-5e-srd-anchor-positive-pairs"
REVISION = "27dfa295e70a42728376955ed792aac756dc58a8"
CUTOFFS = [1, 4, 10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure retrieval on the pairs dataset.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--neo4j", action="store_true", help="Load the graph from Neo4j.")
    parser.add_argument("--pairs", type=Path, help="Local JSONL pairs with a node_id field.")
    parser.add_argument(
        "--sample", type=int, default=2000, help="Pairs to evaluate. 0 evaluates all."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--num-neighbors", type=int, default=8)
    args = parser.parse_args()

    retriever = create_retriever(create_client(), args.markdown, from_neo4j=args.neo4j)
    modes = ["bm25", "dense", "hybrid"] if retriever.default_mode == "hybrid" else ["bm25"]

    if args.pairs:
        lines = args.pairs.read_text(encoding="utf-8").splitlines()
        pairs = [json.loads(line) for line in lines]
    else:
        pairs = list(load_dataset(DATASET, revision=REVISION, split="train"))
    if args.sample:
        pairs = random.Random(args.seed).sample(pairs, min(args.sample, len(pairs)))

    gold_index = GoldIndex(retriever.graph)
    totals: dict[str, dict[str, float]] = {mode: defaultdict(float) for mode in modes}
    evaluated = 0
    for pair in tqdm(pairs, desc="Evaluating"):
        gold = {pair["node_id"]} if "node_id" in pair else gold_index.find(pair["positive"])
        if not gold:
            continue
        evaluated += 1
        for mode in modes:
            ranked = list(retriever.rank(pair["anchor"], mode))[: max(CUTOFFS)]
            context = retriever.retrieve(
                pair["anchor"], args.num_seeds, args.num_neighbors, mode
            )
            for metric, value in score(ranked, context, gold, CUTOFFS).items():
                totals[mode][metric] += value

    print(f"\nEvaluated {evaluated:,} of {len(pairs):,} pairs.")
    print(f"Skipped {len(pairs) - evaluated:,} whose passage matched no graph node.")
    if not evaluated:
        return

    metrics = list(totals[modes[0]])
    print(f"\n{'mode':<8}" + "".join(f"{metric:>16}" for metric in metrics))
    for mode in modes:
        row = "".join(f"{totals[mode][metric] / evaluated:>16.3f}" for metric in metrics)
        print(f"{mode:<8}{row}")


if __name__ == "__main__":
    main()
