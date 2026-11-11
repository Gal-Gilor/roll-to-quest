---
license: cc-by-4.0
language:
  - en
task_categories:
  - text-retrieval
  - sentence-similarity
tags:
  - dnd
  - dungeons-and-dragons
  - srd
pretty_name: "D&D 5e SRD Question-Passage Pairs (graph nodes)"
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: validation.jsonl
      - split: test
        path: test.jsonl
---

# D&D 5e SRD Question-Passage Pairs (graph nodes)

Question-passage pairs from the Dungeons & Dragons 5th Edition System Reference Document v5.2.1, for training and evaluating embedding models with MultipleNegativesRankingLoss.

The passages come from the nodes of the Roll-to-Quest knowledge graph, where a node is a whole entity such as a spell, a monster stat block or a magic item. Nodes longer than about 400 tokens are split into passages on paragraph boundaries, and tables keep their header rows in every piece.

## Size

| Split | Pairs | Nodes |
|-------|-------|-------|
| train | 5,680 | 1,589 |
| validation | 341 | 93 |
| test | 658 | 187 |

The split is by node. Every pair of a node is in the same split, so the test questions are about entities the model saw no question for.

## Fields

- `anchor`: the question.
- `positive`: the passage that answers it. The first line is the path of headers above the passage.
- `node_id`: the graph node the passage belongs to. Node ids are positions in the SRD Markdown and change if the source file changes.
- `passage_id`: the node id and the passage's index within the node.
- `label`: the node's entity type (Spell, Monster, MagicItem, Feature, Section and others).
- `breadcrumb`: the path of headers above the passage.
- `question_type`: one of the four types below.
- `bm25_rank`: the rank BM25 gives the question's own node among all nodes, capped at 51. A rank above 4 marks a hard pair, one a keyword search gets wrong. 695 pairs are hard.

## Question types

| Type | Pairs | What it is |
|------|-------|------------|
| `lookup` | 2,165 | A direct question about one fact. Names the entity. |
| `paraphrase` | 2,021 | A question in a player's words, avoiding the passage's wording. Names the entity. |
| `descriptive` | 1,882 | A question from someone who doesn't know the entity's name. Does not name it. |
| `table` | 611 | A question whose answer is a cell of a table in the passage. |

## How it was made

1. Passages were exported from the graph and near-duplicate passages were dropped, because MNLR uses the other positives in a batch as negatives.
2. Claude Haiku 5.5 wrote three questions per passage, and a fourth for passages with a table.
3. Scripted checks removed 442 of 7,199 questions: those that copy more than 75% of their content words from the passage, questions that refer to "this spell" or "the passage", questions that break the naming rule of their type, and duplicates.
4. BM25 ranked the nodes for each remaining question. The 773 questions whose own node was not in the top 4 were judged by Claude Sonnet 5.5, which saw the intended passage and the passage of another node that BM25 ranked highest. It accepted 695, rejected 69 as ambiguous or unanswered, and marked 9 unsure. The 9 were checked against the SRD by Claude Fable 5.2 and rejected: each fit several entries, such as a trait shared by seven devils.

The 5,984 pairs BM25 ranks in the top 4 were not judged, and no person reviewed every pair.

## Training notes

- Several pairs share a `positive`. Use a batch sampler that keeps duplicates of a positive out of a batch, such as `BatchSamplers.NO_DUPLICATES` in Sentence Transformers.
- The questions were written by a model from the same passages they are paired with. Scores on the test split show how well a model retrieves SRD passages, not how well it generalizes to other rule books.

## License and attribution

This work includes material from the System Reference Document 5.2.1 ("SRD 5.2.1") by Wizards of the Coast LLC, available at https://www.dndbeyond.com/srd. The SRD 5.2.1 is licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode.
