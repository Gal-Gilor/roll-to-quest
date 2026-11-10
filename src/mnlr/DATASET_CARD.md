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
| train | 5,595 | 1,589 |
| validation | 318 | 94 |
| test | 583 | 187 |

The split is by node. Every pair of a node is in the same split, so the test questions are about entities the model saw no question for.

## Fields

- `anchor`: the question.
- `positive`: the passage that answers it. The first line is the path of headers above the passage.
- `node_id`: the graph node the passage belongs to. Node ids are positions in the SRD Markdown and change if the source file changes.
- `passage_id`: the node id and the passage's index within the node.
- `label`: the node's entity type (Spell, Monster, MagicItem, Feature, Section and others).
- `breadcrumb`: the path of headers above the passage.
- `question_type`: one of the four types below.

## Question types

| Type | Pairs | What it is |
|------|-------|------------|
| `lookup` | 2,171 | A direct question about one fact. Names the entity. |
| `paraphrase` | 2,044 | A question in a player's words, avoiding the passage's wording. Names the entity. |
| `descriptive` | 1,668 | A question from someone who doesn't know the entity's name. Does not name it. |
| `table` | 613 | A question whose answer is a cell of a table in the passage. |

## How it was made

1. Passages were exported from the graph and near-duplicate passages were dropped, because MNLR uses the other positives in a batch as negatives.
2. Claude Haiku 5.5 wrote three questions per passage, and a fourth for passages with a table.
3. Scripted checks removed 703 of 7,199 questions: those that copy more than 75% of their content words from the passage, descriptive questions that BM25 matches to another node by a clear margin, questions that refer to "this spell" or "the passage", questions that break the naming rule of their type, and duplicates.

No person reviewed every pair. A sample read by the author found some descriptive questions that fit more than one passage.

## Training notes

- Several pairs share a `positive`. Use a batch sampler that keeps duplicates of a positive out of a batch, such as `BatchSamplers.NO_DUPLICATES` in Sentence Transformers.
- The questions were written by a model from the same passages they are paired with. Scores on the test split show how well a model retrieves SRD passages, not how well it generalizes to other rule books.

## License and attribution

This work includes material from the System Reference Document 5.2.1 ("SRD 5.2.1") by Wizards of the Coast LLC, available at https://www.dndbeyond.com/srd. The SRD 5.2.1 is licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode.
