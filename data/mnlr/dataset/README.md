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
pretty_name: "D&D 5e SRD Question-Passage Pairs"
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

# D&D 5e SRD Question-Passage Pairs

Question-passage pairs from the Dungeons & Dragons 5th Edition System Reference Document v5.2.1, for training and evaluating embedding models with MultipleNegativesRankingLoss.

The passages come from the SRD Markdown, split into sections by header with [markdown-chunkify](https://github.com/Gal-Gilor/markdown-chunkify). An entry keeps its subsections, so a monster's passage holds its whole stat block with its traits and actions. Entries longer than about 400 tokens are split into passages on paragraph boundaries, and tables keep their header rows in every piece.

## Size

| Split | Pairs | Sections |
|-------|-------|----------|
| train | 7,892 | 1,594 |
| validation | 448 | 94 |
| test | 926 | 188 |

The split is by SRD section: every pair whose passage comes from the same section is in the same split, so the test questions are about entries the model saw no question for. The section is the `breadcrumb` field.

## Fields

- `anchor`: the question.
- `positive`: the passage that answers it. The first line is the path of headers above the passage.
- `breadcrumb`: the path of headers above the passage, such as `Spells > Spell Descriptions > Fireball`. It names the SRD section the passage comes from, and the split is grouped by it.
- `question_type`: one of the four types below.

## Question types

| Type | Pairs | What it is |
|------|-------|------------|
| `lookup` | 2,165 | A direct question about one fact. Names the entity. |
| `paraphrase` | 2,021 | A question in a player's words, avoiding the passage's wording. Names the entity. |
| `descriptive` | 4,469 | A question from someone who doesn't know the entity's name. Does not name it. |
| `table` | 611 | A question whose answer is a cell of a table in the passage. |

## How it was made

1. Passages were cut from the SRD sections and near-duplicate passages were dropped, because MNLR uses the other positives in a batch as negatives.
2. Claude Haiku 5.5 wrote three questions per passage, and a fourth for passages with a table.
3. In a second pass over the 1,444 passages of named entries (spells, monsters, magic items, class features and the like), Claude Haiku 5.5 wrote two more descriptive questions per passage, with instructions to avoid the passage's wording and to combine several facts so the question points to one entry.
4. Scripted checks removed 476 of the 9,976 questions: those that copy more than 75% of their content words from the passage, questions that refer to "this spell" or "the passage", questions that break the naming rule of their type, and duplicates.
5. BM25 keyword search ranked the SRD entries for each remaining question. The 2,030 questions whose own entry was not in the top 4 were judged by Claude Sonnet 5.5, which saw the intended passage and the passage of the other entry that BM25 ranked highest. It accepted 1,796, rejected 166 as ambiguous or unanswered, and marked 68 unsure. The unsure ones were rejected: in most, the judge named a near-identical entry the question also fits, such as a trait shared by seven devils or by two ages of the same dragon.

The 7,470 pairs BM25 ranks in the top 4 were not judged, and no person reviewed every pair.

## Training notes

- Several pairs share a `positive`. Use a batch sampler that keeps duplicates of a positive out of a batch, such as `BatchSamplers.NO_DUPLICATES` in Sentence Transformers.
- The questions were written by a model from the same passages they are paired with. Scores on the test split show how well a model retrieves SRD passages, not how well it generalizes to other rule books.

## License and attribution

This work includes material from the System Reference Document 5.2.1 ("SRD 5.2.1") by Wizards of the Coast LLC, available at https://www.dndbeyond.com/srd. The SRD 5.2.1 is licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode.
