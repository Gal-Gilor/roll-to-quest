# Judge question-passage pairs for a D&D rules search engine

You are checking training data for a search engine over the Dungeons & Dragons 5E rules (SRD 5.2.1). Each pair is a question a player might type and the passage that should be found for it.

## Input

A JSONL file. Each line has `row`, `question_type`, `anchor` (the question) and `positive` (the passage; its first line is the path of headers above it).

## What to judge

Read the question first, as someone who cannot see the passage. Then read the passage. Reject the pair if any of these is true:

1. `not_answered`: the passage does not contain the answer to the question.
2. `not_self_contained`: the question only makes sense with the passage in view. It says "this spell" or "the creature", or leaves out what it is asking about.
3. `ambiguous`: many other passages in the D&D rules would answer the question equally well, so this passage is not the clear target. A `descriptive` question does not name its subject on purpose; accept it when the description points to this passage, and reject it when it is so generic that it could describe many spells, monsters or items.
4. `unnatural`: nobody would ask this. It is not a real question, or it copies a sentence of the passage and turns it into a question.

Be strict about `not_answered` and `not_self_contained`. Do not reject a pair for style alone.

## Output

Write a JSONL file with one line per input row, with exactly these keys:

```json
{"row": 0, "verdict": "accept", "reason": ""}
{"row": 1, "verdict": "reject", "reason": "ambiguous"}
```

`verdict` is "accept" or "reject". `reason` is empty for accepted pairs and one of the four codes above for rejected pairs. No other text in the file. Judge every row.
